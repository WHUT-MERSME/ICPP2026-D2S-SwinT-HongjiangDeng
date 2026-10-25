from __future__ import annotations

import inspect
import math
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def heaviside0(x: torch.Tensor):
    return (x >= 0).to(x)


def _get_hw_arg(args, name: str, default):
    return getattr(args, name, default) if args is not None else default


def _quantize_tensor(x: Tensor, bits: int = 8, symmetric: bool = True) -> Tensor:
    if x is None or bits <= 0:
        return x

    levels = max(2 ** bits - 1, 1)
    if symmetric:
        scale = x.detach().abs().amax().clamp_min(1e-6) / levels
        return torch.round(x / scale) * scale

    x_min = x.detach().amin()
    x_max = x.detach().amax()
    span = (x_max - x_min).clamp_min(1e-6)
    scale = span / levels
    q = torch.round((x - x_min) / scale) * scale + x_min
    return q


def _extract_batchnorm_affine(bn: nn.Module) -> tuple[Tensor, Tensor]:
    if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
        raise TypeError(f"Unsupported batchnorm type for affine extraction: {type(bn)}")

    device = bn.weight.device if bn.affine else bn.running_mean.device
    dtype = bn.weight.dtype if bn.affine else bn.running_mean.dtype
    gamma = bn.weight.detach() if bn.affine and bn.weight is not None else torch.ones_like(bn.running_mean, device=device, dtype=dtype)
    beta = bn.bias.detach() if bn.affine and bn.bias is not None else torch.zeros_like(bn.running_mean, device=device, dtype=dtype)
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    scale = gamma / torch.sqrt(var + bn.eps)
    bias = beta - mean * scale
    return scale, bias


def _ensure_linear_bias(linear: nn.Linear) -> nn.Parameter:
    if linear.bias is None:
        bias = torch.zeros(linear.out_features, device=linear.weight.device, dtype=linear.weight.dtype)
        linear.bias = nn.Parameter(bias)
    return linear.bias


def _ensure_conv_bias(conv: nn.Conv2d) -> nn.Parameter:
    if conv.bias is None:
        bias = torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
        conv.bias = nn.Parameter(bias)
    return conv.bias


def _fuse_post_bn_into_prev_linear(linear: nn.Linear, bn: nn.BatchNorm1d):
    scale, bias = _extract_batchnorm_affine(bn)
    with torch.no_grad():
        linear.weight.mul_(scale.view(-1, 1).to(device=linear.weight.device, dtype=linear.weight.dtype))
        linear_bias = _ensure_linear_bias(linear)
        linear_bias.copy_(linear_bias * scale.to(device=linear_bias.device, dtype=linear_bias.dtype) + bias.to(device=linear_bias.device, dtype=linear_bias.dtype))


def _fuse_post_bn_into_prev_conv(conv: nn.Conv2d, bn: nn.Module):
    scale, bias = _extract_batchnorm_affine(bn)
    with torch.no_grad():
        conv.weight.mul_(scale.view(-1, 1, 1, 1).to(device=conv.weight.device, dtype=conv.weight.dtype))
        conv_bias = _ensure_conv_bias(conv)
        conv_bias.copy_(conv_bias * scale.to(device=conv_bias.device, dtype=conv_bias.dtype) + bias.to(device=conv_bias.device, dtype=conv_bias.dtype))


def _fuse_pre_bn_into_next_linear(bn: nn.Module, linear: nn.Linear):
    scale, bias = _extract_batchnorm_affine(bn)
    with torch.no_grad():
        scale_t = scale.to(device=linear.weight.device, dtype=linear.weight.dtype)
        bias_t = bias.to(device=linear.weight.device, dtype=linear.weight.dtype)
        original_weight = linear.weight.detach().clone()
        linear.weight.mul_(scale_t.view(1, -1))
        linear_bias = _ensure_linear_bias(linear)
        linear_bias.add_(original_weight @ bias_t)


def get_data_from_place(place, p, n, args):
    """Resolve calibrated thresholds and firing levels for each ECMT insertion point."""

    sp = p
    sn = n
    gap = 2
    if "vit" in args.model and place == "fc2":
        if abs(float(sp)) < 1e-12:
            sp = 0.25
        if abs(float(sn)) < 1e-12:
            sn = 0.08
    times = args.linear_num
    if place == "q" or place == "k" or place == "v":
        times = args.qkv_num
    if place == "s":
        times = args.softmax_num
        default_softmax_p = float(getattr(args, "softmax_p", 0.0125))
        sp = max(float(sp), default_softmax_p)
        sn = max(float(sn), default_softmax_p)
        softmax_cap = 0.25
        sp = min(sp, softmax_cap)
        sn = min(sn, softmax_cap)
    return sp, sn, times, gap


def replace_testneuron_by_twosideneuron(model, args, hardware_args=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_testneuron_by_twosideneuron(module, args, hardware_args)
        if "testneuron" == module.__class__.__name__.lower():
            place = model._modules[name].place
            scale_p, scale_n, times, gap = get_data_from_place(
                place,
                float(model._modules[name].scale_p[0]),
                float(model._modules[name].scale_n[0]),
                args,
            )
            print(name, scale_p, scale_n, place)
            model._modules[name] = TwoSideNeuron(
                scale_p=scale_p,
                scale_n=scale_n,
                place=place,
                times=times,
                gap=gap,
                enable_hw=_get_hw_arg(hardware_args, "enable_neuron_hardware", True),
            )

    return model


class MyTestPlace(nn.Module):
    def __init__(self, place=None):
        super().__init__()
        self.place = place

    def forward(self, x, *args):
        return [x]


class TestNeuron(nn.Module):
    def __init__(self, place=None, percent=None):
        super().__init__()
        self.place = place
        self.percent = percent
        self.num = 0
        self.scale_p = torch.nn.Parameter(torch.FloatTensor([0.0]))
        self.scale_n = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, x, times=2, gap=3, show=0, tmptime=0, scaletimes=1):
        """Collect running percentile thresholds on calibration data."""

        x2 = x.reshape(-1)
        threshold = torch.kthvalue(x2, int(self.percent * x2.numel()), dim=0).values.item()
        self.scale_p = torch.nn.Parameter((self.scale_p * self.num + threshold) / (self.num + 1))
        threshold = -torch.kthvalue(x2, int((1 - self.percent) * x2.numel()), dim=0).values.item()
        self.scale_n = torch.nn.Parameter((self.scale_n * self.num + threshold) / (self.num + 1))
        self.num += 1
        return [x]

    def reset(self):
        pass


def replace_test_by_testneuron(model, percent=None):
    """Replace calibration placeholders with threshold collectors."""

    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_test_by_testneuron(module, percent)
        if module.__class__.__name__.lower() == "mytestplace":
            model._modules[name] = TestNeuron(place=model._modules[name].place, percent=percent)
    return model


class exp_comp_neuron(nn.Module):
    """Incremental wrapper used by the original ECMT conversion path."""

    def __init__(self, func, *args, **keywords):
        super().__init__(*args, **keywords)
        self.tot = None
        self.func = func
        self.t = 0

    def forward(self, x):
        if x is None:
            return None
        if self.tot is None:
            last = torch.zeros_like(x)
            self.tot = x.clone()
        else:
            last = self.func(self.tot / self.t) * self.t
            self.tot += x
        self.t += 1
        now = self.func(self.tot / self.t) * self.t
        return now - last

    def reset(self):
        self.tot = None
        self.t = 0


def replace_nonlinear_by_neuron(model):
    """Wrap nonlinear ANN operators with the original ECMT incremental adaptor."""

    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_nonlinear_by_neuron(module)
        module_name = module.__class__.__name__.lower()
        if (
            "softmax" in module_name
            or "relu" in module_name
            or "layernorm" in module_name
            or "batchnorm" in module_name
        ):
            model._modules[name] = exp_comp_neuron(func=model._modules[name])
    return model


class MyAt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x @ y


def load_nv_mapping_from_csv(file_path):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    pulse_col = df.columns[0]
    conductance_col = df.columns[1]
    return {
        int(row[pulse_col]): float(row[conductance_col])
        for _, row in df.iterrows()
    }


def load_v_mapping_from_csv(file_path):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    state_col = df.columns[0]
    conductance_col = df.columns[1]
    df[state_col] = df[state_col].astype(str).str.strip().str.zfill(5)
    return {
        row[state_col]: float(row[conductance_col])
        for _, row in df.iterrows()
    }


class AtNeuron(nn.Module):
    """
    Hardware-equivalent AtNeuron based on the existing ECMT accumulator semantics.
    """

    def __init__(self, nv_map=None, enable_hw: bool = True):
        super().__init__()
        self.enable_hw = bool(enable_hw)
        self.G_nv = nv_map if nv_map else load_nv_mapping_from_csv("nv_table.csv")

        pulses = sorted(int(k) for k in self.G_nv.keys())
        self.pulse_min = pulses[0]
        self.pulse_max = pulses[-1]

        lut = torch.zeros(self.pulse_max + 1, dtype=torch.float32)
        for k, v in self.G_nv.items():
            kk = int(k)
            if 0 <= kk <= self.pulse_max:
                lut[kk] = float(v)
        self.register_buffer("G_lut", lut)

        gmax = lut[self.pulse_max].item() if self.pulse_max >= 0 else 1.0
        self.register_buffer("G_max", torch.tensor(gmax if gmax > 0 else 1.0, dtype=torch.float32))

        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        self.t = 0

        self.register_buffer("W_max", torch.tensor(0.0, dtype=torch.float32))
        self._wmax_inited = False

    def reset(self):
        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        self.t = 0

    def _init_wmax(self, y: Tensor):
        wmax = y.detach().abs().max().float().clamp_min(1e-6)
        self.W_max.copy_(wmax)
        self._wmax_inited = True

    def _weight_to_pulse_abs(self, y: Tensor) -> Tensor:
        if not self._wmax_inited:
            self._init_wmax(y)

        a = (y.abs().float() / self.W_max).clamp(0, 1)
        pulses = torch.round(a * (self.pulse_max - self.pulse_min) + self.pulse_min).to(torch.long)
        pulses = torch.clamp(pulses, self.pulse_min, self.pulse_max)
        return pulses

    def forward(self, x: Tensor, y: Tensor):
        if self.enable_hw:
            pulses = self._weight_to_pulse_abs(y)
            g = self.G_lut.to(device=pulses.device)[pulses].to(x.dtype)

            pos_mask = (y >= 0).to(x.dtype)
            neg_mask = (y < 0).to(x.dtype)
            g_pos = g * pos_mask
            g_neg = g * neg_mask
            g_eff = g_pos - g_neg

            scale = self.W_max.to(x.dtype) / (self.G_max.to(x.dtype) + 1e-12)
            w_eff = g_eff * scale
        else:
            w_eff = y.to(dtype=x.dtype)

        i_xy = x @ w_eff

        if self.t == 0:
            self.tot_a = x
            self.tot_b = w_eff
            self.tot_t = i_xy
            self.t = 1
            return i_xy

        self.tot_t = self.tot_t + i_xy + (x @ self.tot_b) + (self.tot_a @ w_eff)
        self.tot_a = self.tot_a + x
        self.tot_b = self.tot_b + w_eff
        kstep = self.t
        delta = (x @ self.tot_b + self.tot_a @ w_eff - i_xy) / kstep - self.tot_t / (kstep * (kstep + 1))
        self.t += 1
        return delta


class TwoSideNeuron(nn.Module):
    """
    Volatile threshold-switching device model that preserves the original ECMT
    firing semantics while allowing optional hardware discharge modulation.
    """

    def __init__(
        self,
        scale_p: float = 1.0,
        scale_n: float = 1.0,
        place=None,
        times: int = 3,
        gap: float = 2.0,
        v_map=None,
        enable_hw: bool = True,
        hw_strength: float = 1.0,
    ):
        super().__init__()
        self.scale_p = float(scale_p)
        self.scale_n = float(scale_n)
        self.place = place
        self.times = int(times)
        self.gap = float(gap)

        self.enable_hw = bool(enable_hw)
        self.hw_strength = float(hw_strength)

        self.t = 0
        self.neuron = None

        self.register_buffer("dummy_tensor", torch.tensor(0.0))

        self.G_v_map = v_map if v_map else load_v_mapping_from_csv("v_table.csv")
        states_int = sorted(int(k) for k in self.G_v_map.keys())
        self.state_min = states_int[0]
        self.state_max = states_int[-1]

        lut = torch.zeros(self.state_max + 1, dtype=torch.float32)
        for k, v in self.G_v_map.items():
            lut[int(k)] = float(v)
        self.register_buffer("G_lut", lut)

        g_nonzero = lut[lut > 0]
        g_ref = g_nonzero.median() if g_nonzero.numel() > 0 else torch.tensor(1.0)
        self.register_buffer("G_ref", g_ref)

        level_states = []
        if self.times <= 1:
            level_states = [self.state_max]
        else:
            for lvl in range(self.times):
                frac = (self.times - 1 - lvl) / (self.times - 1)
                state = int(round(self.state_min + frac * (self.state_max - self.state_min)))
                state = max(self.state_min, min(self.state_max, state))
                level_states.append(state)
        self.register_buffer("level_states", torch.tensor(level_states, dtype=torch.long))

    def reset(self):
        self.t = 0
        self.neuron = None

    def _hw_scale_for_level(self, lvl: int, dtype: torch.dtype) -> Tensor:
        if not self.enable_hw or self.hw_strength <= 0:
            return torch.tensor(1.0, device=self.dummy_tensor.device, dtype=dtype)

        state = int(self.level_states[lvl].item())
        g_on = self.G_lut[state].to(dtype=dtype)
        scale = g_on / (self.G_ref.to(dtype=dtype) + 1e-12)
        return (1.0 - self.hw_strength) + self.hw_strength * scale

    def forward(self, x: Tensor):
        x = x.to(self.dummy_tensor.device)

        if self.t == 0 or self.neuron is None or self.neuron.shape != x.shape:
            self.neuron = torch.zeros_like(x)
        self.neuron = self.neuron + x

        v_threshold1 = self.scale_p
        v_threshold2 = self.scale_n
        for _ in range(self.times - 1):
            v_threshold1 *= self.gap
            v_threshold2 *= self.gap

        fire_list = []
        not_fire = torch.ones_like(x)

        for lvl in range(self.times):
            tmp1 = heaviside0(self.neuron - v_threshold1 + self.scale_p / 2)
            tmp2 = heaviside0(-self.neuron - v_threshold2 + self.scale_n / 2)

            tmp1 = tmp1 * v_threshold1
            tmp2 = tmp2 * (-v_threshold2)

            tmp1 = tmp1 * not_fire
            tmp2 = tmp2 * not_fire
            not_fire[tmp1 != 0] = 0
            not_fire[tmp2 != 0] = 0

            if self.enable_hw and self.hw_strength > 0:
                scale = self._hw_scale_for_level(lvl, x.dtype)
                tmp1 = tmp1 * scale
                tmp2 = tmp2 * scale

            self.neuron = self.neuron - tmp1
            self.neuron = self.neuron - tmp2

            fire_list.append(tmp1)
            fire_list.append(tmp2)

            v_threshold1 /= self.gap
            v_threshold2 /= self.gap

        self.t += 1
        return fire_list


class HardwareIncrementalOp(nn.Module):
    """Base class for incremental hardware-friendly nonlinear operators."""

    def __init__(self, func: nn.Module, compat_mode: bool = False, hw_strength: float = 1.0, quant_bits: int = 8):
        super().__init__()
        self.func = func
        self.compat_mode = bool(compat_mode)
        self.hw_strength = float(hw_strength)
        self.quant_bits = int(quant_bits)
        self.tot = None
        self.t = 0

    def reset(self):
        self.tot = None
        self.t = 0

    def _apply_hardware_effect(self, y: Tensor) -> Tensor:
        if y is None or self.compat_mode or self.hw_strength <= 0:
            return y

        quantized = _quantize_tensor(y, self.quant_bits, symmetric=self._use_symmetric_quant())
        return torch.lerp(y, quantized, self.hw_strength)

    def _use_symmetric_quant(self) -> bool:
        return True

    def _run_func(self, x: Tensor) -> Tensor:
        return self._apply_hardware_effect(self.func(x))

    def forward(self, x):
        if x is None:
            return None

        if self.tot is None:
            last = torch.zeros_like(x)
            self.tot = x.clone()
        else:
            last = self._run_func(self.tot / self.t) * self.t
            self.tot = self.tot + x

        self.t += 1
        now = self._run_func(self.tot / self.t) * self.t
        return now - last


class HardwareSoftmax(HardwareIncrementalOp):
    """Softmax with a base-2 exponential approximation for the hardware path."""

    def _use_symmetric_quant(self) -> bool:
        return False

    def _run_func(self, x: Tensor) -> Tensor:
        soft_y = self.func(x)
        if self.compat_mode or self.hw_strength <= 0:
            return soft_y

        shifted = x - x.max(dim=-1, keepdim=True).values
        scaled = (shifted / math.log(2.0)).clamp(min=-16.0, max=0.0)
        integer = torch.floor(scaled)
        frac = scaled - integer

        exp_frac = 1.0 + frac
        exp_approx = exp_frac * torch.pow(torch.full_like(exp_frac, 2.0), integer)
        exp_approx = exp_approx.clamp_min(1e-12)

        denom = exp_approx.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        hw_y = exp_approx / denom
        return torch.lerp(soft_y, hw_y, self.hw_strength)


class HardwareReLU(HardwareIncrementalOp):
    """ReLU with a simple half-wave rectifier style approximation."""

    def _use_symmetric_quant(self) -> bool:
        return False

    def _run_func(self, x: Tensor) -> Tensor:
        soft_y = self.func(x)
        if self.compat_mode or self.hw_strength <= 0:
            return soft_y

        hw_y = torch.clamp_min(x, 0.0)
        hw_y = _quantize_tensor(hw_y, self.quant_bits, symmetric=False)
        return torch.lerp(soft_y, hw_y, self.hw_strength)


class HardwareBatchNorm(HardwareIncrementalOp):
    """BatchNorm executed as a fixed affine transform during inference."""

    def __init__(self, func: nn.Module, compat_mode: bool = False, hw_strength: float = 1.0, quant_bits: int = 8):
        super().__init__(func, compat_mode=compat_mode, hw_strength=hw_strength, quant_bits=quant_bits)
        scale, bias = _extract_batchnorm_affine(func)
        self.register_buffer("bn_scale", scale.float())
        self.register_buffer("bn_bias", bias.float())

    def _run_func(self, x: Tensor) -> Tensor:
        if self.compat_mode or self.hw_strength <= 0:
            return self.func(x)

        view_shape = [1, -1] + [1] * (x.ndim - 2)
        scale = self.bn_scale.to(device=x.device, dtype=x.dtype).view(*view_shape)
        bias = self.bn_bias.to(device=x.device, dtype=x.dtype).view(*view_shape)
        y = x * scale + bias
        return y


class HardwareNorm(HardwareIncrementalOp):
    """LayerNorm with fixed calibration statistics collected from calibration data."""

    def __init__(self, func: nn.LayerNorm, compat_mode: bool = False, hw_strength: float = 1.0, quant_bits: int = 8):
        super().__init__(func, compat_mode=compat_mode, hw_strength=hw_strength, quant_bits=quant_bits)
        self.eps = float(func.eps)
        self.normalized_shape = tuple(func.normalized_shape)
        if func.elementwise_affine:
            self.register_buffer("ln_weight", func.weight.detach().clone().float())
            self.register_buffer("ln_bias", func.bias.detach().clone().float())
        else:
            self.register_buffer("ln_weight", torch.ones(self.normalized_shape, dtype=torch.float32))
            self.register_buffer("ln_bias", torch.zeros(self.normalized_shape, dtype=torch.float32))
        self.register_buffer("mu_avg", torch.zeros(self.normalized_shape, dtype=torch.float32))
        self.register_buffer("sigma_avg", torch.ones(self.normalized_shape, dtype=torch.float32))
        self.register_buffer("calib_count", torch.tensor(0.0, dtype=torch.float32))

    def update_calibration(self, x: Tensor):
        if x is None:
            return

        x_float = x.detach().float()
        reduce_dims = tuple(range(max(0, x.ndim - len(self.normalized_shape))))
        if len(reduce_dims) == 0:
            batch_mu = x_float
            batch_sigma = torch.zeros_like(x_float) + 1.0
            batch_count = 1.0
        else:
            batch_mu = x_float.mean(dim=reduce_dims)
            batch_sigma = x_float.std(dim=reduce_dims, unbiased=False).clamp_min(1e-6)
            batch_count = float(math.prod([x_float.shape[d] for d in reduce_dims]))

        prev_count = float(self.calib_count.item())
        new_count = prev_count + batch_count

        self.mu_avg.copy_((self.mu_avg * prev_count + batch_mu * batch_count) / max(new_count, 1.0))
        self.sigma_avg.copy_((self.sigma_avg * prev_count + batch_sigma * batch_count) / max(new_count, 1.0))
        self.calib_count.fill_(new_count)

    def _run_func(self, x: Tensor) -> Tensor:
        soft_y = self.func(x)
        if self.compat_mode or self.hw_strength <= 0:
            return soft_y

        view_shape = [1] * (x.ndim - len(self.normalized_shape)) + list(self.normalized_shape)
        weight = self.ln_weight.to(device=x.device, dtype=x.dtype).view(*view_shape)
        bias = self.ln_bias.to(device=x.device, dtype=x.dtype).view(*view_shape)
        mu = self.mu_avg.to(device=x.device, dtype=x.dtype).view(*view_shape)
        sigma = self.sigma_avg.to(device=x.device, dtype=x.dtype).clamp_min(1e-6).view(*view_shape)
        hw_y = ((x - mu) / sigma) * weight + bias
        return torch.lerp(soft_y, hw_y, self.hw_strength)


class HardwareLinear(nn.Module):
    """Linear layer mapped to a nonvolatile conductance lookup table."""

    def __init__(
        self,
        linear: nn.Linear,
        nv_map: Optional[Dict[int, float]] = None,
        compat_mode: bool = False,
        enable_hw: bool = True,
        hw_strength: float = 1.0,
        input_quant_bits: int = 8,
        weight_quant_bits: int = 8,
        use_input_quant: bool = False,
        use_weight_quant: bool = False,
        mapping_mode: str = "lut_linear",
        scale_mode: str = "wmax_gmax",
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        self.compat_mode = bool(compat_mode)
        self.enable_hw = bool(enable_hw)
        self.hw_strength = float(hw_strength)
        self.input_quant_bits = int(input_quant_bits)
        self.weight_quant_bits = int(weight_quant_bits)
        self.use_input_quant = bool(use_input_quant)
        self.use_weight_quant = bool(use_weight_quant)
        self.mapping_mode = str(mapping_mode)
        self.scale_mode = str(scale_mode)

        self.G_nv = nv_map if nv_map else load_nv_mapping_from_csv("nv_table.csv")
        pulses = sorted(int(k) for k in self.G_nv.keys())
        self.pulse_min = pulses[0]
        self.pulse_max = pulses[-1]

        lut = torch.zeros(self.pulse_max + 1, dtype=torch.float32)
        for k, v in self.G_nv.items():
            kk = int(k)
            if 0 <= kk <= self.pulse_max:
                lut[kk] = float(v)
        self.register_buffer("G_lut", lut)

        gmax = lut[self.pulse_max].item() if self.pulse_max >= 0 else 1.0
        self.register_buffer("G_max", torch.tensor(gmax if gmax > 0 else 1.0, dtype=torch.float32))
        self.register_buffer("W_max", torch.tensor(0.0, dtype=torch.float32))
        self._wmax_inited = False

    def reset(self):
        return None

    def _init_wmax(self):
        wmax = self.weight.detach().abs().max().float().clamp_min(1e-6)
        self.W_max.copy_(wmax)
        self._wmax_inited = True

    def _quantize_weight_for_mapping(self) -> Tensor:
        weight = self.weight.float()
        if self.compat_mode or not self.use_weight_quant:
            return weight
        return _quantize_tensor(weight, self.weight_quant_bits, symmetric=True)

    def _map_abs_weight_to_pulse(self, abs_weight: Tensor) -> Tensor:
        if self.mapping_mode != "lut_linear":
            raise ValueError(f"Unsupported linear_mapping_mode: {self.mapping_mode}")

        pulses = torch.round(abs_weight * (self.pulse_max - self.pulse_min) + self.pulse_min).to(torch.long)
        return torch.clamp(pulses, self.pulse_min, self.pulse_max)

    def _scale_conductance_to_weight(self, signed_conductance: Tensor) -> Tensor:
        if self.scale_mode != "wmax_gmax":
            raise ValueError(f"Unsupported linear_scale_mode: {self.scale_mode}")

        scale = self.W_max / (self.G_max + 1e-12)
        return signed_conductance * scale

    def _hardware_weight(self) -> Tensor:
        if not self._wmax_inited:
            self._init_wmax()

        weight_for_mapping = self._quantize_weight_for_mapping()
        abs_weight = (weight_for_mapping.abs() / self.W_max).clamp(0, 1)
        pulses = self._map_abs_weight_to_pulse(abs_weight)
        conductance = self.G_lut.to(device=pulses.device)[pulses]
        signed_conductance = conductance * torch.sign(weight_for_mapping)
        return self._scale_conductance_to_weight(signed_conductance)

    def _effective_input(self, x: Tensor) -> Tensor:
        if self.compat_mode or not self.use_input_quant:
            return x
        return _quantize_tensor(x, self.input_quant_bits, symmetric=True)

    def _effective_weight(self, dtype: torch.dtype) -> Tensor:
        weight = self.weight.to(dtype=dtype)
        if self.compat_mode or not self.enable_hw or self.hw_strength <= 0:
            return weight

        hw_weight = self._hardware_weight().to(device=weight.device, dtype=dtype)
        return torch.lerp(weight, hw_weight, self.hw_strength)

    def forward(self, x):
        if x is None:
            return None

        weight = self._effective_weight(x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


class HardwareConv2d(nn.Module):
    """1x1 convolution mapped to the same conductance table as linear layers."""

    def __init__(
        self,
        conv: nn.Conv2d,
        nv_map: Optional[Dict[int, float]] = None,
        compat_mode: bool = False,
        enable_hw: bool = True,
        hw_strength: float = 1.0,
        input_quant_bits: int = 8,
        weight_quant_bits: int = 8,
        use_input_quant: bool = False,
        use_weight_quant: bool = False,
        mapping_mode: str = "lut_linear",
        scale_mode: str = "wmax_gmax",
    ):
        super().__init__()

        if conv.kernel_size != (1, 1) or conv.stride != (1, 1) or conv.padding != (0, 0) or conv.groups != 1:
            raise ValueError(
                "HardwareConv2d currently only supports 1x1, stride=1, padding=0, groups=1 convolutions."
            )

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = nn.Parameter(conv.weight.detach().clone(), requires_grad=False)
        if conv.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(conv.bias.detach().clone(), requires_grad=False)

        self.compat_mode = bool(compat_mode)
        self.enable_hw = bool(enable_hw)
        self.hw_strength = float(hw_strength)
        self.input_quant_bits = int(input_quant_bits)
        self.weight_quant_bits = int(weight_quant_bits)
        self.use_input_quant = bool(use_input_quant)
        self.use_weight_quant = bool(use_weight_quant)
        self.mapping_mode = str(mapping_mode)
        self.scale_mode = str(scale_mode)

        self.G_nv = nv_map if nv_map else load_nv_mapping_from_csv("nv_table.csv")
        pulses = sorted(int(k) for k in self.G_nv.keys())
        self.pulse_min = pulses[0]
        self.pulse_max = pulses[-1]

        lut = torch.zeros(self.pulse_max + 1, dtype=torch.float32)
        for k, v in self.G_nv.items():
            kk = int(k)
            if 0 <= kk <= self.pulse_max:
                lut[kk] = float(v)
        self.register_buffer("G_lut", lut)

        gmax = lut[self.pulse_max].item() if self.pulse_max >= 0 else 1.0
        self.register_buffer("G_max", torch.tensor(gmax if gmax > 0 else 1.0, dtype=torch.float32))
        self.register_buffer("W_max", torch.zeros(self.out_channels, 1, 1, 1, dtype=torch.float32))
        self._wmax_inited = False

    def reset(self):
        return None

    def _init_wmax(self):
        wmax = self.weight.detach().abs().amax(dim=(1, 2, 3), keepdim=True).float().clamp_min(1e-6)
        self.W_max.copy_(wmax)
        self._wmax_inited = True

    def _quantize_weight_for_mapping(self) -> Tensor:
        weight = self.weight.float()
        if self.compat_mode or not self.use_weight_quant:
            return weight
        return _quantize_tensor(weight, self.weight_quant_bits, symmetric=True)

    def _map_abs_weight_to_pulse(self, abs_weight: Tensor) -> Tensor:
        if self.mapping_mode != "lut_linear":
            raise ValueError(f"Unsupported conv_mapping_mode: {self.mapping_mode}")

        pulses = torch.round(abs_weight * (self.pulse_max - self.pulse_min) + self.pulse_min).to(torch.long)
        return torch.clamp(pulses, self.pulse_min, self.pulse_max)

    def _scale_conductance_to_weight(self, signed_conductance: Tensor) -> Tensor:
        if self.scale_mode != "wmax_gmax":
            raise ValueError(f"Unsupported conv_scale_mode: {self.scale_mode}")

        scale = self.W_max / (self.G_max + 1e-12)
        return signed_conductance * scale

    def _hardware_weight(self) -> Tensor:
        if not self._wmax_inited:
            self._init_wmax()

        weight_for_mapping = self._quantize_weight_for_mapping()
        abs_weight = (weight_for_mapping.abs() / self.W_max).clamp(0, 1)
        pulses = self._map_abs_weight_to_pulse(abs_weight)
        conductance = self.G_lut.to(device=pulses.device)[pulses]
        signed_conductance = conductance * torch.sign(weight_for_mapping)
        return self._scale_conductance_to_weight(signed_conductance)

    def _effective_input(self, x: Tensor) -> Tensor:
        if self.compat_mode or not self.use_input_quant:
            return x
        return _quantize_tensor(x, self.input_quant_bits, symmetric=True)

    def _effective_weight(self, dtype: torch.dtype) -> Tensor:
        weight = self.weight.to(dtype=dtype)
        if self.compat_mode or not self.enable_hw or self.hw_strength <= 0:
            return weight

        hw_weight = self._hardware_weight().to(device=weight.device, dtype=dtype)
        return torch.lerp(weight, hw_weight, self.hw_strength)

    def forward(self, x):
        if x is None:
            return None

        x = self._effective_input(x)
        weight = self._effective_weight(x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class HardwareSelectAdaptivePool2d(nn.Module):
    """Pool wrapper that keeps the classifier head interface unchanged."""

    def __init__(
        self,
        pool_module: nn.Module,
        compat_mode: bool = False,
        enable_hw: bool = True,
        hw_strength: float = 1.0,
        quant_bits: int = 8,
    ):
        super().__init__()
        self.pool_type = getattr(pool_module, "pool_type", "")
        self.pool = getattr(pool_module, "pool", nn.Identity())
        self.flatten = getattr(pool_module, "flatten", None)
        if self.flatten is None:
            self.flatten = getattr(pool_module, "pool_if", nn.Identity())

        self.compat_mode = bool(compat_mode)
        self.enable_hw = bool(enable_hw)
        self.hw_strength = float(hw_strength)
        self.quant_bits = int(quant_bits)

    def reset(self):
        return None

    def is_identity(self):
        return not self.pool_type

    def feat_mult(self):
        if hasattr(self.pool, "feat_mult"):
            return self.pool.feat_mult()
        return 1

    def _apply_hardware_effect(self, y: Tensor) -> Tensor:
        if y is None or self.compat_mode or not self.enable_hw or self.hw_strength <= 0:
            return y
        quantized = _quantize_tensor(y, self.quant_bits, symmetric=False)
        return torch.lerp(y, quantized, self.hw_strength)

    def forward(self, x):
        if x is None:
            return None

        x = self.pool(x)
        x = self._apply_hardware_effect(x)
        if self.flatten is not None:
            x = self.flatten(x)
        return x

    def __repr__(self):
        flatten_name = self.flatten.__class__.__name__ if self.flatten is not None else "None"
        return f"{self.__class__.__name__}(pool_type={self.pool_type}, flatten={flatten_name})"


def _create_hardware_nonlinear(module: nn.Module, args) -> Optional[nn.Module]:
    """Create the hardware-aware replacement for a nonlinear or norm module."""

    compat_mode = _get_hw_arg(args, "compat_mode", False)
    replace_norm = _get_hw_arg(args, "replace_norm", True)
    replace_layernorm = _get_hw_arg(args, "replace_layernorm", replace_norm)
    replace_batchnorm = _get_hw_arg(args, "replace_batchnorm", replace_norm)

    if isinstance(module, exp_comp_neuron):
        wrapped = module.func
    else:
        wrapped = module

    if isinstance(wrapped, nn.Softmax) and _get_hw_arg(args, "replace_softmax", True):
        return HardwareSoftmax(
            wrapped,
            compat_mode=compat_mode,
            hw_strength=_get_hw_arg(args, "softmax_hw_strength", 1.0),
            quant_bits=_get_hw_arg(args, "softmax_quant_bits", 8),
        )

    if isinstance(wrapped, nn.ReLU) and _get_hw_arg(args, "replace_relu", True):
        return HardwareReLU(
            wrapped,
            compat_mode=compat_mode,
            hw_strength=_get_hw_arg(args, "relu_hw_strength", 1.0),
            quant_bits=_get_hw_arg(args, "relu_quant_bits", 8),
        )

    if isinstance(wrapped, nn.LayerNorm) and replace_layernorm:
        return HardwareNorm(
            wrapped,
            compat_mode=compat_mode,
            hw_strength=_get_hw_arg(args, "norm_hw_strength", 1.0),
            quant_bits=_get_hw_arg(args, "norm_quant_bits", 8),
        )

    if isinstance(wrapped, (nn.BatchNorm1d, nn.BatchNorm2d)) and replace_batchnorm:
        return HardwareBatchNorm(
            wrapped,
            compat_mode=compat_mode,
            hw_strength=_get_hw_arg(args, "norm_hw_strength", 1.0),
            quant_bits=_get_hw_arg(args, "norm_quant_bits", 8),
        )

    return None


def replace_nonlinear_by_hardware(model, args=None):
    """Recursively replace nonlinear modules with hardware-aware wrappers."""

    for name, module in model._modules.items():
        replacement = _create_hardware_nonlinear(module, args)
        if replacement is not None:
            model._modules[name] = replacement
            continue

        if hasattr(module, "_modules"):
            model._modules[name] = replace_nonlinear_by_hardware(module, args)
    return model


def calibrate_hardware_norms(model, data_loader, device, max_batches: Optional[int] = None, num_timesteps: int = 1):
    """Collect fixed LayerNorm statistics from calibration data."""

    norm_modules = [m for m in model.modules() if isinstance(m, HardwareNorm)]
    if not norm_modules or data_loader is None:
        return model

    for module in norm_modules:
        module.mu_avg.zero_()
        module.sigma_avg.fill_(1.0)
        module.calib_count.zero_()

    hooks = []

    def _make_hook(module):
        def _hook(_, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                module.update_calibration(inputs[0])
        return _hook

    for module in norm_modules:
        hooks.append(module.register_forward_pre_hook(_make_hook(module)))

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch_data[0].to(device)
            reset_net(model)
            for _ in range(max(int(num_timesteps), 1)):
                _ = model(images)

    for hook in hooks:
        hook.remove()

    reset_net(model)
    if was_training:
        model.train()
    return model


def _fold_batchnorm_into_prev_layers_aggressive(model):
    """Aggressively fold inference BatchNorm layers into adjacent affine layers."""

    module_name = model.__class__.__name__.lower()

    if module_name == "batch_patchembed":
        if isinstance(getattr(model, "proj", None), nn.Conv2d) and isinstance(getattr(model, "norm", None), (nn.BatchNorm1d, nn.BatchNorm2d)):
            _fuse_post_bn_into_prev_conv(model.proj, model.norm)
            model.norm = nn.Identity()

    if module_name == "swintransformerblock":
        if isinstance(getattr(model, "norm1", None), (nn.BatchNorm1d, nn.BatchNorm2d)) and isinstance(getattr(model.attn, "qkv", None), nn.Linear):
            _fuse_pre_bn_into_next_linear(model.norm1, model.attn.qkv)
            model.norm1 = nn.Identity()

        if isinstance(getattr(model, "norm2", None), (nn.BatchNorm1d, nn.BatchNorm2d)) and isinstance(getattr(model.mlp, "fc1", None), nn.Linear):
            _fuse_pre_bn_into_next_linear(model.norm2, model.mlp.fc1)
            model.norm2 = nn.Identity()

    if module_name == "batch_mlp":
        if isinstance(getattr(model, "norm", None), (nn.BatchNorm1d, nn.BatchNorm2d)) and isinstance(getattr(model, "fc2", None), nn.Linear):
            _fuse_pre_bn_into_next_linear(model.norm, model.fc2)
            model.norm = nn.Identity()

    if module_name == "patchmerging":
        if isinstance(getattr(model, "norm", None), (nn.BatchNorm1d, nn.BatchNorm2d)) and isinstance(getattr(model, "reduction", None), nn.Linear):
            _fuse_pre_bn_into_next_linear(model.norm, model.reduction)
            model.norm = nn.Identity()

    if module_name == "swintransformer":
        global_pool = getattr(getattr(model, "head", None), "global_pool", None)
        pool_if = getattr(getattr(model, "head", None), "pool_if", None)
        head_fc = getattr(getattr(model, "head", None), "fc", None)
        if (
            isinstance(getattr(model, "norm", None), (nn.BatchNorm1d, nn.BatchNorm2d))
            and isinstance(head_fc, nn.Linear)
            and getattr(global_pool, "pool_type", "") == "avg"
            and isinstance(pool_if, nn.Identity)
        ):
            _fuse_pre_bn_into_next_linear(model.norm, head_fc)
            model.norm = nn.Identity()

    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = _fold_batchnorm_into_prev_layers_aggressive(module)
    return model


def fold_batchnorm_into_prev_layers(model, mode: str = "safe"):
    """Entry point for BatchNorm folding; safe mode keeps explicit BN modules."""

    if mode == "aggressive":
        return _fold_batchnorm_into_prev_layers_aggressive(model)
    return model


def replace_pool_by_hardware(model, args=None):
    """Replace classifier pooling with the hardware-aware pool wrapper."""

    if not _get_hw_arg(args, "replace_pool", True):
        return model

    for name, module in model._modules.items():
        module_name = module.__class__.__name__.lower()
        if module_name in {"selectadaptivepool2d", "snn_selectadaptivepool2d"}:
            compat_mode = _get_hw_arg(args, "compat_mode", False)
            model._modules[name] = HardwareSelectAdaptivePool2d(
                module,
                compat_mode=compat_mode,
                enable_hw=True,
                hw_strength=_get_hw_arg(args, "pool_hw_strength", 1.0),
                quant_bits=_get_hw_arg(args, "pool_quant_bits", 8),
            )
            continue

        if hasattr(module, "_modules"):
            model._modules[name] = replace_pool_by_hardware(module, args)
    return model


def replace_conv_by_hardware(model, args=None, module_prefix: str = ""):
    """Replace the patch embedding 1x1 convolution with a hardware wrapper."""

    if not _get_hw_arg(args, "replace_conv", True):
        return model

    for name, module in model._modules.items():
        child_prefix = f"{module_prefix}.{name}" if module_prefix else name
        if isinstance(module, nn.Conv2d) and child_prefix.endswith("patch_embed.proj"):
            compat_mode = _get_hw_arg(args, "compat_mode", False)
            model._modules[name] = HardwareConv2d(
                module,
                compat_mode=compat_mode,
                enable_hw=True,
                hw_strength=_get_hw_arg(args, "conv_hw_strength", 1.0),
                input_quant_bits=_get_hw_arg(args, "conv_input_quant_bits", 8),
                weight_quant_bits=_get_hw_arg(args, "conv_weight_quant_bits", 8),
                use_input_quant=_get_hw_arg(args, "conv_use_input_quant", False),
                use_weight_quant=_get_hw_arg(args, "conv_use_weight_quant", False),
                mapping_mode=_get_hw_arg(args, "conv_mapping_mode", "lut_linear"),
                scale_mode=_get_hw_arg(args, "conv_scale_mode", "wmax_gmax"),
            )
            continue

        if hasattr(module, "_modules"):
            model._modules[name] = replace_conv_by_hardware(module, args, child_prefix)
    return model


def replace_linear_by_hardware(model, args=None, module_prefix: str = ""):
    """Replace inference-time linear layers with conductance-mapped wrappers."""

    for name, module in model._modules.items():
        child_prefix = f"{module_prefix}.{name}" if module_prefix else name
        should_replace_linear = _get_hw_arg(args, "replace_linear", True)
        should_replace_classifier = _get_hw_arg(args, "replace_classifier", True) and child_prefix.startswith("backbone.head.")

        if isinstance(module, nn.Linear) and (should_replace_linear or should_replace_classifier):
            compat_mode = _get_hw_arg(args, "compat_mode", False)
            model._modules[name] = HardwareLinear(
                module,
                compat_mode=compat_mode,
                enable_hw=True,
                hw_strength=_get_hw_arg(args, "linear_hw_strength", 1.0),
                input_quant_bits=_get_hw_arg(args, "linear_input_quant_bits", 8),
                weight_quant_bits=_get_hw_arg(args, "linear_weight_quant_bits", 8),
                use_input_quant=_get_hw_arg(args, "linear_use_input_quant", False),
                use_weight_quant=_get_hw_arg(args, "linear_use_weight_quant", False),
                mapping_mode=_get_hw_arg(args, "linear_mapping_mode", "lut_linear"),
                scale_mode=_get_hw_arg(args, "linear_scale_mode", "wmax_gmax"),
            )
            continue

        if hasattr(module, "_modules"):
            model._modules[name] = replace_linear_by_hardware(module, args, child_prefix)
    return model


def replace_at_by_neuron(model, args=None):
    """Replace the matrix-product placeholder with the AtNeuron hardware module."""

    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_at_by_neuron(module, args)
        if module.__class__.__name__.lower() == "myat":
            model._modules[name] = AtNeuron(enable_hw=_get_hw_arg(args, "enable_neuron_hardware", True))
    return model


def get_modules(nowname, model):
    flag = 0
    for _, module in model._modules.items():
        if flag == 0:
            print(nowname, end=" ")
            flag = 1
        print(module.__class__.__name__.lower(), end=" ")
    if flag == 1:
        print("")
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = get_modules(name, module)
    return model


def reset_net(model):
    """Reset every module that exposes a zero-argument reset() method."""

    for module in model._modules.values():
        if hasattr(module, "_modules"):
            reset_net(module)

        reset_fn = getattr(module, "reset", None)
        if not callable(reset_fn):
            continue

        try:
            if len(inspect.signature(reset_fn).parameters) == 0:
                reset_fn()
        except (TypeError, ValueError):
            continue
    return model


class BaseMonitor:
    """Minimal hook-based monitor base class."""

    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        if isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for _, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class SOPMonitor(BaseMonitor):
    """Collect sparse-operation statistics from the converted SNN."""

    def __init__(self, net: nn.Module, type=1):
        super().__init__()
        for name, m in net.named_modules():
            module_name = m.__class__.__name__.lower()

            if type == 1 and module_name in {"linear", "hardwarelinear"}:
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook1(name)))

            if type == 1 and module_name in {"conv2d", "hardwareconv2d"}:
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook_conv(name)))

            if type == 1 and module_name == "atneuron":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook1_2(name)))

            if type == 1 and module_name == "hardwareselectadaptivepool2d":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook_pool(name)))

            if type == 2 and module_name == "twosideneuron":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook2(name)))

    def cal_sop1(self, x: Tensor, m):
        y = torch.zeros_like(x)
        y[x != 0] = 1
        with torch.no_grad():
            weight_device = m.weight.device
            weight_dtype = m.weight.dtype

            if m.bias is not None:
                bias_for_ones = torch.zeros_like(m.bias)
                bias_for_sum = torch.zeros_like(m.bias)
            else:
                bias_for_ones = None
                bias_for_sum = None

            out0 = F.linear(y, torch.ones_like(m.weight), bias_for_ones).sum()
            ones_input_for_sum = torch.ones(
                y.shape[:-1] + (m.weight.shape[1],),
                device=weight_device,
                dtype=weight_dtype,
            )

            sum0 = F.linear(
                ones_input_for_sum,
                torch.ones_like(m.weight),
                bias_for_sum,
            ).sum()
            return out0, sum0

    def create_hook1(self, name):
        def hook1(m, x: Tensor, y):
            if self.is_enable():
                self.name_records_index[name].append(len(self.records))
                self.records.append(self.cal_sop1(x[0], m))

        return hook1

    def cal_sop_conv(self, x: Tensor, m):
        y = torch.zeros_like(x)
        y[x != 0] = 1
        with torch.no_grad():
            ones_weight = torch.ones_like(m.weight)
            out0 = F.conv2d(
                y,
                ones_weight,
                bias=None,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                groups=m.groups,
            ).sum()
            sum0 = F.conv2d(
                torch.ones_like(y),
                ones_weight,
                bias=None,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                groups=m.groups,
            ).sum()
            return out0, sum0

    def create_hook_conv(self, name):
        def hook_conv(m, x: Tensor, y):
            if self.is_enable():
                self.name_records_index[name].append(len(self.records))
                self.records.append(self.cal_sop_conv(x[0], m))

        return hook_conv

    def cal_sop1_2(self, a: Tensor, b: Tensor, m: AtNeuron):
        tmp_a = torch.ones_like(a)
        tmp_b = torch.ones_like(b)
        sum0 = (tmp_a @ tmp_b).sum()
        tmp_a = torch.zeros_like(a)
        tmp_b = torch.zeros_like(b)
        tmp_a[a != 0] = 1
        tmp_b[b != 0] = 1
        tmp_as = torch.zeros_like(m.tot_a)
        tmp_bs = torch.zeros_like(m.tot_b)
        tmp_as[m.tot_a != 0] = 1
        tmp_bs[m.tot_b != 0] = 1
        out01 = (tmp_a @ tmp_b).sum()
        out02 = (tmp_a @ tmp_bs).sum()
        out03 = (tmp_as @ tmp_b).sum()
        out0 = out01 + out02 + out03
        return out0, sum0

    def create_hook1_2(self, name):
        def hook1_2(m: AtNeuron, x, y):
            if self.is_enable():
                self.name_records_index[name].append(len(self.records))
                self.records.append(self.cal_sop1_2(x[0], x[1], m))

        return hook1_2

    def cal_sop_pool(self, x: Tensor):
        active = (x != 0).sum()
        total = torch.tensor(x.numel(), device=x.device, dtype=active.dtype)
        return active, total

    def create_hook_pool(self, name):
        def hook_pool(m: HardwareSelectAdaptivePool2d, x: Tensor, y):
            if self.is_enable():
                self.name_records_index[name].append(len(self.records))
                self.records.append(self.cal_sop_pool(x[0]))

        return hook_pool

    def cal_sop2(self, x):
        num_elements = x[0].numel()
        tmp = []
        for i in x:
            tmp.append((i != 0).sum())
        return sum(tmp), num_elements, tmp

    def create_hook2(self, name):
        def hook2(m: TwoSideNeuron, x: Tensor, y):
            if self.is_enable():
                self.name_records_index[name].append(len(self.records))
                self.records.append(self.cal_sop2(y))

        return hook2
