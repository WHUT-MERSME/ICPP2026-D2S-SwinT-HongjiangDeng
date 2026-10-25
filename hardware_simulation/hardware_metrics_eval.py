from __future__ import annotations

import json
import math
import os
import re
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hardware_simulation import trans_utils_hardware


class BaseMonitor:
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


class HardwarePerfMonitor(BaseMonitor):
    """
    独立于硬件模拟主文件的推理性能估算监控器。

    它只负责收集“架构级估算”所需的原始计数：
    - crossbar 类模块的 MAC / tile / 周期相关计数
    - TwoSideNeuron 的比较与脉冲计数
    - Softmax / Norm / ReLU / Pool 的 element-wise 调用规模
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        for name, m in net.named_modules():
            module_name = m.__class__.__name__.lower()

            if module_name in {"linear", "hardwarelinear"}:
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_linear_hook(name)))

            if module_name in {"conv2d", "hardwareconv2d"}:
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_conv_hook(name)))

            if module_name == "atneuron":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_at_hook(name)))

            if module_name == "twosideneuron":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_twoside_hook(name)))

            if module_name == "hardwaresoftmax":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_elementwise_hook(name, "softmax")))

            if module_name in {"hardwarenorm", "hardwarebatchnorm"}:
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                subkind = "layernorm" if module_name == "hardwarenorm" else "batchnorm"
                self.hooks.append(m.register_forward_hook(self.create_elementwise_hook(name, subkind)))

            if module_name == "hardwarerelu":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_elementwise_hook(name, "relu")))

            if module_name == "hardwareselectadaptivepool2d":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_pool_perf_hook(name)))

    def _append_record(self, name: str, payload: Dict):
        self.name_records_index[name].append(len(self.records))
        self.records.append(payload)

    def create_linear_hook(self, name):
        def hook(m, x: Tensor, y):
            if not self.is_enable():
                return
            inp = x[0]
            vectors = int(math.prod(inp.shape[:-1])) if inp.ndim > 1 else 1
            in_features = int(m.weight.shape[1])
            out_features = int(m.weight.shape[0])
            actual_mac = float((inp != 0).sum().item() * out_features)
            potential_mac = float(inp.numel() * out_features)
            self._append_record(name, {
                "kind": "crossbar",
                "subkind": "linear",
                "name": name,
                "actual_mac": actual_mac,
                "potential_mac": potential_mac,
                "vectors": vectors,
                "in_features": in_features,
                "out_features": out_features,
            })
        return hook

    def create_conv_hook(self, name):
        def hook(m, x: Tensor, y):
            if not self.is_enable():
                return
            inp = x[0]
            out = y
            actual_mac, potential_mac = _cal_sop_conv(inp, m)
            out_h = int(out.shape[-2])
            out_w = int(out.shape[-1])
            vectors = int(out.shape[0] * out_h * out_w)
            in_features = int((m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])
            out_features = int(m.out_channels)
            self._append_record(name, {
                "kind": "crossbar",
                "subkind": "conv",
                "name": name,
                "actual_mac": float(actual_mac),
                "potential_mac": float(potential_mac),
                "vectors": vectors,
                "in_features": in_features,
                "out_features": out_features,
            })
        return hook

    def create_at_hook(self, name):
        def hook(m: trans_utils_hardware.AtNeuron, x, y):
            if not self.is_enable():
                return
            a, b = x[0], x[1]
            actual_mac, potential_mac = _cal_sop_at(a, b, m)
            vectors = int(math.prod(a.shape[:-1])) if a.ndim > 1 else 1
            in_features = int(a.shape[-1])
            out_features = int(b.shape[-1])
            self._append_record(name, {
                "kind": "crossbar",
                "subkind": "atneuron",
                "name": name,
                "actual_mac": float(actual_mac),
                "potential_mac": float(potential_mac),
                "vectors": vectors,
                "in_features": in_features,
                "out_features": out_features,
            })
        return hook

    def create_twoside_hook(self, name):
        def hook(m: trans_utils_hardware.TwoSideNeuron, x: Tensor, y):
            if not self.is_enable():
                return
            spike_count, num_elements = _cal_spike_and_compare(y)
            compare_ops = float(num_elements * m.times * 2)
            self._append_record(name, {
                "kind": "neuron",
                "subkind": "twoside",
                "name": name,
                "spike_count": float(spike_count),
                "compare_ops": compare_ops,
                "serial_levels": int(m.times),
            })
        return hook

    def create_elementwise_hook(self, name, subkind: str):
        def hook(m, x: Tensor, y):
            if not self.is_enable():
                return
            inp = x[0] if isinstance(x, (tuple, list)) and len(x) > 0 else y
            elements = int(inp.numel()) if isinstance(inp, torch.Tensor) else 0
            self._append_record(name, {
                "kind": "digital",
                "subkind": subkind,
                "name": name,
                "elements": float(elements),
            })
        return hook

    def create_pool_perf_hook(self, name):
        def hook(m, x: Tensor, y):
            if not self.is_enable():
                return
            inp = x[0]
            self._append_record(name, {
                "kind": "digital",
                "subkind": "pool",
                "name": name,
                "elements": float(inp.numel()),
            })
        return hook


def _cal_sop_conv(x: Tensor, m):
    y = torch.zeros_like(x)
    y[x != 0] = 1
    with torch.no_grad():
        ones_weight = torch.ones_like(m.weight)
        out0 = F.conv2d(
            y, ones_weight, bias=None, stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups
        ).sum()
        sum0 = F.conv2d(
            torch.ones_like(y), ones_weight, bias=None, stride=m.stride, padding=m.padding,
            dilation=m.dilation, groups=m.groups
        ).sum()
        return float(out0.item()), float(sum0.item())


def _cal_sop_at(a: Tensor, b: Tensor, m):
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
    out0 = (tmp_a @ tmp_b).sum() + (tmp_a @ tmp_bs).sum() + (tmp_as @ tmp_b).sum()
    return float(out0.item()), float(sum0.item())


def _cal_spike_and_compare(x):
    num_elements = x[0].numel()
    spike_count = 0
    for i in x:
        spike_count += (i != 0).sum().item()
    return float(spike_count), int(num_elements)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _paper_layer_label(name: str, kind: str, subkind: str) -> str:
    if name == "proj":
        return "Input Projection"
    if name == "norm":
        return "Final LayerNorm"
    if name == "backbone.norm":
        return "Backbone Final Normalization"
    if name == "backbone.head.fc":
        return "Classifier Fully Connected"
    if name == "backbone.head.global_pool":
        return "Global Average Pool"
    if name == "backbone.patch_embed.proj":
        return "Patch Embedding Projection"
    if name == "backbone.patch_embed.norm":
        return "Patch Embedding Normalization"

    downsample_match = re.match(r"backbone\.layers\.(\d+)\.downsample\.(reduction|norm)$", name)
    if downsample_match:
        stage = int(downsample_match.group(1)) + 1
        op = downsample_match.group(2)
        if op == "reduction":
            return f"Stage {stage} Patch Merging Reduction"
        return f"Stage {stage} Patch Merging Normalization"

    block_match = re.match(r"backbone\.layers\.(\d+)\.blocks\.(\d+)\.(.+)$", name)
    if not block_match:
        return name.replace(".", " / ")

    stage = int(block_match.group(1)) + 1
    block = int(block_match.group(2)) + 1
    tail = block_match.group(3)
    prefix = f"Stage {stage} Block {block}"

    mapping = {
        "norm1": "Pre-Attention Normalization",
        "norm2": "Pre-MLP Normalization",
        "attn.softmax": "Attention Softmax",
        "attn.qkv": "QKV Projection",
        "attn.proj": "Attention Output Projection",
        "attn.at1": "Attention Matrix Product 1",
        "attn.at2": "Attention Matrix Product 2",
        "attn.x_in_test": "Attention Input Neuron",
        "attn.q_out_test": "Query Output Neuron",
        "attn.k_out_test": "Key Output Neuron",
        "attn.v_out_test": "Value Output Neuron",
        "attn.attn_out_test": "Attention Map Output Neuron",
        "attn.x_out_test": "Attention Output Neuron",
        "mlp.fc1": "MLP Fully Connected 1",
        "mlp.fc2": "MLP Fully Connected 2",
        "mlp.act": "MLP Activation",
        "mlp.norm": "MLP Normalization",
        "mlp.x1_in_test": "MLP FC1 Input Neuron",
        "mlp.x2_in_test": "MLP FC2 Input Neuron",
    }
    if tail in mapping:
        return f"{prefix} {mapping[tail]}"

    if kind == "digital" and subkind == "batchnorm":
        return f"{prefix} Batch Normalization"
    if kind == "digital" and subkind == "layernorm":
        return f"{prefix} Layer Normalization"

    return f"{prefix} {tail.replace('.', ' ')}"


def load_hardware_perf_config(config_dir: str):
    crossbar_cfg = _load_json(os.path.join(config_dir, "perf_crossbar_config.json"))
    neuron_cfg = _load_json(os.path.join(config_dir, "perf_neuron_config.json"))
    digital_cfg = _load_json(os.path.join(config_dir, "perf_digital_config.json"))

    merged = {"enable": True}
    merged.update(crossbar_cfg)
    merged.update(neuron_cfg)
    merged.update(digital_cfg)
    return SimpleNamespace(**merged)


def estimate_hardware_metrics(records, hw_cfg, total_samples: int):
    if hw_cfg is None:
        return None

    def _param(name, default=0.0):
        value = getattr(hw_cfg, name, default)
        return default if value is None else value

    rows_cfg = getattr(hw_cfg, "crossbar_rows", None)
    cols_cfg = getattr(hw_cfg, "crossbar_cols", None)
    if rows_cfg is None:
        rows_cfg = getattr(hw_cfg, "measurement_array_rows", 5)
    if cols_cfg is None:
        cols_cfg = getattr(hw_cfg, "measurement_array_cols", 5)

    rows = max(int(rows_cfg), 1)
    cols = max(int(cols_cfg), 1)

    total_energy_pj = 0.0
    total_latency_ns = 0.0
    total_actual_mac = 0.0
    total_potential_mac = 0.0
    total_spikes = 0.0
    total_crossbar_cycles = 0.0
    total_crossbar_latency_rounds = 0.0
    crossbar_parallel_banks = max(int(_param("crossbar_parallel_banks", 1)), 1)
    vector_parallel_lanes = max(int(_param("vector_parallel_lanes", 1)), 1)
    per_kind = {
        "crossbar": {"energy_pj": 0.0, "latency_ns": 0.0},
        "neuron": {"energy_pj": 0.0, "latency_ns": 0.0},
        "digital": {"energy_pj": 0.0, "latency_ns": 0.0},
    }
    per_layer = {}

    for rec in records:
        name = rec["name"]
        kind = rec["kind"]
        energy_pj = 0.0
        latency_ns = 0.0

        if kind == "crossbar":
            tile_rows = math.ceil(rec["in_features"] / rows)
            tile_cols = math.ceil(rec["out_features"] / cols)
            tile_count = tile_rows * tile_cols
            cycles = rec["vectors"] * tile_count
            effective_parallel_banks = min(crossbar_parallel_banks, max(tile_count, 1))
            vector_rounds = math.ceil(rec["vectors"] / vector_parallel_lanes)
            latency_rounds = vector_rounds * math.ceil(tile_count / effective_parallel_banks)
            energy_pj = (
                rec["actual_mac"] * _param("crossbar_e_mac_pj", 0.0)
                + cycles * (_param("crossbar_e_adc_pj", 0.0) + _param("crossbar_e_dac_pj", 0.0))
            )
            latency_ns = latency_rounds * (
                _param("crossbar_t_mac_ns", 0.0)
                + _param("crossbar_t_adc_ns", 0.0)
                + _param("crossbar_t_dac_ns", 0.0)
            )
            total_actual_mac += rec["actual_mac"]
            total_potential_mac += rec["potential_mac"]
            total_crossbar_cycles += cycles
            total_crossbar_latency_rounds += latency_rounds

        elif kind == "neuron":
            energy_pj = (
                rec["compare_ops"] * _param("neuron_e_compare_pj", 0.0)
                + rec["spike_count"] * _param("neuron_e_spike_pj", 0.0)
            )
            latency_ns = rec["serial_levels"] * _param("neuron_t_level_ns", 0.0)
            total_spikes += rec["spike_count"]

        elif kind == "digital":
            parallelism_map = {
                "softmax": max(int(_param("softmax_parallelism", 1)), 1),
                "layernorm": max(int(_param("layernorm_parallelism", 1)), 1),
                "batchnorm": max(int(_param("batchnorm_parallelism", 1)), 1),
                "relu": max(int(_param("relu_parallelism", 1)), 1),
                "pool": max(int(_param("pool_parallelism", 1)), 1),
            }
            energy_map = {
                "softmax": _param("softmax_e_elem_pj", 0.0),
                "layernorm": _param("layernorm_e_elem_pj", 0.0),
                "batchnorm": _param("batchnorm_e_elem_pj", 0.0),
                "relu": _param("relu_e_elem_pj", 0.0),
                "pool": _param("pool_e_elem_pj", 0.0),
            }
            latency_map = {
                "softmax": _param("softmax_t_elem_ns", 0.0),
                "layernorm": _param("layernorm_t_elem_ns", 0.0),
                "batchnorm": _param("batchnorm_t_elem_ns", 0.0),
                "relu": _param("relu_t_elem_ns", 0.0),
                "pool": _param("pool_t_elem_ns", 0.0),
            }
            subkind = rec["subkind"]
            energy_pj = rec["elements"] * energy_map.get(subkind, 0.0)
            latency_ns = math.ceil(rec["elements"] / parallelism_map.get(subkind, 1)) * latency_map.get(subkind, 0.0)

        total_energy_pj += energy_pj
        total_latency_ns += latency_ns
        per_kind[kind]["energy_pj"] += energy_pj
        per_kind[kind]["latency_ns"] += latency_ns

        if name not in per_layer:
            per_layer[name] = {
                "paper_layer": _paper_layer_label(name, kind, rec.get("subkind", "")),
                "kind": kind,
                "subkind": rec.get("subkind", ""),
                "energy_pj": 0.0,
                "latency_ns": 0.0,
            }
        per_layer[name]["energy_pj"] += energy_pj
        per_layer[name]["latency_ns"] += latency_ns

    estimation_ready = any([
        _param("crossbar_e_mac_pj", 0.0) > 0,
        _param("crossbar_t_mac_ns", 0.0) > 0,
        _param("neuron_e_spike_pj", 0.0) > 0,
        _param("neuron_t_level_ns", 0.0) > 0,
        _param("softmax_e_elem_pj", 0.0) > 0,
        _param("layernorm_e_elem_pj", 0.0) > 0,
        _param("batchnorm_e_elem_pj", 0.0) > 0,
        _param("relu_e_elem_pj", 0.0) > 0,
        _param("pool_e_elem_pj", 0.0) > 0,
    ])

    latency_per_sample_ns = total_latency_ns / max(total_samples, 1)
    energy_per_sample_pj = total_energy_pj / max(total_samples, 1)
    throughput_sps = 1e9 / latency_per_sample_ns if latency_per_sample_ns > 0 else 0.0
    mac_util = total_actual_mac / total_potential_mac if total_potential_mac > 0 else 0.0
    sorted_layers = sorted(per_layer.items(), key=lambda kv: kv[1]["energy_pj"], reverse=True)
    nonzero_sorted_layers = [item for item in sorted_layers if item[1]["energy_pj"] > 0 or item[1]["latency_ns"] > 0]

    return {
        "estimation_ready": estimation_ready,
        "crossbar_shape": (rows, cols),
        "crossbar_parallel_banks": crossbar_parallel_banks,
        "vector_parallel_lanes": vector_parallel_lanes,
        "total_samples": int(total_samples),
        "total_energy_pj": total_energy_pj,
        "total_latency_ns": total_latency_ns,
        "total_latency_us": total_latency_ns / 1e3,
        "energy_per_sample_uj": energy_per_sample_pj / 1e6,
        "latency_per_sample_us": latency_per_sample_ns / 1e3,
        "latency_per_sample_ms": latency_per_sample_ns / 1e6,
        "throughput_sps": throughput_sps,
        "actual_mac": total_actual_mac,
        "potential_mac": total_potential_mac,
        "mac_utilization": mac_util,
        "total_spikes": total_spikes,
        "total_crossbar_cycles": total_crossbar_cycles,
        "total_crossbar_latency_rounds": total_crossbar_latency_rounds,
        "per_kind": per_kind,
        "per_layer_sorted": sorted_layers,
        "per_layer_nonzero_sorted": nonzero_sorted_layers,
    }


def save_hardware_perf_summary(summary: Dict, output_path: str):
    serializable = dict(summary)
    serializable["crossbar_shape"] = list(summary.get("crossbar_shape", ()))
    serializable["per_layer_sorted"] = [
        {"layer": layer_name, **layer_vals}
        for layer_name, layer_vals in summary.get("per_layer_sorted", [])
    ]
    serializable["per_layer_nonzero_sorted"] = [
        {"layer": layer_name, **layer_vals}
        for layer_name, layer_vals in summary.get("per_layer_nonzero_sorted", [])
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
