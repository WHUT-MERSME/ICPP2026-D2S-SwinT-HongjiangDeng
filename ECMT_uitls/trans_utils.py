import json
import json
import os
import re
import sys
import time
from torch import Tensor
import torch
import torch.nn as nn

SOFTWARE_EMAC_PJ = 4.6
SOFTWARE_EAC_PJ = 0.9


class EnergyAwareRecord(tuple):
    def __new__(cls, actual_ops, potential_ops, actual_energy_pj, dense_energy_pj):
        obj = super().__new__(cls, (actual_ops, potential_ops))
        obj.actual_ops = float(actual_ops)
        obj.potential_ops = float(potential_ops)
        obj.actual_energy_pj = float(actual_energy_pj)
        obj.dense_energy_pj = float(dense_energy_pj)
        return obj


def _sync_cuda_if_needed():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _now_sync():
    _sync_cuda_if_needed()
    return time.perf_counter()


def _software_paper_layer_label(name: str, kind: str, subkind: str) -> str:
    if name == "proj":
        return "Input Projection"
    if name == "norm":
        return "Final LayerNorm"
    if name == "backbone.norm":
        return "Backbone Final Normalization"
    if name == "backbone.head.fc":
        return "Classifier Fully Connected"
    if name == "backbone.patch_embed.proj":
        return "Patch Embedding Projection"

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
        "attn.qkv": "QKV Projection",
        "attn.proj": "Attention Output Projection",
        "attn.at1": "Attention Matrix Product 1",
        "attn.at2": "Attention Matrix Product 2",
        "mlp.fc1": "MLP Fully Connected 1",
        "mlp.fc2": "MLP Fully Connected 2",
    }
    if tail in mapping:
        return f"{prefix} {mapping[tail]}"

    if kind == "crossbar" and subkind == "atneuron":
        return f"{prefix} Attention Matrix Product"

    return f"{prefix} {tail.replace('.', ' ')}"


def summarize_energy_records(records):
    total_actual_ops = 0.0
    total_potential_ops = 0.0
    total_actual_energy_pj = 0.0
    total_dense_energy_pj = 0.0

    for record in records:
        if isinstance(record, EnergyAwareRecord):
            total_actual_ops += record.actual_ops
            total_potential_ops += record.potential_ops
            total_actual_energy_pj += record.actual_energy_pj
            total_dense_energy_pj += record.dense_energy_pj
        else:
            total_actual_ops += float(record[0])
            total_potential_ops += float(record[1])
            total_actual_energy_pj += float(record[0]) * SOFTWARE_EAC_PJ
            total_dense_energy_pj += float(record[1]) * SOFTWARE_EMAC_PJ

    return {
        "actual_ops": total_actual_ops,
        "potential_ops": total_potential_ops,
        "actual_energy_pj": total_actual_energy_pj,
        "dense_energy_pj": total_dense_energy_pj,
        "actual_energy_uj": total_actual_energy_pj / 1e6,
        "dense_energy_uj": total_dense_energy_pj / 1e6,
    }

def heaviside0(x: torch.Tensor):
    return ((x>=0)).to(x)

class TwoSideNeuron(nn.Module):
    def __init__(self, scale_p=1., scale_n=1., place=None, times = 3, gap=2):
        super(TwoSideNeuron, self).__init__()
        self.scale_p = scale_p
        self.scale_n = scale_n
        self.place = place
        self.times = times
        self.gap = gap
        self.t = 0
        self.neuron = None
    def forward(self, x):
        if self.t == 0:
            self.neuron = torch.zeros_like(x)
        self.neuron += x
        v_threshold1 = self.scale_p
        v_threshold2 = self.scale_n
        for i in range(self.times - 1):
            v_threshold1 *= self.gap
            v_threshold2 *= self.gap
        fire_list = []
        not_fire = torch.ones_like(x)
        for i in range(self.times):
            tmp1 = heaviside0(self.neuron-v_threshold1+self.scale_p/2)
            tmp2 = heaviside0(-self.neuron-v_threshold2+self.scale_n/2)
            tmp1*=v_threshold1
            tmp2*=-v_threshold2
            tmp1 *= not_fire
            tmp2 *= not_fire
            not_fire[tmp1!=0] =0
            not_fire[tmp2!=0] =0
            self.neuron -= tmp1
            self.neuron -= tmp2
            fire_list.append(tmp1)
            fire_list.append(tmp2)
            v_threshold1 /= self.gap
            v_threshold2 /= self.gap
        self.t += 1
        return fire_list
    def reset(self):
        self.t = 0
        self.neuron=None

#Modify the multi-threshold neuron parameters within this function
def get_data_from_place(place,p,n,args):
    sp = p
    sn = n
    gap = 2
    if 'vit' in args.model and place == 'fc2':
        sp = 0.25
        sn = 0.08
    times = args.linear_num
    if place == 'q' or place == 'k' or place == 'v':
        times = args.qkv_num
    if place == 's':
        times = args.softmax_num
        sp = args.softmax_p
        sn = args.softmax_p
    return sp,sn,times,gap
        
def replace_testneuron_by_twosideneuron(model,args):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_testneuron_by_twosideneuron(module,args)
        if 'testneuron' == module.__class__.__name__.lower():
            place = model._modules[name].place
            scale_p,scale_n,times,gap = get_data_from_place(place,float(model._modules[name].scale_p[0]),float(model._modules[name].scale_n[0]),args)
            print(name,scale_p,scale_n,place)
            model._modules[name] = TwoSideNeuron(scale_p=scale_p,scale_n=scale_n,place=place,times=times,gap=gap)
    return model

class MyTestPlace(nn.Module):
    def __init__(self,place=None):
        super(MyTestPlace, self).__init__()
        self.place = place
    def forward(self, x, *args):
        return [x,]

class TestNeuron(nn.Module):
    def __init__(self,place=None,percent=None):
        super(TestNeuron, self).__init__()
        self.place = place
        self.percent = percent
        self.num = 0
        self.scale_p = torch.nn.Parameter(torch.FloatTensor([0.]))
        self.scale_n = torch.nn.Parameter(torch.FloatTensor([0.]))
    #choose threshold
    def forward(self, x, times=2,gap=3,show=0, tmptime=0,scaletimes=1):
        x2 = x.reshape(-1)
        threshold = torch.kthvalue(x2, int(self.percent * x2.numel()), dim=0).values.item()
        self.scale_p = torch.nn.Parameter((self.scale_p*self.num+threshold)/(self.num+1))
        threshold = -torch.kthvalue(x2, int((1-self.percent) * x2.numel()), dim=0).values.item()
        self.scale_n = torch.nn.Parameter((self.scale_n*self.num+threshold)/(self.num+1))
        self.num+=1
        #print(self.scale_p,self.scale_n)
        return [x,]
    def reset(self):
        pass

#replace testplace to testneuron to caculate thresholds
def replace_test_by_testneuron(model,percent=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_test_by_testneuron(module,percent)
        if module.__class__.__name__.lower() == 'mytestplace':
            model._modules[name] = TestNeuron(place=model._modules[name].place,percent=percent)
    return model

class exp_comp_neuron(nn.Module):
    def __init__(self,func,*args,**keywords):
        super(exp_comp_neuron,self).__init__(*args,**keywords)
        self.tot = None
        self.func = func
        self.t=0
    def forward(self, x):
        if self.tot==None:
            last=torch.zeros_like(x)
            self.tot=x.clone()
        else:
            last = self.func(self.tot/self.t)*self.t
            self.tot+=x
        self.t+=1
        now = self.func(self.tot/self.t)*self.t
        return now-last
    def reset(self):
        self.tot = None
        self.t=0

def replace_nonlinear_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_nonlinear_by_neuron(module)
        if 'softmax' in module.__class__.__name__.lower() or 'relu' in module.__class__.__name__.lower() or 'layernorm' in module.__class__.__name__.lower() or 'batchnorm' in module.__class__.__name__.lower():
            model._modules[name] = exp_comp_neuron(func=model._modules[name])
    return model


class MyAt(nn.Module):
    def __init__(self):
        super(MyAt, self).__init__()
    def forward(self, x, y):
        return x @ y

class AtNeuron(nn.Module):
    def __init__(self):
        super(AtNeuron, self).__init__()
        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        self.t=0
    def forward(self, x,y):
        if self.t == 0:
            self.tot_a=x
            self.tot_b=y
            self.tot_t = x@y
            self.t = 1
            return x@y
        else:
            self.tot_t+= x@y+x@self.tot_b+self.tot_a@y
            self.tot_a+=x
            self.tot_b+=y
            self.t += 1
            return (x@self.tot_b+self.tot_a@y-x@y)/(self.t-1)-self.tot_t/(self.t*(self.t-1))

    def reset(self):# Reset the accumulator
        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        self.t=0

def replace_at_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_at_by_neuron(module)
        if module.__class__.__name__.lower()=="myat":
            model._modules[name] = AtNeuron()
    return model

def get_modules(nowname,model):
    flag = 0
    for name, module in model._modules.items():
        if flag==0:
            print(nowname,end=' ')
            flag=1
        print(module.__class__.__name__.lower(),end=' ')
    if flag==1:
        print('')
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = get_modules(name,module)
    return model

def reset_net(model):#initialize all neurons
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            reset_net(module)
        if 'neuron' in module.__class__.__name__.lower():
            module.reset()
    return model

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
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)
    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
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
    def __init__(self, net: nn.Module,type=1):
        super().__init__()
        self.total_actual_ops = 0.0
        self.total_potential_ops = 0.0
        self.total_actual_energy_pj = 0.0
        self.total_dense_energy_pj = 0.0
        self.per_layer_stats = {}
        self.per_kind_stats = {}
        self.sample_anchor_layer = None
        self.anchor_samples_seen = 0.0
        self.wall_start_time = _now_sync()
        self.wall_end_time = None
        for name, m in net.named_modules():
            # calculated energy consumption
            if type ==1 and m.__class__.__name__.lower() == 'linear':
                if self.sample_anchor_layer is None:
                    self.sample_anchor_layer = name
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook1(name)))
            if type ==1 and m.__class__.__name__.lower() == 'atneuron':
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook1_2(name)))
            # calculate fire rate of neurons
            if type ==2 and m.__class__.__name__.lower() == 'twosideneuron':
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook2(name)))

    
    def cal_sop1(self, x: Tensor, m: nn.Linear):
        y = torch.zeros_like(x)
        y[x!=0]=1
        with torch.no_grad():
            weight_device = m.weight.device
            weight_dtype = m.weight.dtype

            # 处理 bias 为 None 的情况
            if m.bias is not None:
                bias_for_ones = torch.zeros_like(m.bias)
                bias_for_sum = torch.zeros_like(m.bias)
            else:
                bias_for_ones = None
                bias_for_sum = None

            out0 = (torch.nn.functional.linear(y, torch.ones_like(m.weight), bias_for_ones)).sum()
            ones_input_for_sum = torch.ones(y.shape[:-1] + (m.weight.shape[1],), device=weight_device,
                                            dtype=weight_dtype)

            sum0 = (torch.nn.functional.linear(ones_input_for_sum,
                                               torch.ones_like(m.weight),
                                               bias_for_sum)).sum()
            actual_ops = float(out0.item())
            potential_ops = float(sum0.item())
            return EnergyAwareRecord(
                actual_ops=actual_ops,
                potential_ops=potential_ops,
                actual_energy_pj=actual_ops * SOFTWARE_EAC_PJ,
                dense_energy_pj=potential_ops * SOFTWARE_EMAC_PJ,
            )

    def create_hook1(self, name):
        def hook1(m: nn.Linear, x: Tensor, y):
            if self.is_enable():
                record = self.cal_sop1(x[0], m)
                self._maybe_track_samples(name, x[0])
                self._accumulate_layer_stats(name, "crossbar", "linear", record)
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(record)
                self.total_actual_ops += record.actual_ops
                self.total_potential_ops += record.potential_ops
                self.total_actual_energy_pj += record.actual_energy_pj
                self.total_dense_energy_pj += record.dense_energy_pj
        return hook1
                             
    def cal_sop1_2(self, A: Tensor,B: Tensor, m: AtNeuron):
        # print(A.shape,B.shape)
        tmp_A = torch.ones_like(A)
        tmp_B = torch.ones_like(B)
        sum0 = (tmp_A@tmp_B).sum()
        tmp_A = torch.zeros_like(A)
        tmp_B = torch.zeros_like(B)
        tmp_A[A!=0]=1
        tmp_B[B!=0]=1
        tmp_As = torch.zeros_like(m.tot_a)
        tmp_As[m.tot_a!=0]=1
        tmp_Bs = torch.zeros_like(m.tot_b)
        tmp_Bs[m.tot_b!=0]=1
        out01 = (tmp_A@tmp_B).sum()
        out02 = (tmp_A@tmp_Bs).sum()
        out03 = (tmp_As@tmp_B).sum()
        out0 = out01+out02+out03
        # print(round(float(out01/sum0),5),round(float(out02/sum0),5),round(float(out03/sum0),5))
        actual_ops = float(out0.item())
        potential_ops = float(sum0.item())
        return EnergyAwareRecord(
            actual_ops=actual_ops,
            potential_ops=potential_ops,
            actual_energy_pj=actual_ops * SOFTWARE_EAC_PJ,
            dense_energy_pj=potential_ops * SOFTWARE_EMAC_PJ,
        )

    def create_hook1_2(self, name):
        def hook1_2(m: AtNeuron, x, y):
            if self.is_enable():
                record = self.cal_sop1_2(x[0],x[1], m)
                self._maybe_track_samples(name, x[0])
                self._accumulate_layer_stats(name, "crossbar", "atneuron", record)
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(record)
                self.total_actual_ops += record.actual_ops
                self.total_potential_ops += record.potential_ops
                self.total_actual_energy_pj += record.actual_energy_pj
                self.total_dense_energy_pj += record.dense_energy_pj
        return hook1_2                   

    def summarize_absolute_energy(self):
        return {
            "actual_ops": self.total_actual_ops,
            "potential_ops": self.total_potential_ops,
            "actual_energy_pj": self.total_actual_energy_pj,
            "dense_energy_pj": self.total_dense_energy_pj,
            "actual_energy_uj": self.total_actual_energy_pj / 1e6,
            "dense_energy_uj": self.total_dense_energy_pj / 1e6,
        }

    def _maybe_track_samples(self, layer_name, x):
        if layer_name != self.sample_anchor_layer:
            return
        if not isinstance(x, torch.Tensor) or x.dim() == 0:
            return
        self.anchor_samples_seen += float(x.shape[0])

    def _accumulate_layer_stats(self, name: str, kind: str, subkind: str, record: EnergyAwareRecord):
        if name not in self.per_layer_stats:
            self.per_layer_stats[name] = {
                "paper_layer": _software_paper_layer_label(name, kind, subkind),
                "kind": kind,
                "subkind": subkind,
                "actual_ops": 0.0,
                "potential_ops": 0.0,
                "actual_energy_pj": 0.0,
                "dense_energy_pj": 0.0,
            }
        layer_stats = self.per_layer_stats[name]
        layer_stats["actual_ops"] += record.actual_ops
        layer_stats["potential_ops"] += record.potential_ops
        layer_stats["actual_energy_pj"] += record.actual_energy_pj
        layer_stats["dense_energy_pj"] += record.dense_energy_pj

        if kind not in self.per_kind_stats:
            self.per_kind_stats[kind] = {
                "actual_ops": 0.0,
                "potential_ops": 0.0,
                "actual_energy_pj": 0.0,
                "dense_energy_pj": 0.0,
            }
        kind_stats = self.per_kind_stats[kind]
        kind_stats["actual_ops"] += record.actual_ops
        kind_stats["potential_ops"] += record.potential_ops
        kind_stats["actual_energy_pj"] += record.actual_energy_pj
        kind_stats["dense_energy_pj"] += record.dense_energy_pj

    def _estimate_total_samples(self, timesteps: int):
        if timesteps <= 0 or self.anchor_samples_seen <= 0:
            return 0.0
        return self.anchor_samples_seen / float(timesteps)

    def _build_perf_summary(self, dataset: str, timesteps: int):
        if self.wall_end_time is None:
            self.wall_end_time = _now_sync()

        total_samples = self._estimate_total_samples(timesteps)
        wall_time_s = max(self.wall_end_time - self.wall_start_time, 0.0)
        summary = self.summarize_absolute_energy()
        summary.update({
            "dataset": dataset,
            "timesteps": int(timesteps),
            "software_emac_pj": SOFTWARE_EMAC_PJ,
            "software_eac_pj": SOFTWARE_EAC_PJ,
            "energy_ratio_vs_dense": self.total_actual_energy_pj / max(self.total_dense_energy_pj, 1e-12),
            "energy_mode": "op_count_baseline",
            "latency_mode": "measured_wall_clock",
            "total_samples": total_samples,
            "wall_time_s": wall_time_s,
        })

        if total_samples > 0:
            summary["actual_energy_per_sample_uj"] = summary["actual_energy_uj"] / total_samples
            summary["dense_energy_per_sample_uj"] = summary["dense_energy_uj"] / total_samples
        else:
            summary["actual_energy_per_sample_uj"] = 0.0
            summary["dense_energy_per_sample_uj"] = 0.0

        if total_samples > 0 and wall_time_s > 0:
            summary["latency_per_sample_ms"] = wall_time_s * 1e3 / total_samples
            summary["throughput_sps"] = total_samples / wall_time_s
        else:
            summary["latency_per_sample_ms"] = 0.0
            summary["throughput_sps"] = 0.0

        per_layer_sorted = sorted(
            self.per_layer_stats.items(),
            key=lambda item: item[1]["actual_energy_pj"],
            reverse=True,
        )
        summary["per_layer_sorted"] = [
            {"layer": layer_name, **layer_stats}
            for layer_name, layer_stats in per_layer_sorted
        ]
        summary["per_layer_nonzero_sorted"] = [
            {"layer": layer_name, **layer_stats}
            for layer_name, layer_stats in per_layer_sorted
            if layer_stats["actual_energy_pj"] > 0
        ]
        summary["per_kind"] = self.per_kind_stats
        return summary

    def export_absolute_energy_summary(self):
        if self.total_potential_ops <= 0:
            return

        main_module = sys.modules.get("__main__")
        result_dir = getattr(main_module, "RESULT_DIR", None)
        timesteps = getattr(main_module, "SNN_TIMESTEPS", None)
        dataset = getattr(main_module, "CHOSEN_DATASET", "unknown")
        if result_dir is None or timesteps is None:
            return

        os.makedirs(result_dir, exist_ok=True)
        summary = self._build_perf_summary(dataset, int(timesteps))

        json_path = os.path.join(result_dir, f"software_energy_summary_T{timesteps}.json")
        txt_path = os.path.join(result_dir, f"software_energy_summary_T{timesteps}.txt")
        perf_json_path = os.path.join(result_dir, f"software_perf_summary_T{timesteps}.json")
        perf_txt_path = os.path.join(result_dir, f"software_perf_summary_T{timesteps}.txt")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(perf_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8-sig") as f:
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Timesteps: {timesteps}\n")
            f.write(f"Total Samples: {summary['total_samples']:.0f}\n")
            f.write(f"Actual Ops: {summary['actual_ops']:.0f}\n")
            f.write(f"Potential Ops: {summary['potential_ops']:.0f}\n")
            f.write(f"Actual Energy (pJ): {summary['actual_energy_pj']:.6f}\n")
            f.write(f"Dense Energy (pJ): {summary['dense_energy_pj']:.6f}\n")
            f.write(f"Actual Energy (uJ): {summary['actual_energy_uj']:.6f}\n")
            f.write(f"Dense Energy (uJ): {summary['dense_energy_uj']:.6f}\n")
            f.write(f"Actual Energy per Sample (uJ): {summary['actual_energy_per_sample_uj']:.6f}\n")
            f.write(f"Dense Energy per Sample (uJ): {summary['dense_energy_per_sample_uj']:.6f}\n")
            f.write(f"EAC baseline (pJ): {SOFTWARE_EAC_PJ:.6f}\n")
            f.write(f"EMAC baseline (pJ): {SOFTWARE_EMAC_PJ:.6f}\n")
            f.write(f"Energy Ratio vs Dense: {summary['energy_ratio_vs_dense']:.6f}\n")
            f.write(f"Wall-clock Runtime (s): {summary['wall_time_s']:.6f}\n")
            f.write(f"Latency per Sample (ms): {summary['latency_per_sample_ms']:.6f}\n")
            f.write(f"Throughput (samples/s): {summary['throughput_sps']:.6f}\n")
            f.write(f"Energy Mode: {summary['energy_mode']}\n")
            f.write(f"Latency Mode: {summary['latency_mode']}\n")

        with open(perf_txt_path, "w", encoding="utf-8-sig") as f:
            f.write(f"Software SNN Model ({dataset}) Performance Summary at T={timesteps}\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"Total Samples: {summary['total_samples']:.0f}\n")
            f.write(f"Actual Energy per Sample (uJ): {summary['actual_energy_per_sample_uj']:.6f}\n")
            f.write(f"Dense Energy per Sample (uJ): {summary['dense_energy_per_sample_uj']:.6f}\n")
            f.write(f"Latency per Sample (ms): {summary['latency_per_sample_ms']:.6f}\n")
            f.write(f"Throughput (samples/s): {summary['throughput_sps']:.6f}\n")
            f.write(f"Wall-clock Runtime (s): {summary['wall_time_s']:.6f}\n")
            f.write(f"Actual Ops: {summary['actual_ops']:.0f}\n")
            f.write(f"Potential Ops: {summary['potential_ops']:.0f}\n")
            f.write(f"Energy Ratio vs Dense: {summary['energy_ratio_vs_dense']:.6f}\n")
            f.write(f"Energy Mode: {summary['energy_mode']}\n")
            f.write(f"Latency Mode: {summary['latency_mode']}\n")

            f.write("\nBy Kind:\n")
            for kind, vals in summary["per_kind"].items():
                f.write(
                    f"{kind}: actual_energy={vals['actual_energy_pj'] / 1e6:.6f} uJ, "
                    f"dense_energy={vals['dense_energy_pj'] / 1e6:.6f} uJ, "
                    f"actual_ops={vals['actual_ops']:.0f}, potential_ops={vals['potential_ops']:.0f}\n"
                )

            nonzero_layers = summary["per_layer_nonzero_sorted"]
            f.write(f"\nTop Energy Layers (All non-zero layers, {len(nonzero_layers)} entries):\n")
            f.write(
                f"{'Paper Layer':<56} | {'Kind':<10} | {'Subkind':<10} | "
                f"{'Actual Energy(uJ)':<18} | {'Dense Energy(uJ)':<18}\n"
            )
            f.write("-" * 128 + "\n")
            for vals in nonzero_layers:
                f.write(
                    f"{vals.get('paper_layer', vals['layer']):<56} | {vals['kind']:<10} | {vals['subkind']:<10} | "
                    f"{vals['actual_energy_pj'] / 1e6:<18.6f} | {vals['dense_energy_pj'] / 1e6:<18.6f}\n"
                )

    def __del__(self):
        try:
            self.export_absolute_energy_summary()
        except Exception:
            pass
        super().__del__()
    
    def cal_sop2(self, x):
        num_elements = x[0].numel()
        tmp = []
        for index,i in enumerate(x):
            tmp.append((i!= 0).sum())
        return sum(tmp),num_elements,tmp

    def create_hook2(self, name):
        def hook2(m: TwoSideNeuron, x: Tensor, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop2(y))
        return hook2
