import torch
import numpy as np
import matplotlib.pyplot as plt
from sim_spike import SimAtNeuron, SimTwoSideNeuron, load_nv_mapping_from_csv, load_v_mapping_from_csv
from ECMT_uitls import trans_utils

RefAtNeuron = trans_utils.AtNeuron
RefTwoSideNeuron = trans_utils.TwoSideNeurons

# ==== 1. Load remapped tables ====
nv_map = load_nv_mapping_from_csv("nv_table.csv")
v_map = load_v_mapping_from_csv("v_table.csv")

# ==== 2. Initialize neurons ====
sim_at = SimAtNeuron(nv_map)
ref_at = RefAtNeuron()

sim_ts = SimTwoSideNeuron(v_map, 1.0, 1.0, 4, 2)
ref_ts = RefTwoSideNeuron(scale_p=1.0, scale_n=1.0, times=4, gap=2)

# ==== 3. Generate diverse input (16 steps, 8-dim input) ====
np.random.seed(42)
input_dim = 8
num_steps = 16
X = [np.random.rand(input_dim) for _ in range(num_steps)]
Y_pulse = np.random.randint(0, 32, size=(input_dim,))
Y = torch.tensor([[nv_map.get(p, 0.0) for p in Y_pulse]], dtype=torch.float32).T  # [8,1]

state_codes = list(v_map.keys())
states = np.random.choice(state_codes, size=num_steps).tolist()
v_inputs = torch.tensor([v_map[s] for s in states], dtype=torch.float32).reshape(-1,1)

# ==== 4. Simulation loop ====
sim_at_out, ref_at_out = [], []
sim_ts_spk, ref_ts_spk = [], []
sim_ts_mem, ref_ts_mem = [], []

ref_at.reset()
sim_at.reset()
ref_ts.reset()
sim_ts.reset()

for t in range(num_steps):
    x_np = X[t]
    x_torch = torch.tensor(x_np, dtype=torch.float32).reshape(1, -1)

    sim_out = sim_at.forward(x_np, Y_pulse)
    ref_out = ref_at(x_torch, Y).squeeze().item()
    sim_at_out.append(sim_out)
    ref_at_out.append(ref_out)

    vin = v_inputs[t]
    s_spk, s_mem = sim_ts.forward(states[t])
    r_spk = ref_ts(vin)
    sim_ts_spk.append(sum([1 for s in s_spk if s != 0]))
    sim_ts_mem.append(s_mem)
    ref_ts_spk.append(sum([(i != 0).sum().item() for i in r_spk]))
    ref_ts_mem.append(ref_ts.neuron.item())

# ==== 5. Plot results ====
t = np.arange(1, num_steps + 1)

plt.figure()
plt.plot(t, sim_at_out, label='Simulated AtNeuron')
plt.plot(t, ref_at_out, label='Original AtNeuron')
plt.title("AtNeuron Output Comparison")
plt.xlabel("Timestep")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, sim_ts_spk, label='Simulated')
plt.plot(t, ref_ts_spk, label='Original')
plt.title("TwoSideNeuron Spike Count Comparison")
plt.xlabel("Timestep")
plt.ylabel("Spikes")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, sim_ts_mem, label='Simulated')
plt.plot(t, ref_ts_mem, label='Original')
plt.title("TwoSideNeuron Membrane Potential")
plt.xlabel("Timestep")
plt.ylabel("Potential")
plt.legend()
plt.grid(True)
plt.show()
