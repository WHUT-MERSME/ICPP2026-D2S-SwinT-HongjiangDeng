import numpy as np
import pandas as pd

class SimAtNeuron:
    """
    基于《仿真实验方案设计》和trans_utils.py中的ECMT方法精确建模AtNeuron。
    - 支持期望补偿机制（Expectation Compensation）。
    - 使用非易失性忆阻器（NVM）进行权重电导映射：权重由脉冲数转换为忆阻电导。
    - 采用矩阵乘法形式进行推理，包含累积项、补偿项，匹配硬件逐时刻积分。
    """
    def __init__(self, nv_conductance_map):
        self.G_nv = nv_conductance_map  # 非易失性映射表：脉冲数 -> 电导(μS)，器件模拟G
        self.T = 0  # 时间步
        self.Q_acc = None  # 输入电荷累计（输入向量累积）
        self.G_acc = None  # 电导累计（权重向量电导）
        self.I_acc = None  # 电流积分累计（输出积分）

    def reset(self):
        self.__init__(self.G_nv)

    def get_conductance(self, pulse_count):
        pulse_count = int(round(pulse_count))
        return self.G_nv.get(pulse_count, 0.0)

    def pulse_to_conductance(self, pulse_vec):
        return np.array([self.get_conductance(p) for p in pulse_vec])

    def forward(self, V_input: np.ndarray, W_pulses: np.ndarray):
        """
        V_input: 输入电压向量（模拟输入）
        W_pulses: 权重脉冲表示，需通过NVM映射为电导
        """
        G_weight = self.pulse_to_conductance(W_pulses)
        if self.T == 0:
            I_out = np.dot(V_input, G_weight)  # 电流输出 = 电压 * 电导
            self.Q_acc = V_input.copy()
            self.G_acc = G_weight.copy()
            self.I_acc = I_out
            self.T += 1
            return I_out
        else:
            I_new = np.dot(V_input, G_weight) + np.dot(V_input, self.G_acc) + np.dot(self.Q_acc, G_weight)
            delta_I = I_new / self.T - self.I_acc / (self.T * (self.T + 1))  # 误差补偿项
            self.I_acc += I_new
            self.Q_acc += V_input
            self.G_acc += G_weight
            self.T += 1
            return delta_I


class SimTwoSideNeuron:
    """
    硬件建模 TwoSideNeuron：使用易失性忆阻器模拟五位状态码-电导关系。
    - 电导来自状态映射（Volatile memory）。
    - 膜电压积分 → 多级阈值判断 → 放电脉冲输出（多位开关映射）。
    """
    def __init__(self, G_v_map, G_pos_base=1.0, G_neg_base=1.0, threshold_levels=3, step_ratio=2):
        self.G_v_map = G_v_map  # 状态码 -> 电导值映射表（VMM）
        self.V_pos_base = G_pos_base  # 正阈值（对应正导通器件开关门限）
        self.V_neg_base = G_neg_base  # 负阈值
        self.N_level = threshold_levels  # 多阈值层数
        self.alpha = step_ratio  # 每一层阈值倍数关系
        self.V_mem = 0.0  # 当前膜电压（由输入电流积分而来）

    def reset(self):
        self.V_mem = 0.0

    def forward(self, V_state_code: str):
        """
        V_state_code: 五位易失性状态码，例如 '10110'
        返回：当前脉冲发放列表 + 更新后的膜电位
        """
        I_input = self.G_v_map.get(V_state_code, 0.0)  # 输入电流由器件导通状态决定
        self.V_mem += I_input

        spikes = []
        V_thr_p = self.V_pos_base * (self.alpha ** (self.N_level - 1))
        V_thr_n = self.V_neg_base * (self.alpha ** (self.N_level - 1))

        for _ in range(self.N_level):
            spike_p, spike_n = 0.0, 0.0
            if self.V_mem >= V_thr_p:
                spike_p = V_thr_p
                self.V_mem -= V_thr_p
            elif self.V_mem <= -V_thr_n:
                spike_n = -V_thr_n
                self.V_mem += V_thr_n
            spikes.append(spike_p)
            spikes.append(spike_n)
            V_thr_p /= self.alpha
            V_thr_n /= self.alpha

        return spikes, self.V_mem


# 映射表加载函数：CSV -> 字典
def load_nv_mapping_from_csv(file_path):
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return {int(row["脉冲个数"]): float(row["电导(μS)"]) for _, row in df.iterrows()}


def load_v_mapping_from_csv(file_path):
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    df['State'] = df['State'].astype(str).str.strip().str.zfill(5)  # 添加补零
    return {row['State']: float(row['电导均值 μS']) for _, row in df.iterrows()}


