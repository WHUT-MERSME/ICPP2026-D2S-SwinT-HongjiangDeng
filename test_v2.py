import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
from operator import truediv # Needed for Average Accuracy calculation
import time
import random

# 项目特定导入
from swin_hsi_model_v2 import HSISwinModel
from ECMT_uitls import trans_utils  # 确保路径正确
from timm.utils import AverageMeter, accuracy as timm_accuracy  # 使用timm的AverageMeter和accuracy

# --- 数据集配置 ---
# 选择要使用的数据集: 'IP', 'SA', 'PU', 'XZ', 'HC', 'HU', 'LK', 'XA'
CHOSEN_DATASET = 'ZY_HHK_2021'

DATASET_SPECIFICS = {
    'IP': {
        'num_classes': 16,
        'data_file': 'Indian_pines_corrected.mat',
        'gt_file': 'Indian_pines_gt.mat',
        'target_names': ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                       'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                       'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                       'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                       'Stone-Steel-Towers']
    },
    'SA': {
        'num_classes': 16,
        'data_file': 'Salinas_corrected.mat',
        'gt_file': 'Salinas_gt.mat',
        'target_names': ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained',
                        'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                        'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']
    },
    'PU': {
        'num_classes': 9,
        'data_file': 'PaviaU.mat', # 帕维亚大学数据集通常没有广泛分发的 'corrected' 版本
        'gt_file': 'PaviaU_gt.mat',
        'target_names': ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    },
    'PC': { # Pavia Centre
        'num_classes': 9, # Inferred from target_names length
        'data_file': 'Pavia.mat', # Hypothesized, ensure this file exists
        'gt_file': 'Pavia_gt.mat',       # Hypothesized, ensure this file exists
        'target_names': ['Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks', 'Bitumen', 'Tiles',
                         'Shadows', 'Meadows', 'Bare Soil',]
    },
    'HU': { # Houston2013
        'num_classes': 15,
        'data_file': 'HSI.mat',
        'gt_file': 'gt.mat',
        'target_names': ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water',
                         'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking Lot 1',
                         'Parking Lot 2', 'Tennis Court', 'Running Track']
    },
    'GF_YC': { # Houston2013
        'num_classes': 7,
        'data_file': 'GF_YC_data.mat',
        'gt_file': 'GF_YC_gt.mat',
        'target_names': ['Architecture', 'River', 'Reed', 'Paddy', 'Fallow land',
                         'Sea', 'Offshore water']
    },
    'ZY_HHK_2021': { # ZY_HHK_2021
        'num_classes': 8,
        'data_file': 'ZY_HHK_data108_20210929.mat',
        'gt_file': 'ZY_HHK_gt108_20210929.mat',
        'target_names': ['Reed', 'Salt flat filtration pond', 'Salt flat evaporation pond',
                         'Salt flat', 'Suaeda', 'River', 'Sea', 'Tide ditch']
    },
    'chongming': { # chongming
        'num_classes': 9,
        'data_file': 'chongming_hsi.mat',
        'gt_file': 'chongming_gt_1027.mat',
        'target_names': ['Aquaculture pond', 'Farmland', 'Forest', 'Paddy', 'Salt marsh', 'Reed',
                         'Offshore water', 'Sea', 'Architecture']
    },
    'shuangtai': { # shuangtai
        'num_classes': 8,
        'data_file': 'shuangtai_hsi.mat',
        'gt_file': 'shuangtai_gt_1027.mat',
        'target_names': ['River', 'Farmland', 'Forest', 'Reed', 'Slat marsh', 'Offshore water',
                         'Sea', 'Architecture']
    },

}

if CHOSEN_DATASET not in DATASET_SPECIFICS:
    raise ValueError(f"数据集 {CHOSEN_DATASET} 未在 DATASET_SPECIFICS 中配置。")

# --- 全局配置 ---
dataset_config = DATASET_SPECIFICS[CHOSEN_DATASET]

DATA_DIR = os.path.join('DataSet', CHOSEN_DATASET)
# 预训练ANN模型的路径，现在基于 CHOSEN_DATASET
PRETRAINED_ANN_PATH = os.path.join('d2s_result', CHOSEN_DATASET, 'best_model.pth')
# 修改输出目录以区分不同数据集的测试结果
RESULT_DIR = os.path.join('d2s_result', 'test_ecmt', CHOSEN_DATASET)
NUM_CLASSES = dataset_config['num_classes'] # 类别数现在是动态的
_DATA_FILE_NAME = dataset_config['data_file'] # 用于加载数据
_GT_FILE_NAME = dataset_config['gt_file']     # 用于加载标签

PATCH_SIZE = 25  # 图像块大小
PCA_COMPONENTS = 30 # PCA降维后的组件数
TEST_BATCH_SIZE = 256
SNN_TIMESTEPS = 4
ECMT_PERCENT = 0.99
OVERALL_TEST_RATIO_FROM_FULL_DATA = 0.7 # 从完整数据中划分测试集的比例
RANDOM_STATE_FOR_FULL_DATA_SPLIT = 345 # 划分完整数据时使用的随机种子
CALIBRATION_SET_SIZE_FROM_TRAIN_PORTION = 512 # 从训练部分中提取的校准集大小
CALIBRATION_BATCH_SIZE = 32

# Adjust patch_size for specific datasets if known to be better
if CHOSEN_DATASET != 'IP':
    PCA_COMPONENTS = 30

class ArgsECMT:
    def __init__(self):
        self.model = 'vit'
        self.linear_num = 8
        self.qkv_num = 8
        self.softmax_num = 8
        self.softmax_p = 0.0125


args_ecmt_instance = ArgsECMT()


class ArgsTestScript:
    def __init__(self):
        self.monitor = True  # 控制SOP监控


args_test_script_instance = ArgsTestScript()

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 辅助函数 ---
# (set_random_seed, applyPCA, padWithZeros, createImageCubes 保持不变)
def set_random_seed(seed_val):  # (内容不变)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def applyPCA(X, num_components):  # (内容不变)
    new_X = np.reshape(X, (-1, X.shape[2]))
    pca_model = PCA(n_components=num_components, whiten=True)
    new_X = pca_model.fit_transform(new_X)
    new_X = np.reshape(new_X, (X.shape[0], X.shape[1], num_components))
    return new_X, pca_model


def padWithZeros(X, margin=2):  # (内容不变)
    new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset, y_offset = margin, margin
    new_X[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return new_X

def split_data_per_class(X_cube, y_cube, train_samples_per_class=30, random_seed=345):
    """
    为每个类别选择固定数量的样本作为训练集，其余作为测试集。

    Args:
        X_cube (numpy.ndarray): 包含所有有效图像块的数据，形状为 (N, H, W, C)。
        y_cube (numpy.ndarray): 对应的标签（0-based），形状为 (N,)。
        train_samples_per_class (int): 每个类别选作训练样本的数量。
        random_seed (int): 用于复现的随机种子。

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    np.random.seed(random_seed)
    train_indices = []
    test_indices = []

    num_classes = int(np.max(y_cube) + 1)

    for class_id in range(num_classes):
        # 找到当前类别的所有样本索引
        indices_of_class = np.where(y_cube == class_id)[0]

        # 如果当前类别的样本总数不足以提供训练样本，则发出警告
        if len(indices_of_class) < train_samples_per_class:
            print(
                f"警告: 类别 {class_id} 只有 {len(indices_of_class)} 个样本，少于指定的训练样本数 {train_samples_per_class}。将使用所有样本作为训练数据。")
            # 在这种情况下，该类别没有测试样本
            train_indices.extend(indices_of_class)
            continue  # 继续处理下一个类别

        # 随机打乱当前类别的索引
        np.random.shuffle(indices_of_class)

        # 选择前 N 个作为训练索引，其余作为测试索引
        train_indices.extend(indices_of_class[:train_samples_per_class])
        test_indices.extend(indices_of_class[train_samples_per_class:])

    # 根据索引提取数据
    X_train = X_cube[train_indices]
    y_train = y_cube[train_indices]
    X_test = X_cube[test_indices]
    y_test = y_cube[test_indices]

    return X_train, X_test, y_train, y_test

def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    """
    创建图像块和对应的标签。
    这个优化版本只为带有有效标签的像素创建立方体，以节省内存。
    """
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    # 默认行为是移除背景像素，这是最高效的方式
    if removeZeroLabels:
        # 1. 首先，只找到带有有效标签（非零）的像素的坐标
        labeled_pixels_r, labeled_pixels_c = np.where(y > 0)
        num_effective_samples = len(labeled_pixels_r)

        if num_effective_samples == 0:
            print("警告: 在标签图中没有找到任何有效（非零）的像素。")
            return np.array([]), np.array([])

        # 2. 仅为这些有效像素分配内存
        print(f"找到 {num_effective_samples} 个有效（非背景）像素。将为它们创建图像块。")
        patchesData = np.zeros((num_effective_samples, windowSize, windowSize, X.shape[2]), dtype=X.dtype)
        patchesLabels = np.zeros(num_effective_samples, dtype=y.dtype)

        # 3. 遍历这些有效像素的坐标来创建图像块
        for i, (r_orig, c_orig) in enumerate(zip(labeled_pixels_r, labeled_pixels_c)):
            # 在 zeroPaddedX 中的对应中心点
            r_padded_center = r_orig + margin
            c_padded_center = c_orig + margin

            patch = zeroPaddedX[r_padded_center - margin : r_padded_center + margin + 1,
                                  c_padded_center - margin : c_padded_center + margin + 1]
            patchesData[i, :, :, :] = patch
            patchesLabels[i] = y[r_orig, c_orig]

        # 4. 将标签从1-based转换为0-based
        patchesLabels -= 1

    else: # 如果不移除零标签，则使用原始的、可能导致内存问题的逻辑
        print("警告: removeZeroLabels=False，将为所有像素创建立方体，这可能导致内存错误。")
        num_total_pixels = y.shape[0] * y.shape[1]
        patchesData = np.zeros((num_total_pixels, windowSize, windowSize, X.shape[2]), dtype=X.dtype)
        patchesLabels = np.zeros(num_total_pixels, dtype=y.dtype)
        patch_idx = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patch_idx, :, :, :] = patch
                patchesLabels[patch_idx] = y[r-margin, c-margin]
                patch_idx = patch_idx + 1

    return patchesData, patchesLabels


# --- SNN评估函数 (整合能耗指标, 使用AverageMeter) ---
@torch.no_grad()
def evaluate_snn_ecmt(data_loader, model, device, total_timesteps, ecmt_args, test_args):
    # 使用 timm.utils.AverageMeter 记录指标
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()  # 如果需要Top5准确率
    # loss_meter = AverageMeter() # 如果计算损失

    header = f'测试 ECMT SNN ({CHOSEN_DATASET}):'
    print_freq = 10

    sop_monitor = trans_utils.SOPMonitor(model, type=1)
    monitor_enabled = getattr(test_args, 'monitor', False)
    sop_monitor.enable() if monitor_enabled else sop_monitor.disable()

    model.eval()

    acc_over_timesteps_sum = np.zeros(total_timesteps, dtype=float)
    total_samples_processed = 0
    dataset_total_actual_sops_per_t = np.zeros(total_timesteps, dtype=float)
    dataset_total_potential_ops_per_t = np.zeros(total_timesteps, dtype=float)

    num_batches = len(data_loader)
    start_eval_time = time.time() # 记录评估开始时间

    # 用于计算每个时间步的AA和Kappa
    all_targets_collected = []
    all_preds_collected_per_timestep = [[] for _ in range(total_timesteps)]

    for batch_idx, batch_data in enumerate(data_loader):
        batch_start_time = time.time() # 单个批次开始时间
        images, target = batch_data[0], batch_data[-1]
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        current_batch_size = images.shape[0]

        all_targets_collected.extend(target.cpu().numpy()) # 收集当前批次的目标

        if monitor_enabled: sop_monitor.clear_recorded_data()
        trans_utils.reset_net(model)

        outputs_at_each_timestep = []
        batch_actual_sops_t = np.zeros(total_timesteps, dtype=float)
        batch_potential_ops_t = np.zeros(total_timesteps, dtype=float)

        for t_step in range(total_timesteps):
            out_one_step_raw = model(images)
            decoded_output_this_step = None
            if isinstance(out_one_step_raw, list) and len(out_one_step_raw) > 0:
                if all(isinstance(t, torch.Tensor) and t.shape == out_one_step_raw[0].shape for t in out_one_step_raw):
                    try:
                        decoded_output_this_step = torch.sum(torch.stack(out_one_step_raw), dim=0)
                    except:
                        decoded_output_this_step = out_one_step_raw[0] if isinstance(out_one_step_raw[0],
                                                                                     torch.Tensor) else None
                else:
                    decoded_output_this_step = out_one_step_raw[0] if isinstance(out_one_step_raw[0],
                                                                                 torch.Tensor) else None
            elif isinstance(out_one_step_raw, torch.Tensor):
                decoded_output_this_step = out_one_step_raw

            valid_output = (decoded_output_this_step is not None and
                            decoded_output_this_step.dim() == 2 and
                            decoded_output_this_step.shape[0] == current_batch_size and
                            decoded_output_this_step.shape[1] == NUM_CLASSES)

            current_step_output = decoded_output_this_step.clone() if valid_output else torch.zeros(current_batch_size, NUM_CLASSES, device=device)
            outputs_at_each_timestep.append(current_step_output)

            # 收集当前时间步的预测
            _, preds_t = torch.max(current_step_output, 1)
            all_preds_collected_per_timestep[t_step].extend(preds_t.cpu().numpy())


            if monitor_enabled:
                current_t_actual_sop, current_t_potential_op = 0.0, 0.0
                for sop_val, pot_op_val in sop_monitor.records:
                    current_t_actual_sop += sop_val.item() if isinstance(sop_val, torch.Tensor) else sop_val
                    current_t_potential_op += pot_op_val.item() if isinstance(pot_op_val, torch.Tensor) else pot_op_val
                batch_actual_sops_t[t_step] = current_t_actual_sop
                batch_potential_ops_t[t_step] = current_t_potential_op
                sop_monitor.clear_recorded_data()

        if len(outputs_at_each_timestep) == total_timesteps:
            total_samples_processed += current_batch_size
            current_batch_acc_per_t_vals = []
            for t_idx_for_acc in range(total_timesteps):
                acc1_t, _ = timm_accuracy(outputs_at_each_timestep[t_idx_for_acc], target, topk=(1, 5))
                current_batch_acc_per_t_vals.append(acc1_t.item())
            acc_over_timesteps_sum += np.array(current_batch_acc_per_t_vals) * current_batch_size

            final_output_metrics = outputs_at_each_timestep[-1]
            acc1_final_batch, acc5_final_batch = timm_accuracy(final_output_metrics, target, topk=(1, 5))
            acc1_meter.update(acc1_final_batch.item(), n=current_batch_size)
            acc5_meter.update(acc5_final_batch.item(), n=current_batch_size)

        if monitor_enabled:
            dataset_total_actual_sops_per_t += batch_actual_sops_t
            dataset_total_potential_ops_per_t += batch_potential_ops_t

        batch_end_time = time.time() # 单个批次结束时间

        if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == num_batches:
            iters_done_in_cycle = print_freq if (batch_idx + 1) % print_freq == 0 else (batch_idx + 1) % print_freq
            # 使用自评估开始以来的总时间计算平均迭代时间
            avg_time_per_iter = (batch_end_time - start_eval_time) / (batch_idx + 1)
            eta_seconds = avg_time_per_iter * (num_batches - (batch_idx + 1))
            eta_string = str(time.strftime("%H:%M:%S", time.gmtime(eta_seconds)))
            current_iter_acc1 = acc1_final_batch.item() if 'acc1_final_batch' in locals() and isinstance(acc1_final_batch, torch.Tensor) else 0.0

            print(f"{header} [{batch_idx + 1}/{num_batches}] "
                  f"ETA: {eta_string} "
                  # f"loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) " # 如果计算loss
                  f"Acc@1_current_batch: {current_iter_acc1:.2f} "
                  f"Acc@1_avg: {acc1_meter.avg:.2f} "
                  f"Acc@5_avg: {acc5_meter.avg:.2f} ")
            end_time = time.time()  # 重置下次打印周期的时间

    # --- 循环结束后，计算并打印最终统计信息 ---
    print(f'* Final Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    oa_per_timestep_list = (
                acc_over_timesteps_sum / total_samples_processed).tolist() if total_samples_processed > 0 else [0.0] * total_timesteps

    # 计算每个时间步的 AA 和 Kappa
    aa_per_timestep_list = []
    kappa_per_timestep_list = []

    if total_samples_processed > 0:
        print('\n计算每个时间步的OA, AA, Kappa...')
        for t_idx in range(total_timesteps):
            preds_t = np.array(all_preds_collected_per_timestep[t_idx])
            targets_all = np.array(all_targets_collected)

            # 计算AA
            cm_t = confusion_matrix(targets_all, preds_t, labels=list(range(NUM_CLASSES)))
            list_diag_t = np.diag(cm_t)
            list_raw_sum_t = np.sum(cm_t, axis=1)
            each_acc_t = np.nan_to_num(truediv(list_diag_t, list_raw_sum_t)) * 100
            aa_t = np.mean(each_acc_t)
            aa_per_timestep_list.append(aa_t)

            # 计算Kappa
            kappa_t = cohen_kappa_score(targets_all, preds_t) * 100
            kappa_per_timestep_list.append(kappa_t)

            print(f'  T={t_idx + 1}: OA={oa_per_timestep_list[t_idx]:.2f}%, AA={aa_t:.2f}%, Kappa={kappa_t:.2f}%')
    else:
        aa_per_timestep_list = [0.0] * total_timesteps
        kappa_per_timestep_list = [0.0] * total_timesteps


    sop_info_to_return, energy_info_to_return = None, None
    if monitor_enabled and total_samples_processed > 0:  # (能耗统计逻辑与上一版本相同)
        print('--- SOP & Energy Statistics ---')
        cumulative_actual_sops = np.cumsum(dataset_total_actual_sops_per_t)
        cumulative_potential_ops = np.cumsum(dataset_total_potential_ops_per_t)
        sop_info_to_return = []
        energy_info_to_return = []

        header_fr = "T | Avg Fire Rate (Cumul.) | Energy (Est. Cumul.)"
        print(header_fr + "\n" + "-" * len(header_fr))
        for t_idx_for_energy in range(total_timesteps):  # 重命名循环变量
            rate = cumulative_actual_sops[t_idx_for_energy] / cumulative_potential_ops[t_idx_for_energy] if \
            cumulative_potential_ops[t_idx_for_energy] > 0 else 0
            energy_factor = 0.9 / 4.6
            energy_val = (t_idx_for_energy + 1) * energy_factor * rate
            sop_info_to_return.append(rate)
            energy_info_to_return.append(energy_val)
            print(f"{t_idx_for_energy + 1:^1} | {rate:22.4f} | {energy_val:22.4f}")

    return acc1_meter.avg, oa_per_timestep_list, aa_per_timestep_list, kappa_per_timestep_list, sop_info_to_return, energy_info_to_return


# --- 主函数 ---
def main():
    set_random_seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True) # 使用新的 RESULT_DIR
    print(f"测试数据集: {CHOSEN_DATASET}")
    print(f"数据加载自: {DATA_DIR}")
    print(f"预训练ANN模型加载自: {PRETRAINED_ANN_PATH}")
    print(f"测试结果将保存至: {RESULT_DIR}")
    print(f"类别数: {NUM_CLASSES}")


    # 1. 加载并分割数据
    if not os.path.exists(os.path.join(DATA_DIR, _DATA_FILE_NAME)):
        print(f"错误: 数据文件 {_DATA_FILE_NAME} 在路径 {DATA_DIR} 未找到。")
        return
    if not os.path.exists(os.path.join(DATA_DIR, _GT_FILE_NAME)):
        print(f"错误: 标签文件 {_GT_FILE_NAME} 在路径 {DATA_DIR} 未找到。")
        return

    X_mat_loaded = sio.loadmat(os.path.join(DATA_DIR, _DATA_FILE_NAME))
    y_mat_loaded = sio.loadmat(os.path.join(DATA_DIR, _GT_FILE_NAME))

    data_key = None
    for key_d in X_mat_loaded.keys():
        if not key_d.startswith('__'): data_key = key_d; break
    gt_key = None
    for key_g in y_mat_loaded.keys():
        if not key_g.startswith('__'): gt_key = key_g; break

    if data_key is None: print(f"错误: 无法在 {_DATA_FILE_NAME} 文件中找到数据键。"); return
    if gt_key is None: print(f"错误: 无法在 {_GT_FILE_NAME} 文件中找到标签键。"); return

    X_mat_data = X_mat_loaded[data_key]
    y_mat_data = y_mat_loaded[gt_key]

    print(f"加载数据: {data_key} (形状: {X_mat_data.shape}), 标签: {gt_key} (形状: {y_mat_data.shape})")

    X_pca, _ = applyPCA(X_mat_data, num_components=PCA_COMPONENTS)
    X_full_cube, y_full_cube = createImageCubes(X_pca, y_mat_data, windowSize=PATCH_SIZE)

    if X_full_cube.shape[0] == 0: print("错误: 创建的图像块为空。"); return

    print("正在使用新策略划分数据：每个类别30个样本用于训练，以获取测试集...")
    # X_train_p, X_test_cube, y_train_p, y_test_cube = split_data_per_class(
    #     X_full_cube, y_full_cube, train_samples_per_class=30, random_seed=345
    # )

    stratify_option = y_full_cube if len(np.unique(y_full_cube)) > 1 else None
    X_train_p, X_test_cube, y_train_p, y_test_cube = train_test_split(
        X_full_cube, y_full_cube,
        test_size=OVERALL_TEST_RATIO_FROM_FULL_DATA,
        random_state=RANDOM_STATE_FOR_FULL_DATA_SPLIT,
        stratify=stratify_option
    )

    num_calib = min(CALIBRATION_SET_SIZE_FROM_TRAIN_PORTION, len(X_train_p))
    if num_calib > 0 and num_calib <= len(X_train_p): # 确保num_calib不大于可用数据
        stratify_calib = y_train_p if np.unique(y_train_p).size > 1 else None
        try:
            if num_calib < len(X_train_p): # 仅当需要分割时才分割
                 cal_idx, _ = train_test_split(np.arange(len(X_train_p)), train_size=num_calib, random_state=SEED, stratify=stratify_calib)
                 X_cal_src, y_cal_src = X_train_p[cal_idx], y_train_p[cal_idx]
            else: # 使用全部
                 X_cal_src, y_cal_src = X_train_p, y_train_p
        except ValueError: # 分层失败，尝试不分层
            print("警告：校准集分层失败，尝试不分层分割。")
            if num_calib < len(X_train_p):
                cal_idx, _ = train_test_split(np.arange(len(X_train_p)), train_size=num_calib, random_state=SEED)
                X_cal_src, y_cal_src = X_train_p[cal_idx], y_train_p[cal_idx]
            else:
                X_cal_src, y_cal_src = X_train_p, y_train_p
    elif num_calib > 0:
        X_cal_src, y_cal_src = X_train_p, y_train_p
    else:
        print("警告: 校准集大小为0或训练部分为空。跳过TestNeuron校准阶段。")
        X_cal_src, y_cal_src = np.array([]), np.array([])

    if len(X_cal_src) > 0:
        X_cal_tensor = torch.FloatTensor(X_cal_src).permute(0, 3, 1, 2)
        y_cal_tensor = torch.LongTensor(y_cal_src)
        calib_loader = DataLoader(TensorDataset(X_cal_tensor, y_cal_tensor), batch_size=CALIBRATION_BATCH_SIZE,
                                  shuffle=False)
    else:
        calib_loader = None  # 表示没有校准数据

    X_test_tensor = torch.FloatTensor(X_test_cube).permute(0, 3, 1, 2)
    y_test_tensor = torch.LongTensor(y_test_cube)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=TEST_BATCH_SIZE, shuffle=False)

    # 2. 加载预训练ANN模型
    ann_model = HSISwinModel(num_classes=NUM_CLASSES, in_chans=PCA_COMPONENTS, img_size=PATCH_SIZE, drop_rate=0.0).to(
        DEVICE)
    try:
        ann_model.load_state_dict(torch.load(PRETRAINED_ANN_PATH, map_location=DEVICE))
        print(f"Successfully loaded ANN weights from {PRETRAINED_ANN_PATH}")
    except Exception as e:
        print(f"Error loading ANN weights: {e}. Exiting.")
        return
    ann_model.eval()
    # print(ann_model)

    # 3. ECMT SNN转换
    print("ECMT: Replacing MyTestPlace with TestNeuron...")
    snn_model = trans_utils.replace_test_by_testneuron(ann_model, percent=ECMT_PERCENT)
    snn_model.to(DEVICE)

    if calib_loader:  # 仅当有校准数据时才进行阈值计算
        print("ECMT: Calculating thresholds using TestNeurons on calibration data...")
        with torch.no_grad():
            cal_iter_count = 0
            max_cal_iters = 10
            for images_cal, _ in calib_loader:
                _ = snn_model(images_cal.to(DEVICE))
                cal_iter_count +=1
                if cal_iter_count >= max_cal_iters and len(calib_loader) > max_cal_iters : # 避免在校准集很小时过早停止
                    print(f"ECMT: 已在 {max_cal_iters} 个校准批次上计算阈值（或已处理完所有校准数据）。")
                    break
        print("ECMT: 阈值计算完成。")
    else:
        print(
            "ECMT: Skipping threshold calculation due to no calibration data. TestNeurons will use default/uninitialized thresholds.")

    print("ECMT: Converting to SNN structure...")
    snn_model = trans_utils.replace_nonlinear_by_neuron(snn_model)
    snn_model = trans_utils.replace_at_by_neuron(snn_model)
    snn_model = trans_utils.replace_testneuron_by_twosideneuron(snn_model, args_ecmt_instance)
    snn_model.to(DEVICE).eval()
    # print(snn_model)

    # 4. 测试SNN模型
    print('\n=== Testing ECMT SNN Model on actual test set ===')
    if len(test_loader.dataset) == 0:
        print("Error: Test dataset is empty. Cannot perform SNN evaluation.")
        return

    oa_final_timestep, oa_t_list, aa_t_list, kappa_t_list, sop_info, energy_info = evaluate_snn_ecmt(
        test_loader, snn_model, DEVICE, SNN_TIMESTEPS, args_ecmt_instance, args_test_script_instance
    )

    print(f"\nECMT SNN最终测试总体准确率 (OA at T={SNN_TIMESTEPS}): {oa_final_timestep:.2f}%")
    if oa_t_list: print(f"每个时间步的OA: [{', '.join([f'{a:.2f}%' for a in oa_t_list])}]")
    if aa_t_list: print(f"每个时间步的AA: [{', '.join([f'{a:.2f}%' for a in aa_t_list])}]")
    if kappa_t_list: print(f"每个时间步的Kappa: [{', '.join([f'{k:.2f}%' for k in kappa_t_list])}]")

    if sop_info and energy_info and args_test_script_instance.monitor:
        print(f"Final Cumulative Avg Firing Rate: {sop_info[-1]:.4f}")
        print(f"Final Estimated Cumulative Energy: {energy_info[-1]:.4f} (relative units)")

    report_filename = os.path.join(RESULT_DIR, f"ecmt_snn_report_T{SNN_TIMESTEPS}.txt")
    with open(report_filename, 'w') as f:
        f.write(f"ECMT SNN 模型 ({CHOSEN_DATASET}) 最终测试总体准确率 (OA at T={SNN_TIMESTEPS}): {oa_final_timestep:.2f}%\n\n")

        if oa_t_list and aa_t_list and kappa_t_list:
            f.write('各时间步详细指标:\n')
            f.write(f"{'T':<3} | {'Overall Acc (%)':<18} | {'Average Acc (%)':<18} | {'Kappa (%)':<10}\n")
            f.write("-" * (3 + 3 + 18 + 3 + 18 + 3 + 10) + "\n")
            for t_report in range(SNN_TIMESTEPS):
                oa_val = oa_t_list[t_report] if t_report < len(oa_t_list) else float('nan')
                aa_val = aa_t_list[t_report] if t_report < len(aa_t_list) else float('nan')
                kappa_val = kappa_t_list[t_report] if t_report < len(kappa_t_list) else float('nan')
                f.write(f"{t_report + 1:<3} | {oa_val:^18.2f} | {aa_val:^18.2f} | {kappa_val:^10.2f}\n")

        if args_test_script_instance.monitor and sop_info and energy_info:
            f.write('\n--- SOP & Energy Statistics ---\n')
            f.write("T | Avg Fire Rate (Cumulative) | Energy (Est. Cumulative)\n")
            f.write("--------------------------------------------------------\n")
            for t_report_energy in range(SNN_TIMESTEPS):  # 重命名循环变量
                f.write(
                    f"{t_report_energy + 1:^1} | {sop_info[t_report_energy]:26.4f} | {energy_info[t_report_energy]:26.4f}\n")
    print(f"Detailed report saved to {report_filename}")


if __name__ == '__main__':
    main()
