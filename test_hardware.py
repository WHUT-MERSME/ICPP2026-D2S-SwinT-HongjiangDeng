import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
from operator import truediv  # Needed for average accuracy calculation.
import time
import random

from swin_hsi_model_v2 import HSISwinModel
from hardware_simulation import trans_utils_hardware
from hardware_simulation import hardware_metrics_eval
from timm.utils import AverageMeter, accuracy as timm_accuracy  

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
        'data_file': 'PaviaU.mat', 
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

dataset_config = DATASET_SPECIFICS[CHOSEN_DATASET]

DATA_DIR = os.path.join('DataSet', CHOSEN_DATASET)
PRETRAINED_ANN_PATH = os.path.join('d2s_result', CHOSEN_DATASET, 'best_model.pth')
RESULT_DIR = os.path.join('d2s_result', 'test_ecmt', CHOSEN_DATASET)
HARDWARE_RESULT_DIR = os.path.join('hareware_result', CHOSEN_DATASET)
HARDWARE_PERF_CONFIG_DIR = os.path.join('hardware_simulation')
NUM_CLASSES = dataset_config['num_classes'] 
_DATA_FILE_NAME = dataset_config['data_file'] 
_GT_FILE_NAME = dataset_config['gt_file']     

PATCH_SIZE = 25  
PCA_COMPONENTS = 30 
TEST_BATCH_SIZE = 256
SNN_TIMESTEPS = 3
ECMT_PERCENT = 0.99
OVERALL_TEST_RATIO_FROM_FULL_DATA = 0.7 
RANDOM_STATE_FOR_FULL_DATA_SPLIT = 345 
CALIBRATION_SET_SIZE_FROM_TRAIN_PORTION = 512 
CALIBRATION_BATCH_SIZE = 32

# Keep this override explicit so dataset-specific settings stay easy to audit.
if CHOSEN_DATASET != 'IP':
    PCA_COMPONENTS = 30

class ArgsECMT:
    """ECMT conversion hyperparameters shared by calibration and inference."""

    def __init__(self):
        self.model = 'vit'
        self.linear_num = 8
        self.qkv_num = 8
        self.softmax_num = 8
        self.softmax_p = 0.0125


args_ecmt_instance = ArgsECMT()


class ArgsHardwareSim:
    """Hardware simulation master switch and per-module replacement switches."""

    def __init__(self):
        # Supported modes:
        # - "software": ECMT SNN with hardware-equivalent modules forced back to software semantics.
        # - "table_only": current hardware simulation using nv/v conductance lookup tables.
        # - "full": table-driven simulation plus full hardware-parameter path when available.
        self.hardware_mode = "full"

        self.enable_hardware_simulation = True
        self.enable_neuron_hardware = True
        self.use_full_hardware_params = False

        self.fold_batchnorm = True
        self.replace_conv = True
        self.replace_linear = True
        self.replace_softmax = True
        self.replace_norm = True
        self.replace_batchnorm = True
        self.replace_layernorm = True
        self.replace_relu = True
        self.replace_pool = True
        self.replace_classifier = True

        self.conv_input_quant_bits = 8
        self.conv_weight_quant_bits = 8
        self.conv_use_input_quant = True
        self.conv_use_weight_quant = True
        self.conv_mapping_mode = "lut_linear"
        self.conv_scale_mode = "wmax_gmax"

        self.norm_quant_bits = 8
        self.softmax_quant_bits = 8
        self.relu_quant_bits = 8
        self.pool_quant_bits = 8

        self.linear_input_quant_bits = 8
        self.linear_weight_quant_bits = 8
        self.linear_use_input_quant = True
        self.linear_use_weight_quant = True
        self.linear_mapping_mode = "lut_linear"
        self.linear_scale_mode = "wmax_gmax"


args_hardware_sim_instance = ArgsHardwareSim()


class ArgsTestScript:
    """Script-level options unrelated to model structure."""

    def __init__(self):
        self.monitor = True
        self.enable_hardware_perf_estimation = True


args_test_script_instance = ArgsTestScript()
args_hardware_perf_instance = hardware_metrics_eval.load_hardware_perf_config(HARDWARE_PERF_CONFIG_DIR)
args_hardware_perf_instance._config_enable = bool(getattr(args_hardware_perf_instance, "enable", True))


def apply_hardware_mode(hw_args, test_args, perf_args):
    """Derive all runtime hardware switches from a single high-level mode."""

    mode = str(getattr(hw_args, "hardware_mode", "table_only")).strip().lower()
    valid_modes = {"software", "table_only", "full"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported hardware_mode: {mode}. Expected one of {sorted(valid_modes)}")

    hw_args.hardware_mode = mode
    hw_args.enable_neuron_hardware = mode != "software"
    hw_args.enable_hardware_simulation = mode != "software"
    hw_args.use_full_hardware_params = mode == "full"

    perf_cfg_enabled = bool(getattr(perf_args, "_config_enable", getattr(perf_args, "enable", True)))
    perf_args.enable = bool(
        perf_cfg_enabled
        and getattr(test_args, "enable_hardware_perf_estimation", True)
        and mode == "full"
    )


apply_hardware_mode(args_hardware_sim_instance, args_test_script_instance, args_hardware_perf_instance)

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed_val):
    """Make preprocessing, calibration, and evaluation reproducible."""

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def applyPCA(X, num_components):
    """Apply PCA along the spectral dimension."""

    new_X = np.reshape(X, (-1, X.shape[2]))
    pca_model = PCA(n_components=num_components, whiten=True)
    new_X = pca_model.fit_transform(new_X)
    new_X = np.reshape(new_X, (X.shape[0], X.shape[1], num_components))
    return new_X, pca_model


def padWithZeros(X, margin=2):
    """Pad the spatial plane so edge pixels can form full patches."""

    new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset, y_offset = margin, margin
    new_X[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return new_X

def split_data_per_class(X_cube, y_cube, train_samples_per_class=30, random_seed=345):
    """Build a per-class split with a fixed number of training samples."""

    np.random.seed(random_seed)
    train_indices = []
    test_indices = []

    num_classes = int(np.max(y_cube) + 1)

    for class_id in range(num_classes):
        indices_of_class = np.where(y_cube == class_id)[0]

        if len(indices_of_class) < train_samples_per_class:
            print(
                f"警告: 类别 {class_id} 只有 {len(indices_of_class)} 个样本，少于指定的训练样本数 {train_samples_per_class}。将使用该类全部样本作为训练数据。"
            )
            train_indices.extend(indices_of_class)
            continue  

        np.random.shuffle(indices_of_class)

        train_indices.extend(indices_of_class[:train_samples_per_class])
        test_indices.extend(indices_of_class[train_samples_per_class:])

    X_train = X_cube[train_indices]
    y_train = y_cube[train_indices]
    X_test = X_cube[test_indices]
    y_test = y_cube[test_indices]

    return X_train, X_test, y_train, y_test

def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    """Extract spatial-spectral patches centered on labeled pixels."""

    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    if removeZeroLabels:
        labeled_pixels_r, labeled_pixels_c = np.where(y > 0)
        num_effective_samples = len(labeled_pixels_r)

        if num_effective_samples == 0:
            print("警告: 在标签图中没有找到任何有效的非零像素。")
            return np.array([]), np.array([])

        print(f"找到 {num_effective_samples} 个有效的非背景像素，将为它们创建图像块。")
        patchesData = np.zeros((num_effective_samples, windowSize, windowSize, X.shape[2]), dtype=X.dtype)
        patchesLabels = np.zeros(num_effective_samples, dtype=y.dtype)

        for i, (r_orig, c_orig) in enumerate(zip(labeled_pixels_r, labeled_pixels_c)):
            r_padded_center = r_orig + margin
            c_padded_center = c_orig + margin

            patch = zeroPaddedX[r_padded_center - margin : r_padded_center + margin + 1,
                                  c_padded_center - margin : c_padded_center + margin + 1]
            patchesData[i, :, :, :] = patch
            patchesLabels[i] = y[r_orig, c_orig]

        patchesLabels -= 1

    else: 
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


@torch.no_grad()
def evaluate_snn_ecmt(data_loader, model, device, total_timesteps, ecmt_args, test_args, hardware_perf_args=None):
    """Run multi-timestep SNN evaluation and optional hardware metric estimation."""

    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()  

    header = f'测试 ECMT SNN ({CHOSEN_DATASET}):'
    print_freq = 10

    sop_monitor = trans_utils_hardware.SOPMonitor(model, type=1)
    monitor_enabled = getattr(test_args, 'monitor', False)
    sop_monitor.enable() if monitor_enabled else sop_monitor.disable()

    perf_monitor = None
    perf_records_all = []
    if hardware_perf_args is not None and getattr(hardware_perf_args, 'enable', False):
        perf_monitor = hardware_metrics_eval.HardwarePerfMonitor(model)
        perf_monitor.enable()

    model.eval()

    acc_over_timesteps_sum = np.zeros(total_timesteps, dtype=float)
    total_samples_processed = 0
    dataset_total_actual_sops_per_t = np.zeros(total_timesteps, dtype=float)
    dataset_total_potential_ops_per_t = np.zeros(total_timesteps, dtype=float)

    num_batches = len(data_loader)
    start_eval_time = time.time() 

    all_targets_collected = []
    all_preds_collected_per_timestep = [[] for _ in range(total_timesteps)]

    for batch_idx, batch_data in enumerate(data_loader):
        batch_start_time = time.time() 
        images, target = batch_data[0], batch_data[-1]
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        current_batch_size = images.shape[0]

        all_targets_collected.extend(target.cpu().numpy()) 

        if monitor_enabled: sop_monitor.clear_recorded_data()
        if perf_monitor is not None: perf_monitor.clear_recorded_data()
        trans_utils_hardware.reset_net(model)

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

            if perf_monitor is not None:
                perf_records_all.extend(perf_monitor.records)
                perf_monitor.clear_recorded_data()

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

        batch_end_time = time.time() 

        if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == num_batches:
            iters_done_in_cycle = print_freq if (batch_idx + 1) % print_freq == 0 else (batch_idx + 1) % print_freq
            avg_time_per_iter = (batch_end_time - start_eval_time) / (batch_idx + 1)
            eta_seconds = avg_time_per_iter * (num_batches - (batch_idx + 1))
            eta_string = str(time.strftime("%H:%M:%S", time.gmtime(eta_seconds)))
            current_iter_acc1 = acc1_final_batch.item() if 'acc1_final_batch' in locals() and isinstance(acc1_final_batch, torch.Tensor) else 0.0

            print(f"{header} [{batch_idx + 1}/{num_batches}] "
                  f"ETA: {eta_string} "
                  f"Acc@1_current_batch: {current_iter_acc1:.2f} "
                  f"Acc@1_avg: {acc1_meter.avg:.2f} "
                  f"Acc@5_avg: {acc5_meter.avg:.2f} ")
            end_time = time.time()  

    print(f'* Final Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    oa_per_timestep_list = (
                acc_over_timesteps_sum / total_samples_processed).tolist() if total_samples_processed > 0 else [0.0] * total_timesteps

    aa_per_timestep_list = []
    kappa_per_timestep_list = []

    if total_samples_processed > 0:
        print('\n计算每个时间步的 OA、AA 和 Kappa...')
        for t_idx in range(total_timesteps):
            preds_t = np.array(all_preds_collected_per_timestep[t_idx])
            targets_all = np.array(all_targets_collected)

            cm_t = confusion_matrix(targets_all, preds_t, labels=list(range(NUM_CLASSES)))
            list_diag_t = np.diag(cm_t)
            list_raw_sum_t = np.sum(cm_t, axis=1)
            each_acc_t = np.nan_to_num(truediv(list_diag_t, list_raw_sum_t)) * 100
            aa_t = np.mean(each_acc_t)
            aa_per_timestep_list.append(aa_t)

            kappa_t = cohen_kappa_score(targets_all, preds_t) * 100
            kappa_per_timestep_list.append(kappa_t)

            print(f'  T={t_idx + 1}: OA={oa_per_timestep_list[t_idx]:.2f}%, AA={aa_t:.2f}%, Kappa={kappa_t:.2f}%')
    else:
        aa_per_timestep_list = [0.0] * total_timesteps
        kappa_per_timestep_list = [0.0] * total_timesteps

    perf_summary = None
    if perf_monitor is not None:
        perf_summary = hardware_metrics_eval.estimate_hardware_metrics(
            perf_records_all, hardware_perf_args, total_samples_processed
        )

    return acc1_meter.avg, oa_per_timestep_list, aa_per_timestep_list, kappa_per_timestep_list, perf_summary

def main():
    """Load data, convert the ANN, and run the hardware-aware SNN evaluation."""

    apply_hardware_mode(args_hardware_sim_instance, args_test_script_instance, args_hardware_perf_instance)
    set_random_seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True) 
    os.makedirs(HARDWARE_RESULT_DIR, exist_ok=True)
    print(f"测试数据集: {CHOSEN_DATASET}")
    print(f"数据加载自: {DATA_DIR}")
    print(f"预训练 ANN 模型加载自: {PRETRAINED_ANN_PATH}")
    print(f"测试结果将保存至: {RESULT_DIR}")
    print(f"硬件指标结果将保存至: {HARDWARE_RESULT_DIR}")
    print(f"类别数: {NUM_CLASSES}")


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

    if X_full_cube.shape[0] == 0: print("错误: 创建得到的图像块为空。"); return

    print("正在使用当前策略划分数据集：训练部分用于构建校准集，测试部分用于最终评估...")
    # Keep the old per-class split recipe here as a reference.
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
    if num_calib > 0 and num_calib <= len(X_train_p): 
        stratify_calib = y_train_p if np.unique(y_train_p).size > 1 else None
        try:
            if num_calib < len(X_train_p): 
                 cal_idx, _ = train_test_split(np.arange(len(X_train_p)), train_size=num_calib, random_state=SEED, stratify=stratify_calib)
                 X_cal_src, y_cal_src = X_train_p[cal_idx], y_train_p[cal_idx]
            else: 
                 X_cal_src, y_cal_src = X_train_p, y_train_p
        except ValueError: 
            print("警告：校准集分层失败，将尝试不分层切分。")
            if num_calib < len(X_train_p):
                cal_idx, _ = train_test_split(np.arange(len(X_train_p)), train_size=num_calib, random_state=SEED)
                X_cal_src, y_cal_src = X_train_p[cal_idx], y_train_p[cal_idx]
            else:
                X_cal_src, y_cal_src = X_train_p, y_train_p
    elif num_calib > 0:
        X_cal_src, y_cal_src = X_train_p, y_train_p
    else:
        print("警告: 校准集大小为 0 或训练部分为空，跳过 TestNeuron 校准阶段。")
        X_cal_src, y_cal_src = np.array([]), np.array([])

    if len(X_cal_src) > 0:
        X_cal_tensor = torch.FloatTensor(X_cal_src).permute(0, 3, 1, 2)
        y_cal_tensor = torch.LongTensor(y_cal_src)
        calib_loader = DataLoader(TensorDataset(X_cal_tensor, y_cal_tensor), batch_size=CALIBRATION_BATCH_SIZE,
                                  shuffle=False)
    else:
        calib_loader = None  

    X_test_tensor = torch.FloatTensor(X_test_cube).permute(0, 3, 1, 2)
    y_test_tensor = torch.LongTensor(y_test_cube)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=TEST_BATCH_SIZE, shuffle=False)

    ann_model = HSISwinModel(num_classes=NUM_CLASSES, in_chans=PCA_COMPONENTS, img_size=PATCH_SIZE, drop_rate=0.0).to(
        DEVICE)
    try:
        ann_model.load_state_dict(torch.load(PRETRAINED_ANN_PATH, map_location=DEVICE))
        print(f"Successfully loaded ANN weights from {PRETRAINED_ANN_PATH}")
    except Exception as e:
        print(f"Error loading ANN weights: {e}. Exiting.")
        return
    ann_model.eval()
    # Uncomment for a full ANN architecture dump.

    print("ECMT: Replacing MyTestPlace with TestNeuron...")
    snn_model = trans_utils_hardware.replace_test_by_testneuron(ann_model, percent=ECMT_PERCENT)
    snn_model.to(DEVICE)

    if calib_loader:  
        print("ECMT: Calculating thresholds using TestNeurons on calibration data...")
        with torch.no_grad():
            cal_iter_count = 0
            max_cal_iters = 10
            for images_cal, _ in calib_loader:
                _ = snn_model(images_cal.to(DEVICE))
                cal_iter_count +=1
                if cal_iter_count >= max_cal_iters and len(calib_loader) > max_cal_iters : 
                    print(f"ECMT: 已在 {max_cal_iters} 个校准批次上计算阈值，或已处理完全部校准数据。")
                    break
        print("ECMT: 阈值计算完成。")
    else:
        print(
            "ECMT: Skipping threshold calculation due to no calibration data. TestNeurons will use default/uninitialized thresholds.")

    print("ECMT: Converting to SNN structure...")
    snn_model = trans_utils_hardware.replace_at_by_neuron(snn_model, args_hardware_sim_instance)
    snn_model = trans_utils_hardware.replace_testneuron_by_twosideneuron(
        snn_model,
        args_ecmt_instance,
        args_hardware_sim_instance,
    )

    if args_hardware_sim_instance.enable_hardware_simulation:
        if args_hardware_sim_instance.fold_batchnorm:
            print("Hardware: Folding BatchNorm into adjacent linear/conv layers...")
            snn_model = trans_utils_hardware.fold_batchnorm_into_prev_layers(snn_model)

        if (
            args_hardware_sim_instance.replace_norm
            or args_hardware_sim_instance.replace_batchnorm
            or args_hardware_sim_instance.replace_layernorm
            or args_hardware_sim_instance.replace_softmax
            or args_hardware_sim_instance.replace_relu
        ):
            print("Hardware: Replacing nonlinear and normalization modules...")
            snn_model = trans_utils_hardware.replace_nonlinear_by_hardware(snn_model, args_hardware_sim_instance)

        if args_hardware_sim_instance.replace_pool:
            print("Hardware: Replacing pooling modules...")
            snn_model = trans_utils_hardware.replace_pool_by_hardware(snn_model, args_hardware_sim_instance)

        if args_hardware_sim_instance.replace_conv:
            print("Hardware: Replacing patch embedding convolution modules...")
            snn_model = trans_utils_hardware.replace_conv_by_hardware(snn_model, args_hardware_sim_instance)

        if args_hardware_sim_instance.replace_linear or args_hardware_sim_instance.replace_classifier:
            print("Hardware: Replacing linear and classifier projection modules...")
            snn_model = trans_utils_hardware.replace_linear_by_hardware(snn_model, args_hardware_sim_instance)
    else:
        print("Hardware: extra hardware wrappers are disabled; keep the ECMT-converted SNN only.")

    snn_model.to(DEVICE).eval()

    if (
        args_hardware_sim_instance.enable_hardware_simulation
        and (
        args_hardware_sim_instance.replace_layernorm
        and calib_loader is not None
        )
    ):
        print("Hardware: Calibrating LayerNorm fixed statistics on calibration data...")
        snn_model = trans_utils_hardware.calibrate_hardware_norms(
            snn_model, calib_loader, DEVICE, max_batches=10, num_timesteps=SNN_TIMESTEPS
        )
    snn_model.to(DEVICE).eval()
    # Uncomment for a full converted SNN architecture dump.

    print('\n=== Testing ECMT SNN Model on actual test set ===')
    if len(test_loader.dataset) == 0:
        print("Error: Test dataset is empty. Cannot perform SNN evaluation.")
        return

    oa_final_timestep, oa_t_list, aa_t_list, kappa_t_list, perf_summary = evaluate_snn_ecmt(
        test_loader, snn_model, DEVICE, SNN_TIMESTEPS, args_ecmt_instance, args_test_script_instance,
        args_hardware_perf_instance
    )

    print(f"\nECMT SNN 最终测试总体准确率 (OA at T={SNN_TIMESTEPS}): {oa_final_timestep:.2f}%")
    if oa_t_list: print(f"每个时间步的 OA: [{', '.join([f'{a:.2f}%' for a in oa_t_list])}]")
    if aa_t_list: print(f"每个时间步的 AA: [{', '.join([f'{a:.2f}%' for a in aa_t_list])}]")
    if kappa_t_list: print(f"每个时间步的 Kappa: [{', '.join([f'{k:.2f}%' for k in kappa_t_list])}]")

    if perf_summary is not None:
        print(f"Estimated crossbar shape: {perf_summary['crossbar_shape'][0]}x{perf_summary['crossbar_shape'][1]}")
        print(f"Estimated parallel crossbar banks: {perf_summary.get('crossbar_parallel_banks', 1)}")
        print(f"Estimated vector parallel lanes: {perf_summary.get('vector_parallel_lanes', 1)}")
        print(f"Estimated MAC utilization: {perf_summary['mac_utilization'] * 100:.2f}%")
        print(f"Estimated per-sample latency: {perf_summary['latency_per_sample_us']:.6f} us")
        print(f"Estimated throughput: {perf_summary['throughput_sps']:.2f} samples/s")
        print(f"Estimated per-sample energy: {perf_summary['energy_per_sample_uj']:.6f} uJ")
        if not perf_summary.get("estimation_ready", False):
            print("Hardware performance parameters are placeholders; fill the JSON config files under hardware_simulation to obtain non-zero energy/latency estimates.")

    report_filename = os.path.join(HARDWARE_RESULT_DIR, f"ecmt_snn_report_T{SNN_TIMESTEPS}_hardware.txt")
    # Use UTF-8 with BOM so the generated report opens correctly on Windows editors.
    with open(report_filename, 'w', encoding='utf-8-sig') as f:
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

        if perf_summary is not None:
            f.write('\n硬件推理估计结果:\n')
            f.write(f"Crossbar shape: {perf_summary['crossbar_shape'][0]}x{perf_summary['crossbar_shape'][1]}\n")
            f.write(f"Parallel crossbar banks: {perf_summary.get('crossbar_parallel_banks', 1)}\n")
            f.write(f"Vector parallel lanes: {perf_summary.get('vector_parallel_lanes', 1)}\n")
            f.write(f"Actual MACs: {perf_summary['actual_mac']:.0f}\n")
            f.write(f"Potential MACs: {perf_summary['potential_mac']:.0f}\n")
            f.write(f"MAC utilization: {perf_summary['mac_utilization'] * 100:.4f}%\n")
            f.write(f"Total spikes: {perf_summary['total_spikes']:.0f}\n")
            f.write(f"Total crossbar cycles: {perf_summary['total_crossbar_cycles']:.0f}\n")
            f.write(f"Total crossbar latency rounds: {perf_summary.get('total_crossbar_latency_rounds', perf_summary['total_crossbar_cycles']):.0f}\n")
            f.write(f"Estimated per-sample latency (us): {perf_summary['latency_per_sample_us']:.6f}\n")
            f.write(f"Estimated throughput (samples/s): {perf_summary['throughput_sps']:.6f}\n")
            f.write(f"Estimated per-sample energy (uJ): {perf_summary['energy_per_sample_uj']:.6f}\n")
            f.write(f"Estimated total latency over test set (us): {perf_summary['total_latency_us']:.6f}\n")

            f.write('\n按模块类型汇总:\n')
            for kind, vals in perf_summary['per_kind'].items():
                f.write(
                    f"{kind}: energy={vals['energy_pj'] / 1e6:.6f} uJ, "
                    f"latency={vals['latency_ns'] / 1e3:.6f} us\n"
                )

            nonzero_layers = perf_summary.get('per_layer_nonzero_sorted', perf_summary['per_layer_sorted'])
            f.write(f"\n按层能耗排序 (All non-zero layers, {len(nonzero_layers)} entries):\n")
            f.write(f"{'Paper Layer':<56} | {'Kind':<10} | {'Subkind':<12} | {'Energy(uJ)':<14} | {'Latency(us)':<14}\n")
            f.write("-" * 122 + "\n")
            for layer_name, vals in nonzero_layers:
                f.write(
                    f"{vals.get('paper_layer', layer_name):<56} | {vals['kind']:<10} | {vals['subkind']:<12} | "
                    f"{vals['energy_pj'] / 1e6:<14.6f} | {vals['latency_ns'] / 1e3:<14.6f}\n"
                )

            if not perf_summary.get("estimation_ready", False):
                f.write(
                    "\n注：当前 hardware_simulation 目录下的硬件参数 JSON 仍是占位值，"
                    "因此上面的能量与时延估计会是 0 或接近 0。"
                    "填入真实器件参数后即可直接用于论文中的硬件指标估算。\n"
                )
    if perf_summary is not None:
        perf_json_filename = os.path.join(HARDWARE_RESULT_DIR, f"hardware_perf_summary_T{SNN_TIMESTEPS}.json")
        hardware_metrics_eval.save_hardware_perf_summary(perf_summary, perf_json_filename)
        print(f"Hardware performance summary saved to {perf_json_filename}")
    print(f"Detailed report saved to {report_filename}")


if __name__ == '__main__':
    main()
