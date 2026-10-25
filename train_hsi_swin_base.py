"""
使用修改后的Swin-Tiny模型进行高光谱图像分类训练
该脚本实现了数据加载、预处理、模型训练和评估的完整流程
"""

import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import pandas as pd

from swin_hsi_swin_tiny_base import HSISwinTiny4Stage

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 数据集配置 ---
# 选择要使用的数据集:
CHOSEN_DATASET = 'chongming'

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

# 从 CHOSEN_DATASET 派生的全局参数
dataset_config = DATASET_SPECIFICS[CHOSEN_DATASET]
num_classes = dataset_config['num_classes']
_DATA_FILE_NAME = dataset_config['data_file']
_GT_FILE_NAME = dataset_config['gt_file']

DATA_ROOT_DIR = 'DataSet'
RESULT_ROOT_DIR = 'result'

CURRENT_DATA_DIR = os.path.join(DATA_ROOT_DIR, CHOSEN_DATASET)
mid = os.path.join(RESULT_ROOT_DIR, 'Base')
CURRENT_RESULT_DIR = os.path.join(mid, CHOSEN_DATASET)

DATA_FILE_PATH = os.path.join(CURRENT_DATA_DIR, _DATA_FILE_NAME)
GT_FILE_PATH = os.path.join(CURRENT_DATA_DIR, _GT_FILE_NAME)


# 模型超参数 (Some might be dataset-specific, adjust as needed)
test_ratio = 0.97  # 测试集比例 (Consider if this should be dataset-specific)
patch_size = 25   # 图像块大小 (Consider if this should be dataset-specific, e.g. PU often uses smaller patches)
pca_components = 30  # PCA降维后的组件数 (Consider if this should be dataset-specific)
num_epochs = 220  # 训练轮数
batch_size = 32   # 批次大小
learning_rate = 1e-4  # 学习率
weight_decay = 0.05   # 权重衰减

early_stop_patience = 20   # 容忍多少个epoch不提升
early_stop_min_delta = 0.01  # 认为“有提升”的最小精度增量（单位：%）

# Adjust patch_size for specific datasets if known to be better
if CHOSEN_DATASET != 'IP':
    pca_components = 15
    num_epochs = 100




def set_random_seed(seed):
    """设置随机种子以确保结果可复现

    Args:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def applyPCA(X, numComponents):
    """对高光谱数据应用PCA降维

    Args:
        X (numpy.ndarray): 原始高光谱数据，形状为 (H, W, C)
        numComponents (int): 降维后的组件数

    Returns:
        tuple: (降维后的数据 (numpy.ndarray), PCA模型对象)
               形状为 (H, W, numComponents)
    """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

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

def padWithZeros(X, margin=2):
    """用零填充图像边缘

    Args:
        X (numpy.ndarray): 原始图像数据
        margin (int): 填充边缘的宽度

    Returns:
        numpy.ndarray: 填充后的图像
    """
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def train_epoch(model, train_loader, criterion, optimizer):
    """训练一个epoch

    Args:
        model (nn.Module): 模型
        train_loader (DataLoader): 训练数据加载器
        criterion (nn.Module): 损失函数
        optimizer (Optimizer): 优化器

    Returns:
        tuple: (平均损失, 准确率)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        try:
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        except Exception as e:
            # 错误处理
            print(f"在批次 {batch_idx} 中发生错误:")
            print(f"输入形状: {inputs.shape}")
            raise e

    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader):
    """在验证集上评估模型性能

    Args:
        model (nn.Module): 模型
        val_loader (DataLoader): 验证数据加载器

    Returns:
        float: 验证准确率
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total

def test(model, test_loader):
    """测试模型性能

    Args:
        model (nn.Module): 模型
        test_loader (DataLoader): 测试数据加载器

    Returns:
        tuple: (准确率, 预测结果, 真实标签)
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return 100. * correct / total, all_preds, all_targets

def evaluate_prediction(y_test, y_pred, dataset_name): # 为清晰起见，将 'dataset' 更改为 'dataset_name'
    """评估预测结果，计算各种指标

    Args:
        y_test (list): 真实标签
        y_pred (list): 预测标签
        dataset_name (str): 数据集名称 ('IP', 'SA', 'PU', etc.)

    Returns:
        tuple: (分类报告, 混淆矩阵, 总体准确率, 每类准确率, 平均准确率, kappa系数)
    """
    if dataset_name in DATASET_SPECIFICS and 'target_names' in DATASET_SPECIFICS[dataset_name]:
        target_names = DATASET_SPECIFICS[dataset_name]['target_names']
    else:
        # Fallback if target names for a dataset are not explicitly listed here
        # but num_classes is known. This generates generic names.
        # However, it's best to have them defined in DATASET_SPECIFICS for meaningful reports.
        print(f"Warning: Specific target names for dataset {dataset_name} not found. Using generic names.")
        num_c = DATASET_SPECIFICS[dataset_name]['num_classes']
        target_names = [f'Class {i+1}' for i in range(num_c)]


    # 修改这里，添加zero_division=0参数
    classification = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    confusion = confusion_matrix(y_test, y_pred)

    # 计算每类准确率
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)

    # 计算总体准确率和kappa系数
    overall_acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, overall_acc * 100, each_acc * 100, average_acc * 100, kappa * 100



def main():
    """主函数，包含完整的训练和评估流程"""
    # 创建结果目录
    os.makedirs(CURRENT_RESULT_DIR, exist_ok=True)
    print(f"使用数据集: {CHOSEN_DATASET}")
    print(f"类别数量: {num_classes}")
    print(f"数据将从以下路径加载: {DATA_FILE_PATH}")
    print(f"结果将保存到以下路径: {CURRENT_RESULT_DIR}")


    # 加载数据
    print('正在加载数据...')
    if not os.path.exists(DATA_FILE_PATH):
        print(f"错误: 在 {DATA_FILE_PATH} 未找到数据文件")
        return
    if not os.path.exists(GT_FILE_PATH):
        print(f"错误: 在 {GT_FILE_PATH} 未找到真实标签文件")
        return

    X_raw = sio.loadmat(DATA_FILE_PATH)
    y_raw = sio.loadmat(GT_FILE_PATH)

    # 查找数据和真实标签的正确键名
    # 常用数据键名: 'indian_pines_corrected', 'salinas_corrected', 'paviaU' 等
    # 常用真实标签键名: 'indian_pines_gt', 'salinas_gt', 'paviaU_gt' 等
    data_key = None
    for key in X_raw.keys():
        if not key.startswith('__'): # 普通的HSI数据数组键名不以'__'开头
            data_key = key
            break
    gt_key = None
    for key in y_raw.keys():
        if not key.startswith('__'):
            gt_key = key
            break

    if data_key is None :
        print(f"无法在 {DATA_FILE_PATH} 中找到HSI数据数组。文件中的键: {X_raw.keys()}")
        return
    if gt_key is None:
        print(f"无法在 {GT_FILE_PATH} 中找到真实标签数据数组。文件中的键: {y_raw.keys()}")
        return

    X = X_raw[data_key]
    y = y_raw[gt_key]

    print(f"数据数组键名: '{data_key}', 形状: {X.shape}")
    print(f"真实标签数组键名: '{gt_key}', 形状: {y.shape}")


    # 数据预处理
    print('正在应用PCA...')
    X_pca, pca_model = applyPCA(X, numComponents=pca_components) # 存储pca_model

    print('正在创建数据立方体...')
    X_cube, y_cube = createImageCubes(X_pca, y, windowSize=patch_size)

    if X_cube.shape[0] == 0:
        print("错误: 未创建数据立方体。请检查真实标签图中的已标记像素或 createImageCubes 函数的逻辑。")
        return

    # 首先分割出测试集
    X_remaining, X_test, y_remaining, y_test_labels = train_test_split(
        X_cube, y_cube, test_size=test_ratio, random_state=345, stratify=y_cube
    )

    # 再将剩余数据分为训练集和验证集
    # 确保 y_remaining 非空且有足够样本进行分层抽样
    val_split_test_size = 0.2
    if len(np.unique(y_remaining)) < 2 : # 类别少于2个或某个类别的样本少于2个时无法进行分层抽样
        stratify_val = None
        print("警告: y_remaining 中的类别多样性不足以进行分层验证集划分。将使用非分层划分。")
    else: # 检查 y_remaining 中的任何类别是否样本过少而无法进行划分 (sklearn 通常要求至少2个)
        unique_classes, counts = np.unique(y_remaining, return_counts=True)
        if np.any(counts < 2):
                               # 关键在于如果一个类别只有一个样本，则无法划分。
            stratify_val = None
            print("警告: y_remaining 中的某些类别样本过少，无法进行分层验证集划分。将使用非分层划分。")
        else:
            stratify_val = y_remaining

    if len(y_remaining) * val_split_test_size < 1 or len(y_remaining) * (1-val_split_test_size) < 1 :
         print("警告: y_remaining 中没有足够的样本来创建验证集。跳过验证集创建。")
         X_train, X_val, y_train_labels, y_val_labels = X_remaining, np.array([]), y_remaining, np.array([])
    elif len(y_remaining) == 0:
        print("错误: 测试集划分后 X_remaining 为空。")
        return
    else:
        X_train, X_val, y_train_labels, y_val_labels = train_test_split(
            X_remaining, y_remaining, test_size=val_split_test_size, random_state=345, stratify=stratify_val
        )


    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)  # [N, C, H, W]
    if X_val.shape[0] > 0:
        X_val_tensor = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    else:
        X_val_tensor = torch.FloatTensor([]) # 空张量

    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_train_tensor = torch.LongTensor(y_train_labels)
    if y_val_labels.shape[0] > 0:
        y_val_tensor = torch.LongTensor(y_val_labels)
    else:
        y_val_tensor = torch.LongTensor([])


    y_test_tensor = torch.LongTensor(y_test_labels)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    if X_val.shape[0] > 0:
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_dataset = None
        val_loader = None
        print("由于验证集为空，跳过创建验证数据加载器。")

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建swin-tiny基线模型
    model = HSISwinTiny4Stage(
        num_classes=num_classes,
        in_chans=pca_components,
        img_size=patch_size,
        pad_size=32,
        pretrained=False,
        drop_rate=0.1,
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑

    # 为不同部分的参数设置不同的学习率
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': learning_rate * 0.1},  # 主干网络使用较小的学习率
        {'params': model.patch_extract.parameters(), 'lr': learning_rate},
    ], weight_decay=weight_decay)


    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 第一次重启的周期
        T_mult=2,  # 每次重启后周期长度的倍数
        eta_min=1e-6  # 最小学习率
    )

    # 训练模型
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    model_save_path = os.path.join(CURRENT_RESULT_DIR, 'best_model.pth')

    early_stop_counter = 0


    print('Training started...')
    for epoch in range(num_epochs):

        # -------- Train one epoch --------
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer
        )

        # -------- Validation --------
        if val_loader:
            val_acc = validate(model, val_loader)

            # Early stopping & best model saving
            if val_acc > best_val_acc + early_stop_min_delta:
                best_val_acc = val_acc
                early_stop_counter = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                early_stop_counter += 1
        else:
            val_acc = 0.0  # 保证 val_acc 每轮都有值

        # -------- Record history (CRITICAL: every epoch, before break) --------
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # -------- Logging --------
        if val_loader:
            print(
                f'Epoch {epoch + 1}/{num_epochs}: '
                f'Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.2f}%, '
                f'Val Acc: {val_acc:.2f}%'
            )
        else:
            print(
                f'Epoch {epoch + 1}/{num_epochs}: '
                f'Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.2f}%'
            )

        # -------- Early stopping trigger --------
        if val_loader and early_stop_counter >= early_stop_patience:
            print(
                f'\nEarly stopping triggered at epoch {epoch + 1}. '
                f'Best Val Acc: {best_val_acc:.2f}%'
            )
            break

        # -------- LR scheduler step --------
        scheduler.step()

    # =======================
    # Training completed
    # =======================

    if val_loader:
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    else:
        print("Training completed (no validation set used for best model selection).")
        # 若无验证集，保存最后模型
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at end of training: {model_save_path}")

    # =======================
    # Save training history (SAFE)
    # =======================

    # 防御式检查
    assert len(history['train_loss']) == len(history['train_acc']) == len(history['val_acc']), \
        f"History length mismatch: " \
        f"loss={len(history['train_loss'])}, " \
        f"acc={len(history['train_acc'])}, " \
        f"val={len(history['val_acc'])}"

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(
        os.path.join(CURRENT_RESULT_DIR, 'training_history.csv'),
        index=False
    )
    print("Training history saved.")

    # 测试阶段开始
    print("\nTesting phase started...")

    # 加载最佳模型
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded best model from {model_save_path} for testing.")
    else:
        print(f"Warning: Model file {model_save_path} not found. Testing with the current model state (likely last epoch's).")


    # 在测试集上评估最佳模型性能
    test_acc, preds, targets = test(model, test_loader)
    print(f'Test Accuracy: {test_acc:.2f}%')

    # 计算并保存最终评估指标
    classification, confusion, oa, each_acc, aa, kappa = evaluate_prediction(targets, preds, CHOSEN_DATASET)

    print(f'Overall Accuracy: {oa:.2f}%')
    print(f'Average Accuracy: {aa:.2f}%')
    print(f'Kappa: {kappa:.2f}%')

    report_path = os.path.join(CURRENT_RESULT_DIR, 'final_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'Dataset: {CHOSEN_DATASET}\n')
        f.write(f'Test Accuracy: {test_acc:.2f}%\n')
        f.write(f'Overall Accuracy: {oa:.2f}%\n')
        f.write(f'Average Accuracy: {aa:.2f}%\n')
        f.write(f'Kappa: {kappa:.2f}%\n\n')
        f.write('Classification Report:\n')
        f.write(classification)
        f.write(f'\nConfusion Matrix:\n{confusion}\n')
    print(f"Classification report saved to {report_path}")

    print("Testing phase completed!")

if __name__ == '__main__':
    set_random_seed(42)  # 设置随机种子
    main()