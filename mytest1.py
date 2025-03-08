import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from sklearn.model_selection import KFold, train_test_split
from collections import defaultdict
from myEEGInception import myEEGInception


# 实验参数配置
class Config:


    # 新增调试参数
    verbose = True  # 是否打印训练细节
    balance_weights = True  # 是否启用类别平衡
    debug_mode = True

    data_dir = "./PreData"
    subjects = ["subject1", "subject2", "subject3", "subject4",
                "subject6", "subject7", "subject8", "subject9"]

    # 数据维度参数
    n_trials = 24  # 试验数
    n_repeats = 20  # 重复次数
    n_stimuli = 6  # 刺激数
    n_channels = 32  # 通道数
    n_timepoints = 128  # 时间点

    # 训练参数
    batch_size = 64
    lr = 0.001
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2


class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.LongTensor([self.labels[idx]])
        return x, y.squeeze()


def load_subject_data(subject):
    """加载并重组被试数据，保留试验结构"""
    with open(os.path.join(Config.data_dir, f"{subject}.pkl"), "rb") as f:
        data_dict = pickle.load(f)

        # 原始数据维度验证
        data = data_dict["data"]  # (24,20,6,32,128)
        labels = data_dict["label"]  # (24,20,6)
        commands = data_dict["com"]  # (24,)
        # print(f"数据维度检查: {data.shape}")  # 应输出 (24*20*6,1,128,32) = (2880,1,128,32)
        # print(f"标签维度检查: {labels.shape}")  # 应输出 (2880,)
        # print(f"指令维度检查: {commands.shape}")  # 应输出 (24,)

        # 重组数据为(trial*repeat*stimuli, 1, timepoints, channels)
        data = np.transpose(data, (0, 1, 2, 4, 3))  # (24,20,6,128,32)
        data = data.reshape(-1, Config.n_timepoints, Config.n_channels)
        data = np.expand_dims(data, 1).astype(np.float32)  # (24*20*6,1,128,32)

        # 标签处理
        labels = labels.reshape(-1).astype(np.int64)

        # 数据完整性检查
        assert len(data) == Config.n_trials * Config.n_repeats * Config.n_stimuli
        assert labels.min() >= 0 and labels.max() <= 1

        return data, labels, commands


def evaluate_command_acc(preds, labels, commands):
    """优化版指令级准确率计算（适配6刺激20重复场景）

    参数说明：
    - preds: 模型预测的概率矩阵，形状为(N, 2)，N=试验数*20*6=24*20*6=2880
    - labels: 真实标签，形状为(N,)，其中仅20*24=480个为1（每个试验20个目标）
    - commands: 指令编码数组，形状为(24,)，每个元素为target+100
    """
    # ================= 输入验证 =================
    assert len(preds) == len(labels), "预测结果与标签数量不匹配"
    print(len(commands))
    assert len(commands) == Config.n_trials, f"指令数{len(commands)}≠试验数{Config.n_trials}"
    expected_samples = Config.n_trials * Config.n_repeats * Config.n_stimuli
    assert len(preds) == expected_samples, f"数据量异常，应有{expected_samples}试次，实际{len(preds)}"

    cmd_acc = 0
    total_commands = 0

    # ================= 遍历每个试验 =================
    for trial_idx in range(Config.n_trials):
        # 当前试验的真实目标刺激（从commands解码）
        try:
            true_stim = (commands[trial_idx] % 100) - 1  # 指令编码为target+100
            assert 0 <= true_stim < Config.n_stimuli, f"无效目标刺激{true_stim}"
        except Exception as e:
            print(f"[错误] 试验{trial_idx}指令解码失败: {e}")
            continue

        # ================= 提取当前试验的所有试次 =================
        trial_data = []
        for repeat in range(Config.n_repeats):
            # 计算当前重复的试次索引范围
            start = trial_idx * Config.n_repeats * Config.n_stimuli + repeat * Config.n_stimuli
            end = start + Config.n_stimuli
            trial_slice = slice(start, end)

            # 验证索引范围
            if end > len(preds):
                print(f"[警告] 试验{trial_idx}重复{repeat}数据不完整")
                continue

            # 提取当前重复的6个刺激数据
            repeat_preds = preds[trial_slice]  # 形状(6,2)
            repeat_labels = labels[trial_slice]  # 形状(6,)

            # 应当包含恰好1个目标试次
            target_idx = np.where(repeat_labels == 1)[0]
            if len(target_idx) != 1:
                print(f"[异常] 试验{trial_idx}重复{repeat}包含{len(target_idx)}个目标")
                continue

            # 按刺激顺序存储预测概率
            for stim_idx in range(Config.n_stimuli):
                # 当前刺激是否是本重复的目标
                is_target = (stim_idx == target_idx[0])
                # 记录目标类概率和刺激索引
                trial_data.append((
                    stim_idx,
                    repeat_preds[stim_idx, 1],  # 目标类概率
                    is_target
                ))

        # ================= 计算各刺激的平均目标概率 =================
        stim_probs = defaultdict(list)
        for stim_idx, prob, _ in trial_data:
            stim_probs[stim_idx].append(prob)

        # 计算每个刺激的平均概率（至少需要5次有效测量）
        valid_stim_probs = {}
        for stim_idx, probs in stim_probs.items():
            if len(probs) >= Config.n_repeats // 4:  # 至少5次重复
                valid_stim_probs[stim_idx] = np.mean(probs)
            else:
                print(f"[警告] 试验{trial_idx}刺激{stim_idx}数据不足({len(probs)}次)")

        if not valid_stim_probs:
            print(f"[错误] 试验{trial_idx}无有效刺激数据")
            continue

        # ================= 确定预测刺激 =================
        predicted_stim = max(valid_stim_probs, key=valid_stim_probs.get)

        # ================= 统计结果 =================
        total_commands += 1
        if predicted_stim == true_stim:
            cmd_acc += 1
        else:
            if Config.debug_mode:
                print(f"[调试] 试验{trial_idx}预测失败 | 真实:{true_stim} 预测:{predicted_stim}")
                print("刺激概率分布:",
                      {k: f"{v:.3f}" for k, v in valid_stim_probs.items()})

    # ================= 最终校验 =================
    if total_commands == 0:
        print("[严重错误] 无有效指令可评估")
        return 0.0

    final_acc = cmd_acc / total_commands
    print(f"指令评估完成 | 有效指令数:{total_commands} 正确数:{cmd_acc} 准确率:{final_acc:.2%}")
    return final_acc

def within_subject_validation(subject):
    """被试内五折交叉验证（带类别平衡）"""
    data, labels, commands = load_subject_data(subject)
    kf = KFold(n_splits=5, shuffle=True)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        # 划分数据集
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # ==== 添加类别权重计算逻辑 ====
        if Config.balance_weights:
            # 计算类别权重
            class_counts = np.bincount(y_train)
            class_weights = 1. / (class_counts + 1e-6)  # 防止除零
            class_weights = class_weights / class_weights.sum() * Config.num_classes
            weights_tensor = torch.FloatTensor(class_weights).to(Config.device)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)  # 加权损失
        else:
            criterion = nn.CrossEntropyLoss()  # 普通损失
        # ============================

        # 获取测试集对应的指令 有问题
        # test_commands = [commands[i // (Config.n_repeats * Config.n_stimuli)] for i in test_idx]

        # 步骤1: 提取测试集涉及的所有唯一试验索引
        test_trial_indices = np.unique([i // (Config.n_repeats * Config.n_stimuli) for i in test_idx])

        # 步骤2: 根据试验索引获取对应指令
        test_commands = [commands[trial_idx] for trial_idx in test_trial_indices]

        print(f"测试样本数: {len(test_idx)}")
        print(f"涉及试验数: {len(test_trial_indices)}")
        print(f"生成指令数: {len(test_commands)}")




        # 初始化模型
        model = myEEGInception(
            input_time=1000,
            fs=128,
            ncha=Config.n_channels,
            filters_per_branch=8,
            scales_time=(500, 250, 125),
            dropout_rate=0.25,
            activation='elu',
            n_classes=2
        ).to(Config.device)

        # 数据加载器
        train_loader = DataLoader(EEGDataset(X_train, y_train),
                                  batch_size=Config.batch_size, shuffle=True)
        test_loader = DataLoader(EEGDataset(X_test, y_test),
                                 batch_size=Config.batch_size * 2)

        # 训练循环
        optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)  # 使用加权损失

        for epoch in range(Config.epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(Config.device)
                targets = targets.to(Config.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # 计算训练指标
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # 打印训练进度
            train_acc = 100. * correct / total
            # print(
            #     f"Fold {fold + 1} Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f} | Acc: {train_acc:.2f}%")

        # 评估
        trial_acc, cmd_acc = evaluate_model(model, test_loader, test_commands)
        results.append((trial_acc, cmd_acc))
        print(f"[{subject}] Fold {fold + 1} | Trial: {trial_acc:.2f}% | Command: {cmd_acc:.2f}%")

    # 计算平均结果
    avg_trial = np.mean([r[0] for r in results])
    avg_cmd = np.mean([r[1] for r in results])
    print(f"[{subject}] 平均结果 | 试次: {avg_trial:.2f}% | 指令: {avg_cmd:.2f}%")
    return avg_trial, avg_cmd


def cross_subject_validation(test_subject):
    """跨被试验证（留一被试法）"""
    # 加载训练数据
    train_data, train_labels = [], []
    for sub in Config.subjects:
        if sub == test_subject:
            continue
        data, labels, _ = load_subject_data(sub)
        train_data.append(data)
        train_labels.append(labels)

    X_train = np.concatenate(train_data)
    y_train = np.concatenate(train_labels)

    # 加载测试数据
    X_test, y_test, commands = load_subject_data(test_subject)

    # 初始化模型
    model = myEEGInception(
        input_time=1000,
        fs=128,
        ncha=Config.n_channels,
        filters_per_branch=8,
        scales_time=(500, 250, 125),
        dropout_rate=0.25,
        activation='elu',
        n_classes=2
    ).to(Config.device)

    # 训练配置
    train_loader = DataLoader(EEGDataset(X_train, y_train),
                              batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(EEGDataset(X_test, y_test),
                             batch_size=Config.batch_size * 2)

    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练过程
    for epoch in range(Config.epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(Config.device)
            targets = targets.to(Config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # 最终评估
    trial_acc, cmd_acc = evaluate_model(model, test_loader, commands)
    print(f"[跨被试] {test_subject} | 试次: {trial_acc:.2f}% | 指令: {cmd_acc:.2f}%")
    return trial_acc, cmd_acc


def evaluate_model(model, loader, commands):
    """统一评估函数"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(Config.device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(targets.numpy())

    # 试次级准确率
    trial_preds = np.argmax(all_preds, axis=1)
    trial_acc = np.mean(trial_preds == np.array(all_labels)) * 100

    # print(len(np.array(all_preds)))
    # print(len(np.array(all_labels)))
    # np.array(len(commands))

    # 指令级准确率
    cmd_acc = evaluate_command_acc(np.array(all_preds), np.array(all_labels), commands) * 100

    return trial_acc, cmd_acc


if __name__ == "__main__":
    # 被试内验证
    # print("=" * 50 + "\n被试内验证结果")
    # for sub in Config.subjects:
    #     within_subject_validation(sub)

    # 跨被试验证
    print("\n" + "=" * 50 + "\n跨被试验证结果")
    for sub in Config.subjects:
        cross_subject_validation(sub)