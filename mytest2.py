import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from myEEGInception import myEEGInception  # 确保myEEGInception.py在相同目录

# %% 参数设置
dataset_dir = "./PreData"  # 预处理数据存放路径
subjects_to_use = [f"subject{i}" for i in range(1, 10) if i != 5]  # 跳过subject5
n_classes = 2
batch_size = 48
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% 自定义数据集类
class ERPDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# %% 加载并预处理数据
# %% 修正后的数据加载函数
def load_data():
    all_data = []
    all_labels = []
    all_commands = []

    for sub in subjects_to_use:
        with open(os.path.join(dataset_dir, f"{sub}.pkl"), "rb") as f:
            eeg_data = pickle.load(f)
            # 原始形状: (n_sessions, 20, 6, 32, 128)
            n_sessions = eeg_data['data'].shape[0]

            # 展平数据
            data = eeg_data['data'].reshape(-1, 32, 128)  # (n_sessions*20*6, 32, 128)
            labels = eeg_data['label'].reshape(-1)  # (n_sessions*20*6,)

            # 同步展开命令数据
            commands = np.repeat(eeg_data['com'], 20 * 6)  # 每个session的com重复20*6次
            commands = commands.reshape(-1)  # (n_sessions*20*6,)

            all_data.append(data)
            all_labels.append(labels)
            all_commands.append(commands)

    # 合并所有数据
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    commands = np.concatenate(all_commands, axis=0)

    # 转换为PyTorch张量
    X = torch.FloatTensor(X[:, None, :, :])  # (样本数, 1, 通道数, 时间点)
    y = torch.LongTensor(y)

    return X, y, commands

# %% 计算指令ACC
def calculate_command_acc(model, X, commands):
    model.eval()
    command_dict = {}

    # 按指令分组
    for i, cmd in enumerate(commands):
        if cmd not in command_dict:
            command_dict[cmd] = []
        command_dict[cmd].append(X[i])

    correct = 0
    total = 0

    for cmd, trials in command_dict.items():
        if len(trials) < 20:  # 跳过不完整的指令
            continue

        # 取20次试次并平均
        trials_tensor = torch.stack(trials[:20]).to(device)
        avg_trial = torch.mean(trials_tensor, dim=0, keepdim=True)

        with torch.no_grad():
            outputs = model(avg_trial)
            pred = torch.argmax(outputs, dim=1).item()

        # 指令标签为1（目标），其他为0
        target_label = 1 if (cmd - 100) == 1 else 0  # 根据原始数据处理逻辑调整
        correct += (pred == target_label)
        total += 1

    return correct / total if total > 0 else 0


# %% 主程序
if __name__ == "__main__":
    # 加载数据
    X, y, commands = load_data()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, commands_train, commands_test = train_test_split(
        X, y, commands, test_size=0.2, random_state=42
    )

    # 创建数据加载器
    train_dataset = ERPDataset(X_train, y_train)
    test_dataset = ERPDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = myEEGInception(
        input_time=1000,
        fs=128,
        ncha=32,  # 根据实际通道数修改
        filters_per_branch=8,
        scales_time=(500, 250, 125),
        dropout_rate=0.25,
        activation='elu',
        n_classes=n_classes
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        trial_acc = accuracy_score(all_labels, all_preds)
        command_acc = calculate_command_acc(model, X_test, commands_test)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Loss: {running_loss / len(train_loader):.4f} | "
              f"Trial Acc: {trial_acc:.4f} | "
              f"Command Acc: {command_acc:.4f}")

    # 最终测试
    final_trial_acc = accuracy_score(all_labels, all_preds)
    final_command_acc = calculate_command_acc(model, X_test, commands_test)
    print(f"\nFinal Results:")
    print(f"Trial Accuracy: {final_trial_acc:.4f}")
    print(f"Command Accuracy: {final_command_acc:.4f}")