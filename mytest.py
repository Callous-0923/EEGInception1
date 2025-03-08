import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from sklearn.model_selection import train_test_split
from myEEGInception import myEEGInception  # 确保myEEGInception.py在同一目录下


# 参数配置
class Config:
    data_dir = "./PreData"  # pkl文件存放路径
    subjects = ["subject1", "subject2", "subject3", "subject4",
                "subject6", "subject7", "subject8", "subject9"]  # 跳过subject5
    batch_size = 48
    lr = 0.001
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    input_channels = 1  # EEG通道数将被视为高度维度
    time_points = 128  # 下采样后的时间点


# 自定义数据集类
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


# 数据加载函数
def load_data(config):
    all_data = []
    all_labels = []

    for sub in config.subjects:
        with open(os.path.join(config.data_dir, f"{sub}.pkl"), "rb") as f:
            data_dict = pickle.load(f)

            # 处理数据形状 (n_trials, flash_num, channels, time_points) -> (n_samples, 1, time_points, channels)
            data = data_dict["data"].reshape(-1, 32, 128)
            data = np.transpose(data, (0, 2, 1))  # (n_samples, 128, 32)
            data = np.expand_dims(data, 1)  # (n_samples, 1, 128, 32)

            # 处理标签
            labels = data_dict["label"].reshape(-1)

            all_data.append(data)
            all_labels.append(labels)

    # 合并所有subject的数据
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_val, y_train, y_val


# 训练函数
def train(model, train_loader, val_loader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_val_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, config)

        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Acc: {100. * correct / total:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {100. * val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model!")


# 验证函数
def evaluate(model, loader, config):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total


if __name__ == "__main__":
    config = Config()

    # 加载数据
    X_train, X_val, y_train, y_val = load_data(config)

    # 创建数据集和数据加载器
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 初始化模型
    model = myEEGInception(
        input_time=1000,  # 与原始参数保持一致
        fs=128,
        ncha=32,  # 通道数
        filters_per_branch=8,
        scales_time=(500, 250, 125),
        dropout_rate=0.25,
        activation='elu',
        n_classes=2
    ).to(config.device)

    # 打印模型结构
    print(model)

    # 开始训练
    train(model, train_loader, val_loader, config)