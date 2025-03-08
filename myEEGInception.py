import torch
import torch.nn as nn
import torch.nn.functional as F


class myEEGInception(nn.Module):
    def __init__(self, input_time=1000, fs=128, ncha=32, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation='elu', n_classes=2):
        super().__init__()

        # ============================= 参数计算 ============================= #
        input_samples = int(input_time * fs / 1000)
        scales_samples = [int(s * fs / 1000) for s in scales_time]

        # ============================= BLOCK 1 ============================== #
        # 并行分支定义
        self.b1_branches = nn.ModuleList()
        for kernel_size in scales_samples:
            # 每个分支的层显式命名
            conv = nn.Conv2d(1, filters_per_branch, (kernel_size, 1), padding='same')
            bn = nn.BatchNorm2d(filters_per_branch)
            act = nn.ELU() if activation == 'elu' else nn.ReLU()
            drop = nn.Dropout2d(dropout_rate)

            # Depthwise卷积
            depthwise = nn.Conv2d(filters_per_branch, filters_per_branch * 2, (1, ncha),
                                  groups=filters_per_branch, bias=False)
            depthwise_bn = nn.BatchNorm2d(filters_per_branch * 2)
            depthwise_act = nn.ELU() if activation == 'elu' else nn.ReLU()
            depthwise_drop = nn.Dropout2d(dropout_rate)

            # 使用ModuleList存储层序列
            self.b1_branches.append(nn.ModuleList([
                conv, bn, act, drop,
                depthwise, depthwise_bn, depthwise_act, depthwise_drop
            ]))

        # Block1池化
        self.b1_pool = nn.AvgPool2d((4, 1))

        # ============================= BLOCK 2 ============================== #
        self.b2_branches = nn.ModuleList()
        for kernel_size in scales_samples:
            conv = nn.Conv2d(len(scales_samples) * filters_per_branch * 2, filters_per_branch,
                             (kernel_size // 4, 1), padding='same', bias=False)
            bn = nn.BatchNorm2d(filters_per_branch)
            act = nn.ELU() if activation == 'elu' else nn.ReLU()
            drop = nn.Dropout2d(dropout_rate)

            self.b2_branches.append(nn.ModuleList([conv, bn, act, drop]))

        self.b2_pool = nn.AvgPool2d((2, 1))

        # ============================= BLOCK 3 ============================== #
        # 第一子模块
        self.b3_conv1 = nn.Conv2d(
            len(scales_samples) * filters_per_branch,
            int(filters_per_branch * len(scales_samples) / 2),
            (8, 1), padding='same', bias=False
        )
        self.b3_bn1 = nn.BatchNorm2d(int(filters_per_branch * len(scales_samples) / 2))
        self.b3_act1 = nn.ELU() if activation == 'elu' else nn.ReLU()
        self.b3_pool1 = nn.AvgPool2d((2, 1))
        self.b3_drop1 = nn.Dropout2d(dropout_rate)

        # 第二子模块
        self.b3_conv2 = nn.Conv2d(
            int(filters_per_branch * len(scales_samples) / 2),
            int(filters_per_branch * len(scales_samples) / 4),
            (4, 1), padding='same', bias=False
        )
        self.b3_bn2 = nn.BatchNorm2d(int(filters_per_branch * len(scales_samples) / 4))
        self.b3_act2 = nn.ELU() if activation == 'elu' else nn.ReLU()
        self.b3_pool2 = nn.AvgPool2d((2, 1))
        self.b3_drop2 = nn.Dropout2d(dropout_rate)

        # ============================= 输出层 =============================== #
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self._get_fc_input_size(ncha, input_samples), n_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _get_fc_input_size(self, ncha, input_samples):
        dummy = torch.randn(1, 1, input_samples, ncha)
        with torch.no_grad():
            features = self.forward_features(dummy)
        return features.view(1, -1).size(1)

    def forward_features(self, x):
        # print(f"Input shape: {x.shape}")
        # ============================= BLOCK 1 ============================== #
        branch_outputs = []
        for branch in self.b1_branches:
            # 显式命名每个操作步骤
            conv, bn, act, drop, depthwise, depthwise_bn, depthwise_act, depthwise_drop = branch

            # 前向传播流程
            out = conv(x)
            out = bn(out)
            out = act(out)
            out = drop(out)

            out = depthwise(out)
            out = depthwise_bn(out)
            out = depthwise_act(out)
            out = depthwise_drop(out)

            branch_outputs.append(out)

        # 拼接与池化
        b1_out = torch.cat(branch_outputs, dim=1)
        b1_out = self.b1_pool(b1_out)

        # ============================= BLOCK 2 ============================== #
        branch_outputs = []
        for branch in self.b2_branches:
            conv, bn, act, drop = branch

            out = conv(b1_out)
            out = bn(out)
            out = act(out)
            out = drop(out)

            branch_outputs.append(out)

        b2_out = torch.cat(branch_outputs, dim=1)
        b2_out = self.b2_pool(b2_out)

        # ============================= BLOCK 3 ============================== #
        # 第一子模块
        b3_u1 = self.b3_conv1(b2_out)
        b3_u1 = self.b3_bn1(b3_u1)
        b3_u1 = self.b3_act1(b3_u1)
        b3_u1 = self.b3_pool1(b3_u1)
        b3_u1 = self.b3_drop1(b3_u1)

        # 第二子模块
        b3_u2 = self.b3_conv2(b3_u1)
        b3_u2 = self.b3_bn2(b3_u2)
        b3_u2 = self.b3_act2(b3_u2)
        b3_u2 = self.b3_pool2(b3_u2)
        b3_out = self.b3_drop2(b3_u2)

        return b3_out

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.forward_features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)


# 测试代码
if __name__ == "__main__":
    model = myEEGInception()
    dummy = torch.randn(2, 1, 128, 8)  # (batch, in_channels, height, width)
    output = model(dummy)
    print(f"输入形状: {dummy.shape}")
    print(f"输出形状: {output.shape}")
    print(f"最后一层权重形状: {model.fc.weight.shape}")