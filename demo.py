# 临时测试脚本
from mytest1 import load_subject_data

data, labels, commands = load_subject_data("subject1")
print(f"数据维度检查: {data.shape}")    # 应输出 (24*20*6,1,128,32) = (2880,1,128,32)
print(f"标签维度检查: {labels.shape}")  # 应输出 (2880,)
print(f"指令维度检查: {commands.shape}")# 应输出 (24,)
print(len(commands))