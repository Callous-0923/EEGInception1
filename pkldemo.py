import os
import pickle
import numpy as np


class Config:
    data_dir = "./PreData"
    subjects = ["subject1", "subject2", "subject3", "subject4",
                "subject6", "subject7", "subject8", "subject9"]


def load_subject_data(subject):
    """加载并检查被试数据"""
    with open(os.path.join(Config.data_dir, f"{subject}.pkl"), "rb") as f:
        data_dict = pickle.load(f)
        data = data_dict["data"]
        labels = data_dict["label"]
        commands = data_dict["com"]
        return data, labels, commands


def check_data_alignment(subject):
    """检查数据和标签是否对齐"""
    data, labels, commands = load_subject_data(subject)

    # 检查数据和标签的形状
    print(f"Subject: {subject}")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Command shape: {commands.shape}")

    # 检查数据和标签的样本数是否一致
    if data.shape[0] != labels.shape[0]:
        print(f"Warning: Data and labels do not have the same number of samples!")
        print(f"Data samples: {data.shape[0]}, Labels samples: {labels.shape[0]}")
    else:
        print("Data and labels have the same number of samples.")

    # 检查每个 trial 的数据和标签是否对齐
    for trial_idx in range(data.shape[0]):
        trial_data = data[trial_idx]
        trial_labels = labels[trial_idx]

        # print(f"Trial {trial_idx + 1}:")
        # print(f"  Data shape: {trial_data.shape}")
        # print(f"  Labels shape: {trial_labels.shape}")

        # 检查每个 stimulus 的数据和标签是否对齐
        for stimulus_idx in range(trial_data.shape[0]):
            stimulus_data = trial_data[stimulus_idx]
            stimulus_label = trial_labels[stimulus_idx]
            # print(f"  Stimulus {stimulus_idx + 1}:")
            # print(f"    Data shape: {stimulus_data.shape}")
            # print(f"    Label: {stimulus_label}")
            #

    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    for subject in Config.subjects:
        check_data_alignment(subject)