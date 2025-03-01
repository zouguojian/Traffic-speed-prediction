import numpy as np
import pandas as pd

def split_and_save_dataset(file_path, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, nodes = 108, add_time_in_day=True):
    """
    将数据集按顺序划分为训练集、验证集和测试集，并保存为 .npy 文件。

    参数:
        file_path (str): 输入的 .npy 文件路径。
        output_dir (str): 输出文件夹路径，用于存储训练集、验证集和测试集。
        train_ratio (float): 训练集的比例（默认为 0.7）。
        val_ratio (float): 验证集的比例（默认为 0.1）。
        test_ratio (float): 测试集的比例（默认为 0.2）。
    """
    # 加载数据
    df = pd.read_csv(file_path, encoding='utf-8')
    data = np.reshape(df.values[:,-1:],newshape=[-1, nodes, 1]) # [N, nodes, 1]
    # 按顺序切分数据
    train_data = data

    # 保存为 .npy 文件
    np.savez(f"train.npz", data=train_data)

    print(f"数据集大小: {train_data.shape[0]}")


# 示例使用
if __name__ == "__main__":
    # file_path = "train.csv"  # 替换为你的 .csv 文件路径
    # split_and_save_dataset(file_path)
    total_training_time = None
    if total_training_time is None:
        data = np.load('adjacent.npz')['data']
        print(data.shape, np.sum(data))