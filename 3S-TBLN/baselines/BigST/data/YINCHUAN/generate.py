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
    num_samples, num_nodes, _ = data.shape
    data_list = [data]

    if add_time_in_day:
        df['datatime'] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h') + pd.to_timedelta(
            df.minute, unit='m')
        time_ind = (df['datatime'].values - df['datatime'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = time_ind
        time_in_day = time_in_day.reshape(-1, nodes, 1)
        data_list.append(time_in_day)

    data = np.concatenate(data_list, axis=-1)
    total_samples = data.shape[0]

    # 计算各个集的大小
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    # 按顺序切分数据
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # 保存为 .npy 文件
    np.save(f"train.npy", train_data)
    np.save(f"val.npy", val_data)
    np.save(f"test.npy", test_data)

    print(f"数据已成功划分为:")
    print(f"训练集大小: {train_data.shape[0]}")
    print(f"验证集大小: {val_data.shape[0]}")
    print(f"测试集大小: {test_data.shape[0]}")


# 示例使用
if __name__ == "__main__":
    file_path = "train.csv"  # 替换为你的 .csv 文件路径

    split_and_save_dataset(file_path)