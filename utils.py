from typing import Union

import numpy as np
import pandas as pd
from dtaidistance import dtw


def rolling_window(data: Union[np.ndarray, pd.DataFrame], window_size: int, step: int = 1):
    """
    滑动窗口计算函数，可以指定窗口长度和滑动步长
    :param data: 输入原始数据
    :param window_size: 滑动窗口长度
    :param step: 滑动窗口步长，默认为1
    """
    data = np.asarray(data)
    data_window = data[
        np.arange(window_size)[None, :] + np.arange(data.shape[0] - window_size)[:, None]]
    data_window = data_window[np.arange(0, data_window.shape[0], step)]

    return data_window


def align_time_series_dtw(ts1: np.ndarray, ts2: np.ndarray):
    """
    用dtw去除两个时间序列的相位差
    :param ts1:
    :param ts2:
    :return:
    """
    # 计算DTW距离矩阵
    alignment = dtw.warping_path(ts1, ts2)

    # 根据最佳匹配路径重新排列第二个时间序列
    ts2_aligned = np.zeros(len(ts1))
    for idx1, idx2 in alignment:
        ts2_aligned[idx1] = ts2[idx2]

    return ts2_aligned


def plt_chinese():
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
