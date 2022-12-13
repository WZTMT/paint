import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径


def smooth(x, degree):  # degree为平滑等级，0-9，0最低，9最高
    ovr = 0.1 * degree  # 旧值比例
    nvr = 1 - 0.1 * degree  # 新值比例
    x_smooth = []
    for e in x:
        if x_smooth:
            x_smooth.append(ovr * x_smooth[-1] + nvr * e)
        else:
            x_smooth.append(e)
    return x_smooth


def to_pandas_array(x):
    nx = np.array(x)
    data = []
    for i in range(nx.shape[1]):
        for j in range(nx.shape[0]):
            data.append([i+1, nx[j][i]])
    data_df = pd.DataFrame(data, columns=['c1', 'c2'])
    return data_df


def als():
    als_former = np.load('data/AGA/train_als_former.npy', allow_pickle=True)
    als_latter = np.load('data/AGA/train_als_latter.npy', allow_pickle=True)
    sns.set(style='whitegrid')
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("Actor Loss")
    plt.xlabel('train episodes')
    plt.ylabel('actor loss')
    plt.plot(als_former, label='former', color='r')
    plt.plot(als_latter, label='latter', color='b')
    plt.legend()
    plt.savefig(curr_path + "/actor_loss")
    plt.show()


if __name__ == '__main__':
    als()
