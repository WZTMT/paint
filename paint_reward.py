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


def reward():
    rewards_aga = np.load('data/AGA/train_rewards.npy')
    ma_rewards_age = np.load('data/AGA/train_ma_rewards.npy')
    rewards_td3 = np.load('data/TD3/train_rewards.npy')
    ma_rewards_td3 = np.load('data/TD3/train_ma_rewards.npy')
    rewards_rtd3 = np.load('data/RTD3/train_rewards.npy')
    ma_rewards_rtd3 = np.load('data/RTD3/train_ma_rewards.npy')
    ma_rewards_age = smooth(rewards_aga, 9.5)
    ma_rewards_td3 = smooth(rewards_td3, 9.5)
    ma_rewards_rtd3 = smooth(rewards_rtd3, 9.5)
    sns.set(style='whitegrid')
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve")
    plt.xlabel('train episodes')
    plt.ylabel('average rewards')
    plt.ylim(-100, 100)
    plt.plot(ma_rewards_td3, label='TD3', color='r')
    plt.plot(ma_rewards_rtd3, label='RTD3', color='green')
    plt.plot(ma_rewards_age, label='AGA', color='b')
    plt.plot(rewards_td3, color='r', alpha=0.2)
    plt.plot(rewards_rtd3, color='green', alpha=0.2)
    plt.plot(rewards_aga, color='b', alpha=0.2)
    plt.legend(labels=['TD3', 'RTD3', 'AGA'])
    plt.savefig(curr_path + "/rewards_curve")
    plt.show()


if __name__ == '__main__':
    reward()
