import numpy as np
import matplotlib.pyplot as plt
from tools import generate_percentages
from constants import SETTINGS_CONV4
from matplotlib.ticker import ScalarFormatter

histories = np.load("data/conv4_rand-False_es-True_data.npz", allow_pickle=True)['histories']

percentages, _ = generate_percentages([1.0, 1.0, 1.0], 0.02,SETTINGS_CONV4['pruning_percentages'])

percentages_list = list()
percentages_list.append(100.0)
for i in range(len(percentages)):
    percentages_list.append(percentages[i][1]*100)

plt.plot(percentages_list, histories)

plt.xticks(np.arange(100, 0, -5))
plt.xlim(100.5, -0.31)
plt.show()