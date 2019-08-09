# import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def curve(array, freq, values, color):
    kde = stats.gaussian_kde(array)
    # curve_values = np.zeros_like(v)
    # curve_values[1:] += freq
    # curve_values[:-1] += freq
    # curve_values /= 2
    curve_values = kde(values)
    peak = np.argpartition(freq, len(freq) - 4)[-3:]
    plt.plot(values, curve_values * freq[peak].mean() / curve_values.max(), color=color)
