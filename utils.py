import numpy as np
import matplotlib.pyplot as plt

def plot_stats(stats, window=10):
    fig, axes = plt.subplots(1, len(stats), figsize=(12, 4))
    for ax, (name, values) in zip(axes, stats.items()):
        ax.plot(values, alpha=0.3)
        if len(values) >= window:
            smooth = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(values)), smooth)
        ax.set_title(name)
        ax.set_xlabel("Episode")
    plt.tight_layout()
    plt.show()