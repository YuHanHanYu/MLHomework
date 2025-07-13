import matplotlib.pyplot as plt
import numpy as np

def plot_prediction(true_values, pred_values, title='Prediction vs Ground Truth', horizon=90):
    plt.figure(figsize=(14, 6))
    plt.plot(range(horizon), true_values.flatten(), label='Ground Truth', linewidth=2)
    plt.plot(range(horizon), pred_values.flatten(), label='Prediction', linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('Day')
    plt.ylabel('Energy Consumption (Wh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
