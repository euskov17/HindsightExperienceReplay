import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

def plot_learning_curve(scores, name="algorithm"):
    clear_output()
    n_games = len(scores)
    x = range(1, n_games + 1)
    plt.plot(x, scores, label=name)
    plt.title('Running average of previous 100 scores')
    plt.legend()
