import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

def plot_learning_curve(n_games, scores, running=False):
    clear_output()
    x = range(1, n_games + 1)
    if running:
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
    else:
        plt.plot(x, scores)
    plt.title('Running average of previous 100 scores')
    plt.show()