import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

def plot_learning_curve(scores, name="algorithm", 
                        title='Running average of previous 100 scores',
                        running=False):
    clear_output()
    n_games = len(scores)
    x = range(1, n_games + 1)
    if running:
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg, label=name)
    else:
        plt.plot(x, scores, label=name)
        
    plt.title(title)
    plt.legend()
