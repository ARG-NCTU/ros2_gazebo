import matplotlib.pyplot as plt
import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))

def plot_curve():
    x = np.linspace(-1.5, 1.5, 100)
    T = np.linspace(-1, 1, 9)
    for i in T:
        operator = lambda x: 1 if x >= 0 else -1
        y = np.log(1+np.exp(-10*abs(x/1.25-i)**2))/np.log(2)
        # y = np.tanh(3+3*(operator(i))*(x/1.25-i))
        # y = sigmoid(3+3*(operator(i))*(x/1.25-i))
        plt.plot(x, y, label=f'T={i}')
    plt.legend()        
    plt.show()

if __name__ == '__main__':
    plot_curve()