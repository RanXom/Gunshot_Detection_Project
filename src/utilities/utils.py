import matplotlib.pyplot as plt

def plot_signal(signal, sr):
    plt.plot(signal)
    plt.title(f"Signal with Sampling Rate: {sr}")
    plt.show()
