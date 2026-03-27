import matplotlib.pyplot as plt

def trace_plot(samples=None):
    ndim = 5
    fig, ax = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    labels =   ['eps', 'L40', 'fesc10','h', 'fnoise']

    for i in range(ndim):
        ax[i].plot(samples[:,  :, i], 'k', alpha=0.3)
        ax[i].set_ylabel(labels[i])

    ax[-1].set_xlabel("step")
    plt.show()
