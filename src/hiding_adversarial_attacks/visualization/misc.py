import numpy as np
from matplotlib import pyplot as plt


def softplus(x, beta=1):
    return (1 / beta) * np.log(1 + np.exp(beta * x))


def save_relu_and_softplus_plot(figure_path: str):
    x = np.linspace(-4, 4, 100)

    # the function, which is y = e^x here
    y_relu = np.maximum(0, x)
    y_softplus = np.log(1 + np.exp(x))
    y_softplus_beta_2 = softplus(x, beta=2)
    y_softplus_beta_5 = softplus(x, beta=5)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # plot the function
    plt.plot(x, y_relu, "r", label="ReLU(x)")
    plt.plot(x, y_softplus, "b", label=r"Softplus(x), $\beta=1$")
    plt.plot(x, y_softplus_beta_2, "orange", label=r"Softplus(x), $\beta=2$")
    plt.plot(x, y_softplus_beta_5, "g", label=r"Softplus(x), $\beta=5$")
    plt.legend(loc="upper left")

    # show the plot
    plt.savefig(
        figure_path,
        transparent=True,
    )
    plt.show()


if __name__ == "__main__":
    fig_path = "/home/steffi/dev/master_thesis/images/relu_softplus_plot.png"
    save_relu_and_softplus_plot(fig_path)
