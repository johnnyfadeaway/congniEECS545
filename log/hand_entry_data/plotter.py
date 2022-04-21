from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    loss = np.empty((10, ))
    with open("l2_first_result_10epoch.csv", "r") as f:
        for i, line in enumerate(f):
            loss[i] = float(line.split(",")[0].strip())
    
    loss = 100 * loss

    x_range = np.arange(1, 11, 1)
    plt.plot(x_range, loss)
    plt.title("Generator L1 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("l2_first_result_10epoch.png")