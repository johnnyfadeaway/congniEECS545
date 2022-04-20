from os import path
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    hist_loss = []
    hist_acc = []

    log_dir = "pianoroll_classifier_cnn/train_log_2022-03-09_23-05-36.log/train_log_2022-03-09_23-05-36"

    with open(log_dir, 'r') as f:
        for line in f:
            if "Epoch Loss" in line:
                
                loss_line, acc_line = line.split(", ")
                loss = float(loss_line.split(": ")[1])
                acc = float(acc_line.split(": ")[1])
                hist_loss.append(loss)
                hist_acc.append(acc)

    
    hist_loss = np.array(hist_loss)
    hist_acc = np.array(hist_acc)

    print("DEBUG hist_loss length", len(hist_loss))

    x_range = np.arange(1, len(hist_loss) + 1)

    fig, ax = plt.subplots()
    ax.plot(x_range, hist_loss, label="Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CNN classifier Loss vs Epoch")
    ax.legend()
    ax.grid(True)

    log_name = path.basename(log_dir)

    plt.savefig("loss_vs_epoch_{}.png".format(log_name))


