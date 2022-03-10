from tqdm import tqdm
import os
from datetime import datetime
import numpy as np

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def tqdm_wrapper(iter, desc="", total=None):
    return tqdm(iter, desc=desc, ncols=100, position=0, leave=False, total=total)

class Logger(object):
    def __init__(self, log_files_dir):
        self.log_files_dir = log_files_dir
        self.init_date_hour_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_log_dir = os.path.join(self.log_files_dir, "train_log_{}".format(self.init_date_hour_info))

        self._make_dir()

        return 
    
    def _make_dir(self):
        if not os.path.exists(self.log_files_dir):
            os.makedirs(self.log_files_dir)
        return

    def write(self, msg):
        with open(self.current_log_dir, 'a') as f:
            f.write(msg)
            f.close()
        return

    def save_hist(self, hist_loss, hist_acc):
        histLoss_dir = os.path.join(self.log_files_dir, "hist_loss_{}.npz".format(self.init_date_hour_info))
        histAcc_dir = os.path.join(self.log_files_dir, "hist_acc_{}.npz".format(self.init_date_hour_info))
        
        hist_loss = np.array(hist_loss)
        hist_acc = np.array(hist_acc)

        np.savez(histLoss_dir, hist_loss)
        np.savez(histAcc_dir, hist_acc)
        return 
