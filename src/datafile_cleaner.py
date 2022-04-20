import os

if __name__ == "__main__":
    for root_dir, dirs, files in os.walk("../data/lpd_5/lpd_5_cleansed"):
        for file in files:
            if "pos_enc" in file:
                os.remove(os.path.join(root_dir, file))
                # print("Removed:", os.path.join(root_dir, file))w