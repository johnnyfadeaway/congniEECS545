import numpy as np 
import os
import json

from torch.utils.data import DataLoader, Dataset

import pypianoroll as ppr

class tempo_loader(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str, ): the directory of the dataset, please specify to e.g. `lpd_5/lpd_5_cleansed`
        """
        self.data_dir = data_dir

        listing_dir = os.path.join(data_dir, "{}_listing.json".format(os.path.basename(data_dir)))
        # check if the listing file was already built
        if os.path.exists(listing_dir):
            self.listing = json.load(open(listing_dir, "r"))

        # create new listing file 
        else:
            # declaring the structured array as listing file
            ## maxium length of the directories is set to be 150 char's
            self.listing = dict()
            counter = 0
            for root_dir, sub_dir, f_names in os.walk(data_dir):

                # find all .npz files
                for f_name in f_names:
                    if ".npz" in f_name:
                        f_path = os.path.join(root_dir, f_name)
                        self.listing[counter] = f_path, f_name.replace(".npz", "")
                        counter += 1
                        if counter % 100 == 0:
                            print("DEBUG num of songs found", counter)
                        
            # save the listing file
            json.dump(self.listing, open(listing_dir, "w+"))
    
    def __len__(self):
        return len(self.listing)

    def __getitem__(self, idx):
        """
        Args:
            idx (int, ): the index of the track
        """
        # load the track
        track_dir, track_name = self.listing[idx]
        multitrack_object = ppr.load(track_dir)

        return multitrack_object, track_name


if __name__ == "__main__":
    # testbench for the loader
    data_dir = "../data/lpd_5/lpd_5_cleansed"
    loader = tempo_loader(data_dir)
    print("DEBUG length of loader", len(loader))
    print("DEBUG first track", loader[0])
        

