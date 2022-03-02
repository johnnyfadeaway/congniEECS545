import numpy as np 
import os
import json
from pypianoroll import track

from torch.utils.data import DataLoader, Dataset

import pypianoroll as ppr

class TempoItem(object):
    """
    simple stuct for saving some information about lpd dataset items
    """
    def __init__(self, fpath, song_name, genre):
        """
        init

        Args:
            fpath (str): path to the piano roll npz file
            song_name (str): name of the song
            genre (int): genre of the song
        """
        self.fpath = fpath
        self.song_name = song_name
        self.genre = genre

    def __str__(self):
        genre_map = {0: "Unknown", 1: "Country", 2: "Electronic", 3: "Jazz", 4: "Metal", 5: "Pop", 6: "RnB", 7: "Rock"}
        return "Song {} with genre {}.".format(self.song_name, genre_map[self.genre])
    
    def __dict__(self):
        return {"fpath": self.fpath, "song_name": self.song_name, "genre": self.genre}
    
    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    
class TempoLoader(Dataset):
    """
    class for loading LPD data.
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir (str, ): the directory of the dataset, please specify to e.g. `lpd_5/lpd_5_cleansed`
        """
        self.data_dir = data_dir
        self.genre_labels_dir = os.path.join(data_dir, "{}_genre_label.json".format(os.path.basename(data_dir)))
        # check if genre_label directory exist
        if not os.path.exists(self.genre_labels_dir):
            raise ValueError("genre_labels_dir does not exist\n\tgenre_labels_dir: {}".format(self.genre_labels_dir))
        self.genre_labels = json.load(open(self.genre_labels_dir, "r"))
        self.num_unique_genre = len(set(self.genre_labels.values()))

        # try load listing
        listing_dir = os.path.join(data_dir, "{}_listing.json".format(os.path.basename(data_dir)))
        # check if the listing file was already built
        if os.path.exists(listing_dir):
            self.listing = json.load(open(listing_dir, "r"))

        # create new listing file if not exist
        else:
            # declaring a dict for listing file
            self.listing = dict()
            ## for saving json file for future use
            listing_json = dict()

            # counter for indexing purpose
            counter = 0
            for root_dir, sub_dir, f_names in os.walk(data_dir):

                # find all .npz files
                for f_name in f_names:
                    if ".npz" in f_name:
                        f_path = os.path.join(root_dir, f_name)
                        song_name = f_name.replace(".npz", "")
                        
                        # construct TempomItem object
                        tmp_item = TempoItem(f_path, song_name, self.genre_labels[song_name])

                        self.listing[counter] = tmp_item
                        listing_json[counter] = tmp_item.toJson()
                        counter += 1
                        
            # save the listing file
            json.dump(listing_json, open(listing_dir, "w+"))
        return 
    
    def get_song_genre(self, song_name):
        """
        Args:
            song_name (str, ): the name of the song
        """
        if song_name in self.genre_labels.keys():
            # return the genre of the song, in one-hot vector
            genre_val = self.genre_labels[song_name]
            genre_one_hot = np.zeros(self.num_unique_genre)
            genre_one_hot[genre_val] = 1
            return genre_one_hot
        else:
            raise ValueError("Song {} not found in genre_labels.".format(song_name))
    
    
    def __len__(self):
        return len(self.listing)

    def __getitem__(self, idx):
        """
        Args:
            idx (int, ): the index of the track
        """
        # load the track
        tmp_item = self.listing[idx]

        track_dir = tmp_item.fpath
        multitrack_object = ppr.load(track_dir)

        track_name = tmp_item.song_name
        # make one-hot vector for genre
        genre_val = tmp_item.genre
        genre_one_hot = np.zeros(self.num_unique_genre)
        genre_one_hot[genre_val] = 1

        return multitrack_object, track_name, genre_one_hot


if __name__ == "__main__":
    # testbench for the loader
    data_dir = "../data/lpd_5/lpd_5_cleansed"
    loader = TempoLoader(data_dir)
    print("DEBUG length of loader", len(loader))
    print("DEBUG first track", loader[0])
        

