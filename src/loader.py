from audioop import mul
import imp
from itertools import count
import numpy as np 
import os
import json
import time
import math

from pypianoroll import track
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
import torch
# from positional_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D

from utils import print_progress_bar, tqdm_wrapper

import pypianoroll as ppr

# == class removed, not working
'''
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
'''

# fix seed for reproducibility
np.random.seed(545)

class TempoSet(Dataset):
    """
    class for loading LPD data.
    """

    def __init__(self, data_dir=""):
        super().__init__()
        self.data_dir = data_dir
        self.genre_labels_dir = ""

        self.genre_labels = dict()
        self.num_unique_genre = 0
        self.listing = dict()

        return

    def load(self, data_dir=None):
        """
        load LPD data, all data saved in class attributes


        Args:
            data_dir (str, ): the directory of the LPD data
        """
        if data_dir is not None:
            self.data_dir = data_dir
        data_dir = self.data_dir

        # load genre labels
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
            print("constructing new listing file")
            # declaring a dict for listing file
            self.listing = dict()
            ## for saving json file for future use ## == removed
            # listing_json = dict() ## == removed

            # counter for indexing purpose
            counter = 0

            # description
            d = "loading {}".format(os.path.basename(data_dir))
            for root_dir, sub_dir, f_names in tqdm_wrapper(os.walk(data_dir), desc=d):

                # find all .npz files
                for f_name in f_names:
                    if ".npz" in f_name:

                        f_path = os.path.join(root_dir, f_name)
                        song_name = f_name.replace(".npz", "")
                        
                        # construct TempomItem object ##== removed
                        # tmp_item = TempoItem(f_path, song_name, self.genre_labels[song_name]) ##== removed
                        
                        # check if drum track exist
                        multitrack = ppr.load(f_path)
                        # skip if drum track does not exist
                        if multitrack.tracks[0].pianoroll.shape[0] == 0:
                            continue
                        self.listing[str(counter)] = f_path, song_name
                        # listing_json[counter] = tmp_item.toJson() ## -- removed 

                        # removed, intolerable loading time
                        '''
                        # generate positional encoding for the song
                        track_len = multitrack.downbeat.shape[0]
                        pos_enc = np.arange(track_len)
                        
                        sin_mask = (pos_enc % 2 == 0).astype(np.int8)
                        cos_mask = (pos_enc % 2 == 1).astype(np.int8)

                        pos_enc_1d = np.sin(1/np.power(10000, 2 * (pos_enc * sin_mask) /track_len))  + \
                                    np.cos(1/np.power(10000, 2 * (pos_enc * cos_mask) /track_len))
                        
                        pos_enc = np.zeros((track_len, 512)) + pos_enc_1d.reshape(-1, 1)

                        # save the positional encoding information
                        pos_enc_name = os.path.join(root_dir, "{}_pos_enc.npz".format(song_name))
                        np.savez_compressed(pos_enc_name, pos_enc)
                        '''
                        
                        counter += 1
                        
            # save the listing file
            print("DEBUG: save listing file, listing_dir: {}".format(listing_dir))
            json.dump(self.listing, open(listing_dir, "w+"))
        return 
    

    
    def get_song_genre(self, song_name):
        """
        Get the genre of one song through its name

        Args:
            song_name (str, ): the name of the song
        
        Returns:
            genre_one_hot (numpy.ndarray, ): the genre of the song, in one-hot vector
        """
        if song_name in self.genre_labels.keys():
            # return the genre of the song, in one-hot vector
            genre_val = self.genre_labels[song_name]
            genre_one_hot = torch.zeros((self.num_unique_genre, ))
            genre_one_hot[genre_val] = 1
            return genre_one_hot
        else:
            raise ValueError("Song {} not found in genre_labels.".format(song_name))
        
        return
    
    def get_multitrack(self, idx):
        """
        get a multitrack of the data

        Args:
            idx (int, ): the index of the data
        
        Returns:
            ppr.Multitrack, the multitrack of the data
        """
        f_path, song_name = self.listing[str(idx)]
        multitrack = ppr.load(f_path)
        return multitrack
    
    def __len__(self):
        """
        return num of data loaded
        """
        return len(self.listing)

    def __getitem__(self, idx):
        """
        Args:
            idx (int, ): the index of the track

        Returns:
            htracks (torch.Tensor, (track_len, 128*4=512)): horizontally stacked non-drum tracks
            gt_dict (dict, ): the ground truth of the data, containing the following keys:
                - "genre": the genre of the song, in one-hot vector
                - "drum_track": the drum track of the data, in torch.Tensor
        """
        # load the track
        track_dir, track_name = self.listing[str(idx)]

        multitrack_object = ppr.load(track_dir)
        # get size of the track
        track_len = multitrack_object.downbeat.shape[0]

        # construct horizontal stacked pianoroll np array
        htracks = torch.zeros((track_len, 128*4))
        ## starting index at 1, since the first track seems always to be the drums trac
        for i in range(1, len(multitrack_object.tracks)):
            tmp_track = torch.tensor(multitrack_object.tracks[i].pianoroll)
            if tmp_track.shape[0] != track_len:
                tmp_track = torch.zeros((track_len, 128))
            htracks[:, (i-1)*128:i*128] = tmp_track
            


        # make one-hot vector for genre
        genre_one_hot = self.get_song_genre(track_name)

        # take the first track from tracks, which is the drum track
        drum_track = multitrack_object.tracks[0]
        if not drum_track.is_drum:
            raise ValueError("Drum track is not drum track")
        # check if drum track is empty
        drum_track = torch.tensor(drum_track.pianoroll)
        
        # construct ground truth dict
        gt_dict = { # "track_name": track_name, ## TODO: do we actually need this?
                    "genre": genre_one_hot,
                    "drum_track": drum_track,
                    "track_len": track_len,
                    }

        htracks = htracks * 20 + 1 # eliminating zeros to prevent deprecating gradient

        return htracks, gt_dict


class ClassifierSet(Dataset):
    """
    torch Dataset class for Classification tasks
    breaks each track into chunks of (512, 128*4) for training
    since most tracks have a resolution of 24, this is 30 seconds
    """
    def __init__(self, in_set, chunk_size=512):
        """
        Args:
            in_set (Dataset, ): the dataset to be broken into chunks
        """
        super().__init__()
        self.loader = in_set

        self.parsed_listing = dict()
        self.chunk_size = chunk_size

        data_dir = self.loader.data_dir

        # check if classification listing directory exist
        classification_listing_dir = os.path.join(data_dir, "{}_classification_listing_{}.json".format(os.path.basename(data_dir), chunk_size))
        if os.path.exists(classification_listing_dir):
            self.parsed_listing = json.load(open(classification_listing_dir, "r"))
            

            """ # no longer needed
            # create place holding variables to prevent repeated loading
            previous_loader_idx = None

            # no longer needed with new approach
            
            # counter for indexing purpose 
            counter = 0
            for i in pickeled_parsed_listing_gt.keys():
                current_loader_idx = pickeled_parsed_listing_gt[i]["loader_idx"]
                chunk_start = int(pickeled_parsed_listing_gt[i]["chunk_start"])
                chunk_end = int(pickeled_parsed_listing_gt[i]["chunk_end"])
                
                # check if htrack is the same as previous htrack
                if (previous_loader_idx != None) and (current_loader_idx != previous_loader_idx):
                    htracks, gt_dict = self.loader[current_loader_idx]
                chuck = htracks[chunk_start:chunk_end, :]
                self.parsed_listing_gt[counter] = chuck, gt_dict
                counter += 1
            """

        else:
            print("constructing classification listing file")
            # counter for indexing purpose
            counter = 0

            # create index storage for listing with different classes
            idx_genre = [[]] * self.loader.num_unique_genre
            num_idx_genre = [0] * self.loader.num_unique_genre
            chunk_info_dict = dict()
            
            self_loader_len = len(self.loader)

            
            with tqdm(total=self_loader_len, desc="Parsing chunks") as pbar:
                for loader_idx in range(self_loader_len):
                    # get information about the tracks
                    htracks, gt_dict = self.loader[loader_idx]
                    track_len = htracks.shape[0]
                    track_genre = torch.argmax(gt_dict["genre"])

                    # break the track into chunks
                    num_chunks = track_len // chunk_size
                    for j in range(num_chunks):
                        
                        chunk_start = j * chunk_size
                        chunk_end = (j+1) * chunk_size
                        # check if chunk is all zeros
                        if torch.sum(htracks[chunk_start:chunk_end, :]) == 0:
                            continue
                        # saving chunk info
                        chunk_info_dict[counter] = loader_idx, chunk_start, chunk_end
                        # recording the index to respecitve genre
                        idx_genre[track_genre].append(counter)
                        num_idx_genre[track_genre] += 1
                        
                        # housekeeping
                        counter += 1
                    pbar.update(1)
                pbar.close()


                print("Loading completed, {} chunks found\n".format(counter))
                print("Saving classification listing file")
                
                print("=============\nNew Loading Summary:")
                print("Number of chunks created: {}".format(counter))
                print("Genre {} has the most data entries {}".format(np.argmax(num_idx_genre), max(num_idx_genre)))
                print("Genre {} has the least data entries {}".format(np.argmin(num_idx_genre), min(num_idx_genre)))
                print("=============\n")
                print("Start shuffling the data and keep all num data entries as the same...")
                
                # save the listing file
                num_data_entries = min(num_idx_genre)
                # convert to np array, shuffle, and crop to the same number of data entries
                for i in range(self.loader.num_unique_genre):
                    idx_genre[i] = np.array(idx_genre[i])
                    np.random.shuffle(idx_genre[i])
                    idx_genre[i] = idx_genre[i][:num_data_entries]
            
            # start constructing the parsed_listing
            ## for indexing
            counter = 0
            total_len = num_data_entries * self.loader.num_unique_genre
            print("Total number of chunks to be created: {}".format(total_len))
            with tqdm(total=num_data_entries*self.loader.num_unique_genre, desc="Saving listing") as pbar:
                for i in range(idx_genre[0].shape[0]):
                    for j in range(len(idx_genre)):
                        self.parsed_listing[str(counter)] = chunk_info_dict[idx_genre[j][i]]
                        counter += 1
                        pbar.update(1)
            pbar.close()
            json.dump(self.parsed_listing, open(classification_listing_dir, "w+"))
            print("Saving completed")
            print("Initialization completed!")
        return 

    def __len__(self):
        """
        return num of data loaded
        """
        return len(self.parsed_listing)
    
    def __getitem__(self, index):
        loader_idx, chunk_start, chunk_end = self.parsed_listing[str(index)]
        htracks, gt_dict = self.loader[loader_idx]
        chunk = htracks[chunk_start:chunk_end, :]
        return chunk, gt_dict["genre"]

class ClassifierTrainTest(Dataset):
    def __init__(self, classifier_set, idx_list):
        super().__init__()
        self.classifier_set = classifier_set
        self.idx_list = idx_list
    
    def __getitem__(self, index):
        htracks, genre_one_hot = self.classifier_set[self.idx_list[index]]
        htracks = torch.unsqueeze(htracks, 0)
        return htracks, genre_one_hot
    
    def __len__(self):
        return len(self.idx_list)

        
class GANdataset(Dataset):
    def __init__(self, classifier_set):
        self.classifier_set = classifier_set
        
    def __len__(self):
        return len(self.classifier_set)

    def __getitem__(self, index):
        """
        Return a 3-channel image-like tensor. 
        0th channel being the htracks of other instruments
        1st channel being the enlarged, repeated, pertrubed genre of the music
        2nd channel being the positional encoding of the htracks

        Args:
            index (int): Index
        Returns:
            cat_htracks (Torch.Tensor): 3-channel image-like tensor
                - 0th channel being the htracks of other instruments
                - 1st channel being the enlarged, repeated, pertrubed genre of the music
                - 2nd channel being the positional encoding of the htracks
            drum_track (Torch.Tensor): 1-channel drum track pianoroll tensor
        """
        # -- htracks as channel 0
        htracks, genre_one_hot = self.classifier_set[index]
        htracks = torch.unsqueeze(htracks, 0)
        loader_idx, chunk_start, chunk_end = self.classifier_set.parsed_listing[str(index)]
        gt_dict = self.classifier_set.loader[loader_idx][1]
        # song_path, song_name = self.classifier_set.loader.listing[str(loader_idx)]

        drum_track, song_len = gt_dict["drum_track"], gt_dict["track_len"]

        # -- create enlarged genre as channle 1
        # generate perturbed genre to mimic classifier last layer
        genre_noise = torch.rand(self.classifier_set.loader.num_unique_genre) * 0.5
        genre_perturbed = torch.abs(genre_one_hot - genre_noise).reshape(-1, 1)
        
        
        ## remove the zeroth genre (it was UNDEFINED)
        genre_perturbed = genre_perturbed[1:]
        h_genre, w_genre = genre_perturbed.shape
        
        # compute the size needed for tiling the genre
        # genre_size = self.classifier_set.loader.num_unique_genre - 1
        _, h, w = htracks.shape
        h_num_tile = math.ceil(h / h_genre)
        w_num_tile = math.ceil(w / w_genre)

        genre_enlarged = torch.tile(genre_perturbed, (h_num_tile, w_num_tile))
        genre_enlarged = genre_enlarged[:h, :w].unsqueeze(0)
        print("DEBUG shape of genre_enlarged: {}".format(genre_enlarged.shape))

        # -- create positional encoding as channel 2
        pos_enc = torch.arange(0, song_len, dtype=torch.float32)
        sin_mask = (pos_enc % 2 == 0).to(torch.int8)
        cos_mask = (pos_enc % 2 == 1).to(torch.int8)
        
        pos_enc = torch.sin(1/torch.float_power(10000, 2 * (pos_enc * sin_mask) /song_len))  + \
                                    torch.cos(1/torch.float_power(10000, 2 * (pos_enc * cos_mask) /song_len))
        
        pos_enc_enlarged = torch.zeros(song_len, 512) + pos_enc.reshape(-1, 1)
        htracks_pos_enc = pos_enc_enlarged[chunk_start:chunk_end, :].unsqueeze(0)

        ''' # correlated to the pre-saved positional encoding method
        pos_enc_fname = "{}_pos_enc.npz".format(song_name)
        song_path_base = os.path.dirname(song_path)
        pos_enc_path = os.path.join(song_path_base, pos_enc_fname)
        song_pos_enc = np.load(pos_enc_path)

        htracks_pos_enc = torch.from_numpy(song_pos_enc[:, chunk_start:chunk_end, :])
        '''

        # -- concatenate the three channels
        cat_htracks = torch.cat((htracks, genre_enlarged, htracks_pos_enc), dim=0)
        return cat_htracks, torch.unsqueeze(drum_track[chunk_start:chunk_end, :], 0)

if __name__ == "__main__":
    # testbench for the loader
    data_dir = "../data/lpd_5/lpd_5_cleansed"
    loader = TempoSet()
    loader.load(data_dir)
    print("DEBUG length of loader", len(loader))
    
    
    # time_start = time.time()
    # for i in range(1000):
    #     htracks, gt_dict = loader[i]
    # time_end = time.time()
    # print("DEBUG time taken for 1 loads", (time_end - time_start)/1000)
    
    # print("DEBUG wtf is this ", loader.get_multitrack(8146))
    # print("DEBUG and wtf is this", loader.get_multitrack(6307))

    # testbench for the classifier loader
    classifier_loader = ClassifierSet(loader)
    print("DEBUG length of classifier loader", len(classifier_loader))
    print("DEBUG first chunk", classifier_loader[11][0].shape, classifier_loader[11][1])

    # testbench for GAN dataset
    gan_dataset = GANdataset(classifier_loader)
    print("DEBUG length of gan_dataset", len(gan_dataset))
    print("DEBUG first chunk", gan_dataset[0][0].shape, gan_dataset[0][1].shape)

    """
    size_hist = np.zeros((len(loader), ))
    for i in range(len(loader)):
        if i % 100 == 0:
            print("DEBUG current index: {}".format(i))
            print("DEBUG current size: {}".format(size_hist[i]))
            print("DEBUG current min max: {} {}".format(np.min(size_hist), np.max(size_hist)))
            print("DEBUG current argmin argmax: {} {}".format(np.argmin(size_hist), np.argmax(size_hist)))

        mlttrack, song_name, genre = loader[i]
        downbeat_size = mlttrack.downbeat.shape[0]
        temp_lens = []
        for j in range(5):
            track_lens, track_pitchs = mlttrack.tracks[j].pianoroll.shape
            temp_lens.append(track_lens)
            

        size_hist[i] = np.max(temp_lens)
        
    print("DEBUG min max of track length", np.min(size_hist), np.max(size_hist))
    print("DEBUG argmin max of track length", np.argmin(size_hist), np.argmax(size_hist))
    print("DEBUG some smallest lengths: ", np.sort(size_hist)[:10])
    """

    '''


    example, song_name, genre = loader[23]
    track_len, track_pitch = example.tracks[0].pianoroll.shape
    print("DEBUG: track_len: {}, track_pitch: {}".format(track_len, track_pitch))
    tracks_except_drum = list()
    for i, t in enumerate(example.tracks):
        if not t.is_drum and t.pianoroll.shape[0] != 0:
            print("DEBUG inspect non drum idxs", i, t.name)
            print("DEBUG inspect track size", t.pianoroll.shape)
            tracks_except_drum.append(t)
    
    print("DEBUG: len(tracks_except_drum): {}".format(len(tracks_except_drum)))
    tracks_except_drum = np.hstack(tuple([t.pianoroll for t in tracks_except_drum]))
    print("DEBUG: tracks_except_drum.shape: {}".format(tracks_except_drum.shape))
    '''
    # print("DEBUG piano_track shape", piano_track)
        
