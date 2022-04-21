
from loader import TempoSet
import numpy as np
import torch

import pypianoroll as ppr
from pypianoroll import track



def to_midi(generated_drumtrack,idx):

    '''
        inputs: generated_drumtrack : torch.uint8 with shape(generated_length * 128)
                idx : song idx
                only_drum : whether the output midi file has only drum
        output: midi file with songname.midi
    '''


    data_dir = "../data/lpd_5/lpd_5_cleansed"
    loader = TempoSet()
    loader.load(data_dir)
    original_multitrack = loader.get_multitrack(idx)

    # obtain paramenters from the original multitrack
    songname = original_multitrack.name
    #print(songname)
    resolution = original_multitrack.resolution
    tempo = original_multitrack.tempo
    downbeat = original_multitrack.downbeat
    tracks = original_multitrack.tracks
    piano_track = tracks[1]
    guitar_track = tracks[2]
    bass_track = tracks[3]
    string_track = tracks[4]
    original_song_length = ((tracks[0]).pianoroll).shape[0]
    drum_name = (tracks[0]).name
    drum_program = (tracks[0]).program
    drum_is_drum = (tracks[0]).is_drum

    # transfrom the generated drumtrack into class standardtrack
    generated_drumtrack = generated_drumtrack.detach().numpy()
    generated_song_length = generated_drumtrack.shape[0]
    num_pad_zeros = original_song_length - generated_song_length
    generated_drumtrack = np.pad(generated_drumtrack,((0,num_pad_zeros),(0,0)),'constant',constant_values=0)
    generated_drumtrack = generated_drumtrack.astype(np.uint8)

    generated_drum_standardtrack = ppr.StandardTrack(name = drum_name,program = drum_program,is_drum = drum_is_drum, pianoroll = generated_drumtrack)

    # put the generated_drum_standardtrack into a new list called new_tracks
    new_tracks = [generated_drum_standardtrack,piano_track,guitar_track,bass_track,string_track]

    # generate the new_multitrack object
    new_multitrack = ppr.Multitrack(name = songname,resolution = resolution, tempo = tempo, downbeat = downbeat, tracks = new_tracks)
    
    # generate the midi file
    path = "midi_files/" + songname + ".mid"
    print(path)
    ppr.write(path,new_multitrack)


def main():
    # this main funciton is used for testing only
    generated_drumtrack = torch.randint(0,1,(7168,128),dtype=torch.uint8)
    #print(generated_drumtrack)

    to_midi(generated_drumtrack,1)


if __name__ == "__main__":
    main()


    

    