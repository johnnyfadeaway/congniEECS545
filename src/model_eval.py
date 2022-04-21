from re import TEMPLATE
from generator import generator as ge
from loader import TempoSet
import torch 
import numpy as np
import pypianoroll as ppr
from pypianoroll import track

from test import to_midi

def main():
    loader = TempoSet()
    data_dir = "../data/lpd_5/lpd_5_cleansed"
    loader.load(data_dir)
    index = 515
    chucks,drum_chucks = loader.get_chunks_from_song(index)
    num_of_chucks = len(chucks)
    #print("DEBUG num_of_chucks: ",num_of_chucks)
    #print("DEBUG shape of chuck: ",chucks[0].shape)

    output_drumbeats = torch.zeros(512,128)
    generator = ge()

    PATH = "../model/generator_l2_20220421_014944.pth"

    generator.load_weights(PATH)
    generator.eval()

    for i in range(num_of_chucks):
        temp_chuck = (chucks[i].view(-1,3,512,512))
        temp_chuck = temp_chuck.type(torch.float32)
        #print("DEBUG temp_chuck size: ",temp_chuck.dtype)
        with torch.no_grad():
            generated_drumbeats = generator(temp_chuck)
        #print("DEBUG size of generator output: ", generated_drumbeats.shape)

        generated_drumbeats = generated_drumbeats.view(512,128)
        #print("DEBUG size of generator output: ", generated_drumbeats.shape)

        output_drumbeats = torch.concat((output_drumbeats,generated_drumbeats),dim=0)
    
    #print("DEBUG size of output_drumbeats: ", output_drumbeats.shape)

    output_drumbeats = output_drumbeats[512:,:]
    #print("DEBUG size of output_drumbeats: ", output_drumbeats.shape)

    max0 = torch.max(output_drumbeats)
    min0 = torch.min(output_drumbeats)
    mean0 = output_drumbeats.mean()
    thres = mean0 + 0.45 * (max0 - mean0)
    mask = (output_drumbeats >= thres).type(torch.uint8)

    print("DEBUG sum of mask: ", mask.sum())
    print("ratio of mask: ", mask.sum()/512/128)

    
    output_drumbeats = mask*output_drumbeats
    output_drumbeats = (output_drumbeats - mean0) / (max0 - mean0)

    positive_mask = (output_drumbeats > 0).type(torch.uint8)
    output_drumbeats = positive_mask*output_drumbeats

    print("DEBUG min max of output_drumbeats: ", torch.min(output_drumbeats), torch.max(output_drumbeats))
    print("DEBUG mean of output_drumbeats: ", output_drumbeats.mean())

    output_drumbeats = output_drumbeats * 70
    print("DEBUG min max of output_drumbeats: ", torch.min(output_drumbeats), torch.max(output_drumbeats))
    output_drumbeats.type(torch.uint8)
    print(torch.sum(output_drumbeats))

    to_midi(output_drumbeats, index)


if __name__ == "__main__":
    main()


    


