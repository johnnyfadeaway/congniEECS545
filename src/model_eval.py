from re import TEMPLATE
from generator import generator as ge
from loader import TempoSet
import torch 
import numpy
import pypianoroll as ppr
from pypianoroll import track

from test import to_midi

def main():
    loader = TempoSet()
    data_dir = "../data/lpd_5/lpd_5_cleansed"
    loader.load(data_dir)
    chucks,drum_chucks = loader.get_chunks_from_song(545)
    num_of_chucks = len(chucks)
    #print("DEBUG num_of_chucks: ",num_of_chucks)
    #print("DEBUG shape of chuck: ",chucks[0].shape)

    output_drumbeats = torch.zeros(512,128)
    generator = ge()

    PATH = "../model/generator_l2_20220421_073949.pth"

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
    print(max0)
    min0 = torch.min(output_drumbeats)
    print(min0)
    mean0 = output_drumbeats.mean()
    print(mean0)
    #mask = (output_drumbeats >= output_drumbeats.mean()).type(torch.uint8)
    #output_drumbeats = mask*output_drumbeats
    #output_drumbeats = (output_drumbeats - mean0) / (max0 - mean0)
    #output_drumbeats = output_drumbeats * 127
    output_drumbeats.type(torch.uint8)
    print(torch.sum(output_drumbeats))

    #to_midi(output_drumbeats,1)


if __name__ == "__main__":
    main()


    


