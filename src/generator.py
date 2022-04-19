import torch as torch
from torch.nn import ConvTranspose2d, Module, BatchNorm2d
import numpy as np
from loader import TempoSet

class generator_block(Module):
      def __init__(self, in_dim, out_dim, kernel, stride):
          super().__init__()
          self.deconv = ConvTranspose2d(in_dim, out_dim, kernel, stride)
          self.batchnorm = BatchNorm2d(out_dim)

      def forward(self, input):
            input = self.deconv(input)
            input = self.batchnorm(input)
            return torch.nn.functional.relu(input)


class generator(Module):

      def __init__(self):
            super().__init__()
            self.convtrans0 = generator_block(3, 128, 2, 2)
            self.convtrans1 = generator_block(128, 64, 2, 2)
            self.convtrans2 = generator_block(64, 32, 2, 2)
            self.convtrans3 = generator_block(32, 16, 2, 2)
            self.convtrans4 = generator_block(16, 1, 2, 2)

      def generate(self, x):
            x = self.convtrans0(x)
            x = self.convtrans1(x)
            x = self.convtrans2(x)
            x = self.convtrans3(x)
            x = self.convtrans4(x)

            return x

def training_loader(loader:TempoSet):
            """
            Sample inter-track random vector z and intra-track
            random vector z_i

            args:
                  loader: Temposet() object used to load data
                  M: how many z_i will be parsed i = 0,...,M-1 
            """
            # z_i: a segment of a track
            # z: genre vector of the track sampled
            ## randomly choose a track
            indx_of_song = np.random.randint(1, 10000)
            song, z = loader.__getitem__(indx_of_song)
            song_length = song.shape[0]
            z = z["genre"]
            
            # positional encoding:
            pos_code = torch.zeros(song_length, 1)
            for k in range(song_length): # TODO: this would be too slow, we need to vectorize this
                  if k%2 == 0:
                        w_k = 1/(np.power(10000, 2*k/song_length))
                        pos_code[k][0] = np.sin(w_k)
                  else:
                        w_k = 1/(np.power(10000, 2*k/song_length))
                        pos_code[k][0] = np.cos(w_k)

            ## zi: with positional encoding and genre added 
            # as a channel
            start = 0
            num_of_rows = 512
            z_i = song
            position = torch.zeros(num_of_rows, 512) + pos_code[start:start+num_of_rows ,:]
            genre = torch.zeros(8, 512) + z.view(8,1)
            zero_pad = torch.zeros(num_of_rows-8, 512)
            genre = torch.cat([genre, zero_pad], dim=0)
            z_i = torch.stack([z_i, genre, position], dim=2)
            """
            zi_list = torch.zeros(M, 3, 512, 512)
            for i in range(M):
                  start = np.random.randint(0, song_length - 60)
                  num_of_rows = 512
                  z_i = song[start:start + num_of_rows ,:]
                  
                  
                  zi_list[i ,: ,: ,:] = z_i.view(3, num_of_rows, 512)
            """
            return z_i 


class test_generator(Module):
      
      def __init__(self) -> None:
          super().__init__()

      def test(self):
            data_dir = "../data/lpd_5/lpd_5_cleansed"
            loader = TempoSet()
            loader.load(data_dir)

            G = generator()
            zi = training_loader(loader)

            print(zi.size())
            # generate pieces of songs:
            pieces = G.generate(zi)

            return pieces

if __name__ == "__main__":
      a = test_generator()
      result = a.test()
      print(result.size())
      
