from turtle import position
import torch as torch
from torch.nn import ConvTranspose2d, LeakyReLU, Module, BatchNorm2d, ModuleList
import numpy as np
from loader import TempoSet

# TODO: TypeError: conv_transpose2d(): argument 'input' (position 1) must be Tensor, not LeakyReLU
#       Line 14
class generator_block(Module):
      def __init__(self, in_dim, out_dim, kernel, stride):
          super().__init__()
          self.deconv = ConvTranspose2d(in_dim, out_dim, kernel, stride)
          self.batchnorm = BatchNorm2d(out_dim)

      def forward(self, input):
            input = self.deconv(input)
            input = self.batchnorm(input)
            return LeakyReLU(input)


class generator(Module):

      def __init__(self):
            super().__init__()
            self.convtrans0 = generator_block(3, 128, 2, 1)
            self.convtrans1 = generator_block(128, 64, 2, 1)
            self.convtrans2 = generator_block(64, 32, 2, 1)
            self.convtrans3 = generator_block(32, 16, 2, 1)
            self.convtrans4 = ModuleList(
                  generator_block(16, 3, 2, 1)
                  for i in range(4)
            )

      def generate(self, x):
            x = self.convtrans0(x)
            x = self.convtrans1(x)
            x = self.convtrans2(x)
            x = self.convtrans3(x)
            x = [f(x) for f in self.convtrans4]

            return x

      def sampling(self, loader:TempoSet, M):
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
            zi_list = torch.zeros(M, 3, 50, 512)
            for i in range(M):
                  start = np.random.randint(0, song_length - 60)
                  num_of_rows = 50
                  z_i = song[start:start + num_of_rows ,:]
                  
                  position = torch.zeros(50, 512) + pos_code[start:start+50 ,:]
                  genre = torch.zeros(8, 512) + z.view(8,1)
                  zero_pad = torch.zeros(42, 512)
                  genre = torch.cat([genre, zero_pad], dim=0)
                  z_i = torch.stack([z_i, genre, position], dim=2)
                  zi_list[i ,: ,: ,:] = z_i.view(3, 50, 512)

            return z, zi_list 


class test_generator(Module):
      
      def __init__(self) -> None:
          super().__init__()

      def test(self):
            data_dir = "../data/lpd_5/lpd_5_cleansed"
            loader = TempoSet()
            loader.load(data_dir)

            G = generator()
            z, zi_list = G.sampling(loader, M=10)

            # generate pieces of songs:
            pieces = G.generate(zi_list)

            return pieces

if __name__ == "__main__":
      a = test_generator()
      result = a.test()

      print("generation finished !")
