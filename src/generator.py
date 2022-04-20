import torch as torch
from torch.nn import Conv2d, ConvTranspose2d, Module, BatchNorm2d, LeakyReLU, ReLU
import torch.nn as nn
import numpy as np
from loader import GANdataset, TempoSet, ClassifierSet
from torchsummary import summary


class generator_block(Module):
      def __init__(self, in_dim, out_dim, kernel, stride, d, p):
          super().__init__()
          self.deconv = ConvTranspose2d(in_dim, out_dim, kernel, stride, dilation=d, padding=p)
          self.batchnorm = BatchNorm2d(out_dim)
          self.relu = ReLU()

      def forward(self, input):
            input = self.deconv(input)
            input = self.batchnorm(input)
            input = self.relu(input)
            return input

class conv_block(Module):
      def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
          super().__init__()
          self.conv2d = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
          self.batchnorm = BatchNorm2d(out_channels)
          self.leaky_relu = LeakyReLU(0.1)
      
      def forward(self, input, batch=True):
            input = self.conv2d(input)

            if batch == True:
                  input = self.batchnorm(input)
            input = self.leaky_relu(input)
            
            return input

class generator(Module):

      def __init__(self):
            super().__init__()
            self.conv0 = conv_block(in_channels=3, out_channels=32, kernel_size=(4,4), stride=(2,2), padding=(1,1))
            self.conv1 = conv_block(in_channels=32, out_channels=64, kernel_size=(4,4), stride=(2,2), padding=(1,1))
            self.conv2 = conv_block(in_channels=64, out_channels=128, kernel_size=(4,4), stride=(2,2), padding=(1,1))
            self.conv3 = conv_block(in_channels=128, out_channels=128, kernel_size=(2,2), stride=(2,2), padding=(0,0))
            self.conv4 = conv_block(in_channels=128, out_channels=128, kernel_size=(2,2), stride=(2,2), padding=(0,0))
            
            self.convtrans0 = generator_block(128, 128, kernel=(2,2), stride=(2,2), d=(1,1), p=(0,0))
            self.convtrans1 = generator_block(128, 64, kernel=(2,2), stride=(2,2), d=(1,1), p=(0,0))
            self.convtrans2 = generator_block(64, 32, kernel=(2,1), stride=(2,1), d=(1,1), p=(0,0))
            self.convtrans3 = generator_block(32, 16, kernel=(2,1), stride=(2,1), d=(1,1), p=(0,0))
            # self.convtrans4 = generator_block(16, 1, kernel=(2,2), stride=(2,2), d=(1,1), p=(0,0))
            self.last_conv_trans = nn.ConvTranspose2d(16, 1, kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), padding=(0, 0))
            self.last_active = nn.Sigmoid()

      def forward(self, x):
            x = self.conv0(x, batch=False)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x, batch=False)

            x = self.convtrans0(x)
            x = self.convtrans1(x)
            x = self.convtrans2(x)
            x = self.convtrans3(x)
            x = self.last_conv_trans(x)
            x = self.last_active(x)

            return x
      
      def weight_init(self, mean, std):
            for m in self.modules():
                  if isinstance(m, Conv2d):
                        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                  

def training_loader(loader:GANdataset, indx):
            """
            Put in 5 songs at once

            Args:
                  indx: the index of the song
            """
            z_i, drum = loader[indx]
            return z_i 

class test_generator(Module):
      
      def __init__(self) -> None:
          super().__init__()

      def test(self):
            data_dir = "../data/lpd_5/lpd_5_cleansed"
            tempo = TempoSet()
            tempo.load(data_dir)
            c_loader = ClassifierSet(tempo)
            gan_loader = GANdataset(c_loader)

            G = generator()
            zi = training_loader(gan_loader, indx=10)

            print(zi.size())
            # generate pieces of songs:
            pieces = G.generate(zi)

            return pieces

def generator_encode_forger(in_dim, out_dim, kernel, stride, p, batch=True):
      layer = nn.ModuleList()
      layer.append(nn.Conv2d(in_dim, out_dim, kernel, stride, padding=p))
      if batch:
            layer.append(nn.BatchNorm2d(out_dim))
      layer.append(nn.LeakyReLU(0.2))
      return layer


class generator_unet(Module):
      def __init__(self, generator_config):
            super().__init__()
            self.layers = nn.ModuleList()
            for layer_info in generator_config["encoder"]:
                  in_dim, out_dim, kernel, stride, _, p, batch = layer_info
                  self.layers.append(generator_encode_forger(in_dim, out_dim, kernel, stride, p, batch))
            
            for layer_info in generator_config["decoder"]:
                  in_dim, out_dim, kernel, stride, d, p, batch = layer_info

if __name__ == "__main__":
      # a = test_generator()
      # result = a.test()
      # print(result.size())
      G = generator()

      if torch.cuda.is_available():
        device = torch.device("cuda")
        current_device = torch.cuda.current_device()
        print("CUDA engaged, Using GPU")
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using device: {device_name}")
      else:
        device = torch.device("cpu")
        print("CUDA not available, Using CPU")
      
      G.to(device)
      summary(G, (3, 512, 512))
      
