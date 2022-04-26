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
          self.leaky_relu = nn.LeakyReLU()

      def forward(self, input):
            input = self.deconv(input)
            input = self.batchnorm(input)
            input = self.leaky_relu(input)
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
      
      def load_weights(self,PATH):
            temp = torch.load(PATH)
            self.load_state_dict(temp,strict=False)
                  

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


class GeneratorUnet(nn.Module):
      def __init__(self, unet_config, seq):
            super(GeneratorUnet, self).__init__()

            self.unet_config = unet_config
            self.conv_dict = nn.ModuleDict()
            self.deconv_dict = nn.ModuleDict()
            
            self.seq = seq
            encoding_depth = len(self.seq["conv"])
            decoding_depth = len(self.seq["deconv"])

            if encoding_depth != decoding_depth:
                  raise ValueError("Encoding and Decoding depth in GeneratorUnet() must be equal")
            
            self.depth = encoding_depth

            self.running_results = dict()

            for layer_key in unet_config["conv"].keys():
                  layer_info = unet_config["conv"][layer_key]
                  in_channels, out_channels, kernel_size, stride, padding = layer_info
                  self.conv_dict[layer_key] = conv_block(in_channels, out_channels, kernel_size, stride, padding)
            
            for layer_key in unet_config["deconv"].keys():
                  layer_info = unet_config["deconv"][layer_key]
                  in_channels, out_channels, kernel_size, stride, padding, d = layer_info
                  self.deconv_dict[layer_key] = generator_block(in_channels, out_channels, kernel_size, stride, d, padding)
            
            return 

      def forward(self, x):
            self.running_results[self.depth] = x.clone()
            for i, layer_key in enumerate(self.seq["conv"]):
                  b = False
                  if i == 0:
                        b = True
                  x = self.conv_dict[layer_key](x, batch=b)
                  self.running_results[self.depth-i-1] = x.clone()
            
            for i, layer_key in enumerate(self.seq["deconv"]):
                  if i == 0:
                        x_cat = x.clone()
                  else:
                        b = False
                        x_cat = torch.cat([x, self.running_results[i]], dim=1)
                  x = self.deconv_dict[layer_key](x_cat)
            
            return x

      def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




if __name__ == "__main__":
      # a = test_generator()
      # result = a.test()
      # print(result.size())
      
      #G = generator()

      if torch.cuda.is_available():
        device = torch.device("cuda")
        current_device = torch.cuda.current_device()
        print("CUDA engaged, Using GPU")
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using device: {device_name}")
      else:
        device = torch.device("cpu")
        print("CUDA not available, Using CPU")
      
      # G.to(device)
      # summary(G, (3, 512, 512))

      # === testbench for GeneratorUnet ===
      # also would serve as the new generator
      unet_config = {
            "conv": {
                  "conv0": (6, 16, (4,4), (2,2), (1,1)),
                  "conv1": (16, 32, (4,4), (2,2), (1,1)),
                  "conv2": (32, 64, (4,4), (2,2), (1,1)),
                  "conv3": (64, 128, (4,4), (2,2), (1,1)),
                  "conv4": (128, 256, (4,4), (2,2), (1,1)),
                  "conv5": (256, 256, (4,4), (2,2), (1,1)),
            },
            "deconv": {
                  "deconv0": (256, 256, (4,4), (2,2), (1,1), (1,1)),
                  "deconv1": (256*2, 128, (4,4), (2,2), (1,1), (1,1)),
                  "deconv2": (128*2, 64, (4,4), (2,2), (1,1), (1,1)),
                  "deconv3": (64*2, 32, (4,4), (2,2), (1,1), (1,1)),
                  "deconv4": (32*2, 16, (4,4), (2,2), (1,1), (1,1)),
                  "deconv5": (16*2, 1, (4,4), (2,2), (1,1), (1,1)),
            }
      }

      seq = {
            "conv": ["conv0", "conv1", "conv2", "conv3", "conv4", "conv5"],
            "deconv": ["deconv0", "deconv1", "deconv2", "deconv3", "deconv4", "deconv5"]
      }

      G = GeneratorUnet(unet_config, seq)
      G.weight_init()
      G.to(device)
      summary(G, (6, 512, 128))
      
