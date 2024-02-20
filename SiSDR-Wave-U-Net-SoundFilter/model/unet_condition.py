
## python train.py -C config/train/train.json
import torch
import torch.nn as nn
import torch.nn.functional as F

### ADDITIVE SKIP CONNECTION OR CONCAT??? # # non-linearity addition #batch normalization, non linearity in residual unit?
class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out,stride=2):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(channel_in, channel_out, kernel_size=2*stride,
                      stride=stride),
            nn.BatchNorm1d(channel_out),
            nn.ELU(alpha=1.0)
        )
        self.residual11 = nn.Sequential(nn.Conv1d(channel_in, channel_out,kernel_size= 3, padding=0, dilation=1),nn.BatchNorm1d(channel_out), nn.ELU(alpha=1.0))
        self.residual12 = nn.Sequential(nn.Conv1d(channel_in, channel_out,kernel_size= 1),nn.BatchNorm1d(channel_out),  nn.ELU(alpha=1.0))

        self.residual21 = nn.Sequential(nn.Conv1d(channel_in, channel_out,kernel_size= 3, padding=0, dilation=3), nn.BatchNorm1d(channel_out),nn.ELU(alpha=1.0))
        self.residual22 =  nn.Sequential(nn.Conv1d(channel_in, channel_out,kernel_size= 1), nn.BatchNorm1d(channel_out), nn.ELU(alpha=1.0))

        self.residual31 = nn.Sequential(nn.Conv1d(channel_in, channel_out,kernel_size= 3, padding=0, dilation=9),nn.BatchNorm1d(channel_out), nn.ELU(alpha=1.0))
        self.residual32 =  nn.Sequential(nn.Conv1d(channel_in, channel_out,kernel_size= 1), nn.BatchNorm1d(channel_out), nn.ELU(alpha=1.0))



    def forward(self, ipt,gamma, beta):
        o1 = self.main(ipt)
        o1 = gamma * o1 + beta
        o2 = self.residual11(o1)
        o3 = self.residual12(o2)
        o4 = self.residual21(o3 + o1) ############# CONCAT ? #############
        o5 = self.residual22(o4)
        o6 = self.residual31(o1+o3+o5) ############# CONCAT ? #############
        o7 = self.residual32(o6)
        o8 = o1 + o3 + o5 + o7
        o8 = gamma * o8 + beta
        return o8 ############# CONCAT ? #############

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, Stride):
        super(UpSamplingLayer, self).__init__()
        self.upblock = nn.Sequential(
            nn.Conv1d(int(channel_out/2), channel_out, kernel_size=2*Stride,
                      stride=Stride),
            nn.BatchNorm1d(channel_out),
            nn.ELU(alpha=1.0)
            #nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.residual11 = nn.Sequential(nn.Conv1d(channel_in, int(channel_out/2),kernel_size= 3, padding=0, dilation=1),nn.BatchNorm1d( int(channel_out/2)),  nn.ELU(alpha=1.0))
        self.residual12 = nn.Sequential(nn.Conv1d(int(channel_out/2), int(channel_out/2),kernel_size= 1), nn.BatchNorm1d( int(channel_out/2)),nn.ELU(alpha=1.0))

        self.residual21 = nn.Sequential(nn.Conv1d(int(channel_out/2), int(channel_out/2),kernel_size= 3, padding=0, dilation=3),nn.BatchNorm1d( int(channel_out/2)), nn.ELU(alpha=1.0))
        self.residual22 = nn.Sequential(nn.Conv1d(int(channel_out/2), int(channel_out/2),kernel_size= 1),nn.BatchNorm1d( int(channel_out/2)), nn.ELU(alpha=1.0))

        self.residual31 = nn.Sequential(nn.Conv1d(int(channel_out/2), int(channel_out/2),kernel_size= 3, padding=0, dilation=9),nn.BatchNorm1d( int(channel_out/2)), nn.ELU(alpha=1.0))
        self.residual32 = nn.Sequential(nn.Conv1d(int(channel_out/2), int(channel_out/2),kernel_size= 1),nn.BatchNorm1d( int(channel_out/2)), nn.ELU(alpha=1.0))

    def forward(self, ipt):
        o1 = ipt
        o2 = self.residual11(o1)
        o3 = self.residual12(o2)
        o4 = self.residual21(o3 + o1) ############# CONCAT ? #############
        o5 = self.residual22(o4)
        o6 = self.residual31(o1+o3+o5) ############# CONCAT ? #############
        o7 = self.residual32(o6)
        return self.upblock(o1 + o3 + o5 + o7) ############# CONCAT ? #############
        

class Model(nn.Module):
    def __init__(self, n_layers=4):
        super(Model, self).__init__()

        #########################################################
        # ENCODER
        #########################################################
        self.encConv1 = nn.Sequential(nn.Conv1d(1, 32,kernel_size= 7), nn.BatchNorm1d(32), nn.ELU(alpha=1.0))
        self.n_layers = n_layers
        
        encoder_in_channels_list = [32,64,128,256]
        encoder_out_channels_list = [64,128,256,512]
        encoder_stride_list = [2,2,8,8]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i],
                    stride = int(encoder_stride_list[i])
                )
            )
        self.encConv2 = nn.Sequential(nn.Conv1d(512, 256,kernel_size= 7), nn.BatchNorm1d(256), nn.ELU(alpha=1.0))

        #########################################################
        # FILM
        #########################################################
        modules = []
        for i in range(self.n_layers):
            modules.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i],
                    stride = int(encoder_stride_list[i])
                )
            )
        modules.append(nn.Flatten())
        modules.append(nn.Linear(1024,2*(n_layers +1)))
        self.film_gen = nn.Sequential(*modules)
        #########################################################
        # DECODER
        #########################################################

        self.decConv1 = nn.Sequential(nn.Conv1d(256, 512 ,kernel_size= 7), nn.BatchNorm1d(512), nn.ELU(alpha=1.0))
        decoder_in_channels_list = [512,256,128,64]
        decoder_out_channels_list = [256,128,64,32]
        decoder_stride_list = [8,8,2,2]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                    Stride = int(decoder_stride_list[i])
                )
            )
        self.decConv1 = nn.Sequential(nn.Conv1d(32, 1 ,kernel_size= 7),  nn.BatchNorm1d(1), nn.ELU(alpha=1.0))


    def forward(self, input, conditioning1, conditioning2):
        tmp = []
        o = input
        tmp.append(o)
        # Up Sampling
        o = self.encConv1(o)
        tmp.append(o) 
        film_out1 = self.film_gen(conditioning1)
        film_out2 = self.film_gen(conditioning2)

        gammas = (film_out1[:self.n_layers+1] + film_out2[:self.n_layers+1])/2
        betas = (film_out1[self.n_layers+1:] + film_out2[self.n_layers+1:])/2
        for i in range(self.n_layers):
            o = self.encoder[i](o,gammas[i],betas[i])
            #o = o*gammas[i] + betas[i] #affine transform
            tmp.append(o)


        o = self.encConv2(o)
        o = o*gammas[self.n_layers] + betas[self.n_layers]
        o = self.decConv1(o)
        
        o = o + tmp[-1] ############# CONCAT ? #############
        # Down Sampling
        for i in range(self.n_layers):
            o = self.decoder[i](o)
            o = o*gammas[self.n_layers+i+1] + betas[self.n_layers+i+1] #affine transform
            o = o + tmp[self.n_layers - i] ############# CONCAT ? #############

        o = self.decConv1(o)
        o = o + tmp[0]  ############# CONCAT ? #############
        return o
