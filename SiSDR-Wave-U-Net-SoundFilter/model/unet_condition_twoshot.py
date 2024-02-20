## python train.py -C config/train/train.json
import torch
import torch.nn as nn
import torch.nn.functional as F

### ADDITIVE SKIP CONNECTION OR CONCAT??? # # non-linearity addition #batch normalization, non linearity in residual unit?
class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out,stride=2):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=2*stride,
                      stride=stride),
            nn.GroupNorm(channel_out//16,channel_out),
            nn.ELU(alpha=1.0)
        )
        self.residual11 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=1),nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))
        self.residual12 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1),nn.GroupNorm(channel_in//16,channel_in),  nn.ELU(alpha=1.0))

        self.residual21 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=3), nn.GroupNorm(channel_in//16,channel_in),nn.ELU(alpha=1.0))
        self.residual22 =  nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1), nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))

        self.residual31 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=9),nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))
        self.residual32 =  nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1), nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))



    def forward(self, ipt):
        o1 = ipt

        #print("Downsampling ipt:",o1.shape)
        o2 = self.residual11(o1)
        #print("Downsampling o2:",o2.shape)

        o3 = self.residual12(o2)
        #print("Downsampling o3:",o3.shape)
        o4 = self.residual21(o3 + o1) ############# CONCAT ? #############
        #print("Downsampling o4:",o4.shape)
        o5 = self.residual22(o4)
        #print("Downsampling o5:",o5.shape)
        o6 = self.residual31(o1+o3+o5) ############# CONCAT ? #############
        #print("Downsampling o6:",o6.shape)
        o7 = self.residual32(o6)
        #print("Downsampling o7:",o7.shape)
        o8 = self.main(o1 + o3 + o5 + o7)
        #print("Downsampling o8:",o8.shape)
        return o8 ############# CONCAT ? #############

class FilmDownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out,stride=2):
        super(FilmDownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=2*stride,
                      stride=stride),
            nn.GroupNorm(channel_out//16,channel_out),
            nn.ELU(alpha=1.0)
        )
        self.residual11 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=1),nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))
        self.residual12 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1),nn.GroupNorm(channel_in//16,channel_in),  nn.ELU(alpha=1.0))

        self.residual21 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=3), nn.GroupNorm(channel_in//16,channel_in),nn.ELU(alpha=1.0))
        self.residual22 =  nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1), nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))

        self.residual31 = nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 3, padding='same', dilation=9),nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))
        self.residual32 =  nn.Sequential(nn.Conv1d(channel_in, channel_in,kernel_size= 1), nn.GroupNorm(channel_in//16,channel_in), nn.ELU(alpha=1.0))



    def forward(self, ipt):
        o1 = ipt
        o2 = self.residual11(o1)
        o3 = self.residual12(o2)
        o4 = self.residual21(o3 + o1) ############# CONCAT ? #############
        o5 = self.residual22(o4)
        o6 = self.residual31(o1+o3+o5) ############# CONCAT ? #############
        o7 = self.residual32(o6)
        o8 = self.main(o1 + o3 + o5 + o7)
        return o8 ############# CONCAT ? #############
        

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, Stride):
        super(UpSamplingLayer, self).__init__()
        if(channel_in == 512): 
            output_padding = 0
        elif(channel_in == 256):
            output_padding = 5
        else:
            output_padding = 0

        self.upblock = nn.Sequential(
            nn.ConvTranspose1d(channel_in,channel_out, kernel_size=2*Stride,
                      stride=Stride, output_padding=output_padding),
            nn.GroupNorm(channel_out//16,channel_out),
            nn.ELU(alpha=1.0)
            #nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.residual11 = nn.Sequential(nn.Conv1d(channel_out, channel_out, kernel_size= 3, padding='same', dilation=1),nn.GroupNorm(channel_out//16,channel_out),  nn.ELU(alpha=1.0))
        self.residual12 = nn.Sequential(nn.Conv1d(channel_out, channel_out,kernel_size= 1), nn.GroupNorm(channel_out//16,channel_out),nn.ELU(alpha=1.0))
        self.residual21 = nn.Sequential(nn.Conv1d(channel_out, channel_out,kernel_size= 3, padding='same', dilation=3),nn.GroupNorm(channel_out//16,channel_out), nn.ELU(alpha=1.0))
        self.residual22 = nn.Sequential(nn.Conv1d(channel_out, channel_out,kernel_size= 1),nn.GroupNorm(channel_out//16,channel_out), nn.ELU(alpha=1.0))
        self.residual31 = nn.Sequential(nn.Conv1d(channel_out, channel_out,kernel_size= 3, padding='same', dilation=9),nn.GroupNorm(channel_out//16,channel_out), nn.ELU(alpha=1.0))
        self.residual32 = nn.Sequential(nn.Conv1d(channel_out, channel_out,kernel_size= 1),nn.GroupNorm(channel_out//16,channel_out), nn.ELU(alpha=1.0))

    def forward(self, ipt,gamma1,gamma2,beta1,beta2):
        #print("Upsampling ipt:",ipt.shape)
        bs = gamma1.shape[0]
        #print("BATCH SIZE:",bs)
        o1 = self.upblock(ipt)
        #print ("o1", o1.shape, "gamma",gamma.shape, "beta",beta.shape)
        o1 = gamma1.view(bs,1,1) * o1 + beta1.view(bs,1,1)
        #print("Upsampling o1:",o1.shape)
        o2 = self.residual11(o1)
        #print("Upsampling o2:",o2.shape)
        o3 = self.residual12(o2)
        #print("Upsampling o3:",o3.shape)
        o4 = self.residual21(o3 + o1) ############# CONCAT ? #############
        #print("Upsampling o4:",o4.shape)

        o5 = self.residual22(o4)
        #print("Upsampling o5:",o5.shape)

        o6 = self.residual31(o1+o3+o5) ############# CONCAT ? #############
        #print("Upsampling o6:",o6.shape)

        o7 = self.residual32(o6)
        #print("Upsampling o7:",o7.shape)

        o8 = o1 + o3 + o5 + o7
        o8 = gamma2.view(bs,1,1) * o8 + beta2.view(bs,1,1)
        #print("Upsampling o8:",o8.shape)

        return o8 ############# CONCAT ? #############
        
        

class Model(nn.Module):
    def __init__(self, n_layers=4):
        super(Model, self).__init__()

        #########################################################
        # ENCODER
        #########################################################
        self.encConv1 = nn.Sequential(nn.Conv1d(1, 32,kernel_size= 7), nn.GroupNorm(2,32), nn.ELU(alpha=1.0))
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
        self.encConv2 = nn.Sequential(nn.Conv1d(512, 256,kernel_size= 7,padding = 'same'), nn.GroupNorm(16,256), nn.ELU(alpha=1.0))

        #########################################################
        # FILM
        #########################################################
        film_in_channels_list = [32,64,128,256]
        film_out_channels_list = [64,128,256,512]
        modules = []
        for i in range(self.n_layers):
            modules.append(
                FilmDownSamplingLayer(
                    channel_in=film_in_channels_list[i],
                    channel_out=film_out_channels_list[i],
                    stride = int(encoder_stride_list[i])
                )
            )
        #modules.append(nn.Flatten())
        #modules.append(nn.Linear(31744,2*(2*n_layers +1)))
        #self.film_gen = nn.Sequential(nn.Conv1d(1, 32,kernel_size= 7), nn.BatchNorm1d(32), nn.ELU(alpha=1.0), *modules)

        self.first_conv = nn.Sequential(nn.Conv1d(1, 32,kernel_size= 7), nn.GroupNorm(2,32), nn.ELU(alpha=1.0))
        self.list = nn.Sequential(*modules,nn.Conv1d(512, 256,kernel_size= 7,padding = 'same'),
                                  nn.BatchNorm1d(256), nn.ELU(alpha=1.0),
                                  nn.AvgPool1d(kernel_size= 3, stride=3),
                                  nn.ELU(alpha=1.0))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2560,(6*n_layers +2))
        #self.film_gen = nn.Sequential(*modules)
        #########################################################
        # DECODER
        #########################################################

        self.decConv1 = nn.Sequential(nn.Conv1d(256, 512 ,kernel_size= 7,padding = 'same'), nn.GroupNorm(32,512), nn.ELU(alpha=1.0))
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
        self.decConv2 = nn.Sequential(nn.Conv1d(32, 1 ,padding = 6,kernel_size= 7),  nn.BatchNorm1d(1), nn.ELU(alpha=1.0))
        self.double()

    def forward_condition(self,conditioning1, conditioning2):
        x = self.first_conv(conditioning1)
        #print ('before convlist cond1',x.shape)
        x = self.list(x)
        #print ('before flatten cond1',x.shape)
        #x = torch.mean(x,2)
        #print ('after mean cond1',x.shape)
        x = self.flatten(x)
        #print ('before linear cond1',x.shape)
        film_out1 = self.linear(x)
        #print ('after linear cond1',film_out1.shape)

        x = self.first_conv(conditioning2)
        x = self.list(x)
        #x = torch.mean(x,2)
        x = self.flatten(x)
        film_out2 = self.linear(x)

        return film_out1, film_out2

    def forward(self, input, conditioning1, conditioning2):
        tmp = []
        o = input
        #print ('test2', input.shape)
        o = o.double()
        conditioning1 = conditioning1.double()
        conditioning2 = conditioning2.double()
        tmp.append(o)
        # Up Sampling
        #print(">>>>>>>>>>>>>",o.dtype)
        o = self.encConv1(o)
        tmp.append(o)
        #print("********",self.film_gen)
        #film_out1 = self.film_gen(conditioning1)
        #film_out2 = self.film_gen(conditioning2)
        film_out1, film_out2 = self.forward_condition(conditioning1, conditioning2)
        #print("FILM:",film_out1.shape, film_out2.shape)
        gammas = (film_out1[:,:3*self.n_layers+1] + film_out2[:,:3*self.n_layers+1] )/2
        betas = (film_out1[:,3*self.n_layers+1:] + film_out2[:,3*self.n_layers+1:])/2
        #print("gamma,beta:",gammas.shape,betas.shape)
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            #print("OO",o.shape)
            o = o*(gammas[0,i]) + (betas[0,i]) #affine transform
            tmp.append(o)

        #print ('test1', tmp[-1].shape)
        o = self.encConv2(o)
        #print("encConv2:",o.shape)
        o = o*gammas[0,self.n_layers] + betas[0,self.n_layers]
        o = self.decConv1(o)
        #print("decConv1:",o.shape, tmp[-1].shape)
        
        o = o + tmp[-1] ############# CONCAT ? #############
        # Down Sampling
        for i in range(self.n_layers):
            o = self.decoder[i](o,(gammas[:,self.n_layers+i+1]),gammas[:,self.n_layers+i+2],
                                (betas[:,self.n_layers+i+1]),betas[:,self.n_layers+i+2])
            #o = o*gammas[self.n_layers+i+1] + betas[self.n_layers+i+1] #affine transform
            #print("downsampling", o.shape,tmp[self.n_layers - i].shape)
            #print ('decoding stage',i, tmp[self.n_layers-i].shape, o.shape)
            o = o + tmp[self.n_layers - i] ############# CONCAT ? #############
        #print("before:",o.shape,tmp[0].shape)
        o = self.decConv2(o)
        #print("after:",o.shape,tmp[0].shape)

        o = o + tmp[0]  ############# CONCAT ? #############
        
        return o
