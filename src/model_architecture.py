import torch.nn as nn
import torchinfo
import torch.nn.functional as F
from torchsummary import summary


class convLayer(nn.Module):
    def __init__(self, l_input_c, 
                 l_output_c, bias=False, 
                 padding=0, de_sep_conv=False, 
                 skip=False, dilation=1, 
                 dropout=0):
        super (convLayer, self).__init__()

        self.skip = skip

        if (de_sep_conv==True and l_input_c == l_output_c):
            self.convLayer = nn.Sequential(
                nn.Conv2d(in_channels=l_input_c, 
                          out_channels=l_output_c, 
                          kernel_size=(3, 3), 
                          groups=l_input_c,
                          padding= padding,
                          padding_mode='replicate',
                          dilation = dilation,
                          bias=bias),

                nn.BatchNorm2d(l_output_c),
                nn.ReLU(),
                nn.Conv2d(in_channels=l_output_c, 
                          out_channels=l_output_c, 
                          kernel_size=(1, 1),
                          bias=bias)
            )
        else:
            self.convLayer = nn.Sequential(
                nn.Conv2d(in_channels=l_input_c, 
                          out_channels=l_output_c, 
                          kernel_size=(3, 3), 
                          groups=1,
                          padding= padding,
                          padding_mode='replicate',
                          dilation = dilation,
                          bias=bias)
            )

        self.skip = skip
        self.skip_connection = None
        if (skip == True and l_input_c != l_output_c):
            self.skip_connection = nn.Conv2d(
                                             in_channels=l_input_c, 
                                             out_channels=l_output_c, 
                                             kernel_size=(1, 1),
                                             bias = bias                                              
                                             )

        self.normLayer = nn.BatchNorm2d(l_output_c)

        self.activationLayer = nn.ReLU()

        self.dropout = None
        if(dropout > 0):
            self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        tmp_x = x
        x = self.convLayer(x)
        x = self.normLayer(x)

        if (self.skip == True):
            if (self.skip_connection is None):
                x = x + tmp_x
            else:
                x = x + self.skip_connection(tmp_x)

        x = self.activationLayer(x)
        
        if (self.dropout == None):
            x = self.dropout(x)

        return x
    

class Net(nn.Module):
    def __init__(self, dropout=0, skip=False):
        super(Net, self).__init__()

        #self.dropout = dropout

        ##### Convolution Block - 1 #####
        # Dilation=1 applied in the 2 convolutions layers created with this code
        # After the first 3x3 convolution, a 1x1 convolution is created
        # so that the features obtained from the first comvolution layer of this block 
        # is mixed with the second 3x3 convolution layer along with creating it as depthwise 
        # separable convolution 
        self.convblock1 = self.convBlock(in_channels=3, out_channels=23, 
                                         padding=1, skip=False,
                                         de_sep_conv=True, dropout= dropout,
                                         dilation=1, num_iter=2
                                         ) # output_size = 32, rf_out = 5
        

        self.transblock1 = self.transitionBlock(in_channels=23, out_channels=32,
                                                padding=0, skip=False,
                                                de_sep_conv=False, dropout= dropout,
                                                dilation=1) # output_size = 30, rf_out = 7
        
        ##### Convolution Block - 2 #####
        # Dilation=1 applied in the 2 convolutions layers created with this code
        # After the first 3x3 convolution, a 1x1 convolution is created
        # so that the features obtained so far are mixed with the second 3x3 convolution layer 
        # along with creating it as depthwise separable convolution. Also, skip connection is 
        # created for these layers
        self.convblock2 = self.convBlock(in_channels=32, out_channels=32, 
                                    padding=1, skip=skip,
                                    de_sep_conv=True, dropout= dropout,
                                    dilation=1, num_iter=2
                                    ) # output_size = 30, rf_out = 11

        # Dilation=2 applied
        self.transblock2 = self.transitionBlock(32, 63,
                                        padding=0, skip=False,
                                        de_sep_conv=False, dropout= dropout,
                                        dilation=2) # output_size = 26, rf_out = 15
        
        ##### Convolution Block - 3 #####
        # Dilation=1 applied in the 2 convolutions layers created with this code
        # After the first 3x3 convolution, a 1x1 convolution is created
        # so that the features obtained so far are mixed with the second 3x3 convolution layer 
        # along with creating it as depthwise separable convolution. Also, skip connection is 
        # created for these layers         
        self.convblock3 = self.convBlock(in_channels=63, out_channels=63, 
                                    padding=1, skip=skip,
                                    de_sep_conv=True, dropout= dropout,
                                    dilation=1, num_iter=2
                                    ) # output_size = 26, rf_out = 19

        # Dilation=4 applied
        self.transblock3 = self.transitionBlock(in_channels=63, out_channels=93,
                                        padding=0, skip=False,
                                        de_sep_conv=False, dropout= dropout,
                                        dilation=4) # output_size = 18, rf_out = 27
        
        ##### Convolution Block - 4 #####
        # Dilation=1 applied in the 2 convolutions layers created with this code
        # After the first 3x3 convolution, a 1x1 convolution is created
        # so that the features obtained so far are mixed with the second 3x3 convolution layer 
        # along with creating it as depthwise separable convolution. Also, skip connection is 
        # created for these layers
        self.convblock4 = self.convBlock(in_channels=93, out_channels=93, 
                                    padding=1, skip=skip,
                                    de_sep_conv=True, dropout= dropout,
                                    dilation=1, num_iter=2
                                    ) # output_size = 18, rf_out = 31

        # Dilation=8 applied
        self.transblock4 = self.transitionBlock(in_channels=93, out_channels=93,
                                        padding=0, skip=False,
                                        de_sep_conv=False, dropout= dropout,
                                        dilation=8) # output_size = 2, rf_out = 47
        
        ##### Output Block #####
        # GAP + 1x1
        self.outblock = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(93, 10, kernel_size= (1,1), bias=True),
                                nn.Flatten(),
                                nn.LogSoftmax(-1)
                                ) # output_size = 1, rf_out = 48

    @staticmethod
    def  convBlock(in_channels, out_channels, 
                    bias=False, padding=0, 
                    skip=False, de_sep_conv=False, 
                    dilation=1, dropout=0, num_iter=2):   

        layer_seq = []

        for cnt in range(0, num_iter):
            if cnt > 0:
                layer_seq.append(
                    convLayer(l_input_c=out_channels,
                                l_output_c=out_channels, 
                                bias=bias, padding=padding,
                                de_sep_conv=de_sep_conv, skip=skip, 
                                dilation=dilation, dropout=dropout))
            else:
                layer_seq.append(
                    convLayer(l_input_c=in_channels,
                                l_output_c=out_channels, 
                                bias=bias, padding=padding,
                                de_sep_conv=de_sep_conv, skip=skip, 
                                dilation=dilation, dropout=dropout))
        
        return nn.Sequential(*layer_seq)
    
    @staticmethod    
    def transitionBlock(in_channels, out_channels, 
                    bias=False, padding=0, 
                    skip=False, de_sep_conv=False, 
                    dilation=1, dropout=0):
        
        tr_layer_seq =  convLayer(l_input_c=in_channels, l_output_c=out_channels,
                            bias=bias, padding=padding,
                            de_sep_conv=de_sep_conv, skip=skip, 
                            dilation=dilation, dropout=dropout)
        
        return tr_layer_seq
    

    def forward(self, x):

        x = self.convblock1(x)
        x = self.transblock1(x)        
        x = self.convblock2(x)
        x = self.transblock2(x)
        x = self.convblock3(x)
        x = self.transblock3(x)        
        x = self.convblock4(x)
        x = self.transblock4(x)   
        x = self.outblock(x)

        return x
    
    # Network Summary
    def summary(self, input_size=None):
        return torchinfo.summary(self, input_size=input_size, 
                                 col_names=["input_size", 
                                            "output_size", 
                                            "num_params",
                                            "kernel_size", 
                                            "params_percent"])     
        