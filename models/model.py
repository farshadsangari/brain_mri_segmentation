import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
class residual_UNET(nn.Module):
    def __init__(self,in_channels,out_channels,init_features):
        super(residual_UNET, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #######################   Encoder 1 ###########################
        
        self.encoder1 = residual_UNET.encoder_decoder(in_channels,init_features)
        self.conv1_1_e1 = nn.Conv2d(in_channels, init_features, kernel_size=(1,1), stride=(1, 1))
        #######################   Encoder 2 ###########################
        
        self.encoder2 = residual_UNET.encoder_decoder(init_features,init_features*2)
        self.conv1_1_e2 = nn.Conv2d(init_features, init_features*2, kernel_size=(1,1), stride=(1, 1))
        #######################   Encoder 3 ###########################
        
        self.encoder3 = residual_UNET.encoder_decoder(init_features*2,init_features*4)
        self.conv1_1_e3 = nn.Conv2d(init_features*2, init_features*4, kernel_size=(1,1), stride=(1, 1))
        #######################   Encoder 4 ###########################
        
        self.encoder4 = residual_UNET.encoder_decoder(init_features*4,init_features*8)
        self.conv1_1_e4 = nn.Conv2d(init_features*4, init_features*8, kernel_size=(1,1), stride=(1, 1))
        #######################   Bottleneck ###########################
        
        self.bottleneck = residual_UNET.encoder_decoder(init_features*8,init_features*16)
        self.conv1_1_b = nn.Conv2d(init_features*8, init_features*16, kernel_size=(1,1), stride=(1, 1))
        #######################   Decoder 4 ###########################
        
        self.upconv4 = nn.ConvTranspose2d(init_features*16,init_features*8, kernel_size=(2, 2), stride=(2, 2))
        self.decoder4 = residual_UNET.encoder_decoder(init_features*16,init_features*8)
        self.conv1_1_d4 = nn.Conv2d(init_features*16, init_features*8, kernel_size=(1,1), stride=(1, 1))
        #######################   Decoder 3 ###########################
        
        self.upconv3 = nn.ConvTranspose2d(init_features*8,init_features*4, kernel_size=(2, 2), stride=(2, 2))
        self.decoder3 = residual_UNET.encoder_decoder(init_features*8,init_features*4)
        self.conv1_1_d3 = nn.Conv2d(init_features*8, init_features*4, kernel_size=(1,1), stride=(1, 1))
        #######################   Decoder 2 ###########################
        self.upconv2 = nn.ConvTranspose2d(init_features*4,init_features*2, kernel_size=(2, 2), stride=(2, 2))
        self.decoder2 = residual_UNET.encoder_decoder(init_features*4,init_features*2)
        self.conv1_1_d2 = nn.Conv2d(init_features*4, init_features*2, kernel_size=(1,1), stride=(1, 1))
        #######################   Decoder 1 ###########################
        self.upconv1 = nn.ConvTranspose2d(init_features*2,init_features*1, kernel_size=(2, 2), stride=(2, 2))
        self.decoder1 = residual_UNET.encoder_decoder(init_features*2,init_features*1)
        self.conv1_1_d1 = nn.Conv2d(init_features*2, init_features, kernel_size=(1,1), stride=(1, 1))
        ###############   Last Convolutional Layer ####################
        self.last_conv = nn.Conv2d(init_features, out_channels, kernel_size=(1,1), stride=(1, 1))
        
    def forward(self,x):
        e1 = F.relu(self.encoder1(x) + self.conv1_1_e1(x))
        e2 = F.relu(self.encoder2(self.pool(e1)) + self.conv1_1_e2(self.pool(e1)))
        e3 = F.relu(self.encoder3(self.pool(e2)) + self.conv1_1_e3(self.pool(e2)))
        e4 = F.relu(self.encoder4(self.pool(e3)) + self.conv1_1_e4(self.pool(e3)))
        b = F.relu(self.bottleneck(self.pool(e4)) + self.conv1_1_b(self.pool(e4)))
        d4 = self.upconv4(b)
        d4 = torch.cat((e4,d4),dim=1)
        d4 = F.relu(self.decoder4(d4) + self.conv1_1_d4(d4))
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3,d3),dim=1)
        d3 = F.relu(self.decoder3(d3) + self.conv1_1_d3(d3))
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2,d2),dim=1)
        d2 = F.relu(self.decoder2(d2) + self.conv1_1_d2(d2))
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1,d1),dim=1)
        d1 = F.relu(self.decoder1(d1) + self.conv1_1_d1(d1))
        x = self.last_conv(d1)
        return torch.sigmoid(x)
    
    @staticmethod
    def encoder_decoder(input_features,init_features):
        return nn.Sequential(
            nn.Conv2d(input_features, init_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_features, init_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(init_features)
                            )