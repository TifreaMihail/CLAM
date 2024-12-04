import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import initialize_weights

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 512, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
        

class DepthwiseSeparableConv(nn.Module):
    """
    Implements Depthwise Separable Convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   padding=0, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.relu(out)
        return out

class DoubleConv_small(nn.Module):
    """
    Two consecutive Depthwise Separable Convolutions.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_small, self).__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )
        
    def forward(self, x):
        return self.conv(x)

class UNet_small(nn.Module):
    def __init__(self, embed_dim=1024):
        """
        Initializes the U-Net model. takes input (batch_size, embed_dim, width, height)
        
        Parameters:
        - embed_dim (int): Number of input channels.
        """
        super(UNet_small, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Encoder
        self.enc1 = DoubleConv_small(embed_dim, embed_dim)  # Retain embed_dim channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck1 = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.enc2 = DoubleConv_small(embed_dim // 2, embed_dim // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck2 = nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.enc3 = DoubleConv_small(embed_dim // 4, embed_dim // 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck3 = nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.enc4 = DoubleConv_small(embed_dim // 8, embed_dim // 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck4 = nn.Conv2d(embed_dim // 8, embed_dim // 16, kernel_size=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Bottleneck
        self.bottleneck = DoubleConv_small(embed_dim // 16, embed_dim // 16)
        
        # Decoder
        self.up6 = nn.ConvTranspose2d(embed_dim // 16, embed_dim // 16, kernel_size=2, stride=2)
        self.dec6 = DoubleConv_small(embed_dim // 16 + embed_dim // 8, embed_dim // 8)
        
        self.up7 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 8, kernel_size=2, stride=2)
        self.dec7 = DoubleConv_small(embed_dim // 8 + embed_dim // 4, embed_dim // 4)
        
        self.up8 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2)
        self.dec8 = DoubleConv_small(embed_dim // 4 + embed_dim // 2, embed_dim // 2)
        
        self.up9 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 2, kernel_size=2, stride=2)
        self.dec9 = DoubleConv_small(embed_dim // 2 + embed_dim, embed_dim)
        
    def forward(self, x):
        # Encoder Path
        enc1 = self.enc1(x)  # (B, embed_dim, W, H)
        pool1 = self.pool1(enc1)
        bottleneck1 = self.relu1(self.bottleneck1(pool1))
        
        enc2 = self.enc2(bottleneck1)  # (B, embed_dim//2, W/2, H/2)
        pool2 = self.pool2(enc2)
        bottleneck2 = self.relu2(self.bottleneck2(pool2))
        
        enc3 = self.enc3(bottleneck2)  # (B, embed_dim//4, W/4, H/4)
        pool3 = self.pool3(enc3)
        bottleneck3 = self.relu3(self.bottleneck3(pool3))
        
        enc4 = self.enc4(bottleneck3)  # (B, embed_dim//8, W/8, H/8)
        pool4 = self.pool4(enc4)
        bottleneck4 = self.relu4(self.bottleneck4(pool4))
        
        # Bottleneck
        bottleneck = self.bottleneck(bottleneck4)  # (B, embed_dim//16, W/16, H/16)
        
        # Decoder Path
        up6 = self.up6(bottleneck)  # (B, embed_dim//16, W/8, H/8)
        up6 = torch.cat([up6, enc4], dim=1)  # Concatenate along channel dimension
        dec6 = self.dec6(up6)  # (B, embed_dim//8, W/8, H/8)
        
        up7 = self.up7(dec6)  # (B, embed_dim//8, W/4, H/4)
        up7 = torch.cat([up7, enc3], dim=1)  # (B, embed_dim//4, W/4, H/4)
        dec7 = self.dec7(up7)  # (B, embed_dim//4, W/4, H/4)
        
        up8 = self.up8(dec7)  # (B, embed_dim//4, W/2, H/2)
        up8 = torch.cat([up8, enc2], dim=1)  # (B, embed_dim//2, W/2, H/2)
        dec8 = self.dec8(up8)  # (B, embed_dim//2, W/2, H/2)
        
        up9 = self.up9(dec8)  # (B, embed_dim//2, W, H)
        up9 = torch.cat([up9, enc1], dim=1)  # (B, embed_dim + embed_dim//2, W, H)
        dec9 = self.dec9(up9)  # (B, embed_dim, W, H)
        
        return dec9

class Conv_small(nn.Module):
    def __init__(self, embed_dim=1024):
        """
        Initializes the U-Net model. takes input (batch_size, embed_dim, width, height)
        
        Parameters:
        - embed_dim (int): Number of input channels.
        """
        super(Conv_small, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Encoder
        self.enc1 = DoubleConv_small(embed_dim, embed_dim)  # Retain embed_dim channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck1 = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.enc2 = DoubleConv_small(embed_dim // 2, embed_dim // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck2 = nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.enc3 = DoubleConv_small(embed_dim // 4, embed_dim // 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck3 = nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.enc4 = DoubleConv_small(embed_dim // 8, embed_dim // 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck4 = nn.Conv2d(embed_dim // 8, embed_dim // 16, kernel_size=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Bottleneck
        self.bottleneck = DoubleConv_small(embed_dim // 16, embed_dim // 16)
        
        # Decoder
        self.up6 = nn.ConvTranspose2d(embed_dim // 16, embed_dim // 16, kernel_size=2, stride=2)
        self.dec6 = DoubleConv_small(embed_dim // 16 + embed_dim // 8, embed_dim // 8)
        
        self.up7 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 8, kernel_size=2, stride=2)
        self.dec7 = DoubleConv_small(embed_dim // 8 + embed_dim // 4, embed_dim // 4)
        
        self.up8 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2)
        self.dec8 = DoubleConv_small(embed_dim // 4 + embed_dim // 2, embed_dim // 2)
        
        self.up9 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 2, kernel_size=2, stride=2)
        self.dec9 = DoubleConv_small(embed_dim // 2 + embed_dim, embed_dim)
        
    def forward(self, x):
        # Encoder Path
        enc1 = self.enc1(x)  # (B, embed_dim, W, H)
        pool1 = self.pool1(enc1)
        bottleneck1 = self.relu1(self.bottleneck1(pool1))
        
        enc2 = self.enc2(bottleneck1)  # (B, embed_dim//2, W/2, H/2)
        pool2 = self.pool2(enc2)
        bottleneck2 = self.relu2(self.bottleneck2(pool2))
        
        enc3 = self.enc3(bottleneck2)  # (B, embed_dim//4, W/4, H/4)
        pool3 = self.pool3(enc3)
        bottleneck3 = self.relu3(self.bottleneck3(pool3))
        
        enc4 = self.enc4(bottleneck3)  # (B, embed_dim//8, W/8, H/8)
        pool4 = self.pool4(enc4)
        bottleneck4 = self.relu4(self.bottleneck4(pool4))
        
        # Bottleneck
        bottleneck = self.bottleneck(bottleneck4)  # (B, embed_dim//16, W/16, H/16)
        
        return bottleneck
class DoubleConv(nn.Module):
    """
    Two consecutive standard convolutions (Conv2D + ReLU) as in the original U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, embed_dim=1024, count= 1):
        """
        Implements Depthwise Separable Convolution.
        """
        super(DepthwiseSeparableConv, self).__init__()
        layers = []
        for i in range(count):
            # Depthwise convolution (each input channel has its own kernel)
            layers.append(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False))
            # Pointwise convolution (reduces number of channels)
            layers.append(nn.Conv2d(embed_dim, 2, kernel_size=1, padding=0, bias=False))
            layers.append(nn.ReLU(inplace=True))
            # Pointwise convolution (restores number of channels)
            layers.append(nn.Conv2d(2, embed_dim, kernel_size=1, padding=0, bias=False))
        # Storing the layers in nn.Sequential
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)



class UNet(nn.Module):
    def __init__(self, embed_dim=1024):
        """
        Initializes the U-Net model. Takes input (batch_size, embed_dim, width, height)
        
        Parameters:
        - embed_dim (int): Number of input and output channels.
        """
        super(UNet, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Encoder
        self.enc1 = DoubleConv(embed_dim, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec6 = DoubleConv(1024, 512)
        
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec7 = DoubleConv(512, 256)
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec8 = DoubleConv(256, 128)
        
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec9 = DoubleConv(128, 64)
        
        # Final Convolution
        self.final_conv = nn.Conv2d(64, embed_dim, kernel_size=1)
        
    def forward(self, x):
        # Encoder Path
        enc1 = self.enc1(x)  # (B, 64, W, H)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)  # (B, 128, W/2, H/2)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)  # (B, 256, W/4, H/4)
        pool3 = self.pool3(enc3)
        
        enc4 = self.enc4(pool3)  # (B, 512, W/8, H/8)
        pool4 = self.pool4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)  # (B, 1024, W/16, H/16)
        
        # Decoder Path
        up6 = self.up6(bottleneck)  # (B, 512, W/8, H/8)
        up6 = torch.cat([up6, enc4], dim=1)  # (B, 512 + 512, W/8, H/8)
        dec6 = self.dec6(up6)  # (B, 512, W/8, H/8)
        
        up7 = self.up7(dec6)  # (B, 256, W/4, H/4)
        up7 = torch.cat([up7, enc3], dim=1)  # (B, 256 + 256, W/4, H/4)
        dec7 = self.dec7(up7)  # (B, 256, W/4, H/4)
        
        up8 = self.up8(dec7)  # (B, 128, W/2, H/2)
        up8 = torch.cat([up8, enc2], dim=1)  # (B, 128 + 128, W/2, H/2)
        dec8 = self.dec8(up8)  # (B, 128, W/2, H/2)
        
        up9 = self.up9(dec8)  # (B, 64, W, H)
        up9 = torch.cat([up9, enc1], dim=1)  # (B, 64 + 64, W, H)
        dec9 = self.dec9(up9)  # (B, 64, W, H)
        
        # Final Convolution
        out = self.final_conv(dec9)  # (B, embed_dim, W, H)
        
        return out

class ReverseUNetLSE(nn.Module):
    def __init__(self, gate = True, unet_type="small", embed_dim=1024, size_arg="small", n_classes=2, features=[16, 8, 4, 2], dropout=0.2):
        super(ReverseUNetLSE, self).__init__()
        self.C = embed_dim
        self.dropout = dropout
        

        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:    
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
   
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)

        if unet_type == "small":
            self.unet_part = UNet_small(embed_dim=embed_dim)
        elif unet_type == "normal":
            self.unet_part = UNet(embed_dim=embed_dim) # normal UNet
        elif unet_type == "conv":
            self.unet_part = Conv_small(embed_dim=embed_dim) # conv
            self.C = 64
            fc = [nn.Linear(64, 32), nn.ReLU()]
            if dropout:
                fc.append(nn.Dropout(0.25))
            if gate:
                attention_net = Attn_Net_Gated(L = 32, D = 16, dropout = dropout, n_classes = 1)
            else:    
                attention_net = Attn_Net(L = 32, D = 16, dropout = dropout, n_classes = 1)
            fc.append(attention_net)
            self.attention_net = nn.Sequential(*fc)
            self.classifiers = nn.Linear(32, n_classes)
        else: # using depthwiseSeparableConv
            self.unet_part = DepthwiseSeparableConv(embed_dim=embed_dim)

        self.n_classes = n_classes
        initialize_weights(self)

    def forward(self, h, coords, attention_only=False):
        device = h.device # Nxfeat_dim    
        N, C = h.shape
                        
        "input data with preserved spatial information"
         
        min_x= coords[:, 0].min()//512 # 512 because of the 256*256 patch size at *20 magnification 
        max_x= coords[:, 0].max()//512
        min_y= coords[:, 1].min()//512
        max_y= coords[:, 1].max()//512
                
        zz= torch.zeros((max_y-min_y+1, max_x-min_x+1,  C), device= device) # XxYxfeat_dim
        coords[:, :] //= 512
        cc_x= coords[:, 0]-min_x
        cc_y= coords[:, 1]-min_y
        zz[cc_y, cc_x]= h
        H, W, C = zz.shape
        # pad H and W dimension to be divisible by 16
        pad_h = (16 - H % 16) % 16  # This ensures we don't add extra padding if already divisible
        pad_w = (16 - W % 16) % 16  # Same for width

        # reshape zz to (1, C, W, H) for UNet

        zz = zz.permute(2, 1, 0).unsqueeze(0)  # 1xCxXxY
        # pad zz
        zz = F.pad(zz, (0, pad_h, 0, pad_w), mode='constant', value=0)

        # Apply UNet to zz

        h = self.unet_part(zz)  # 1xCxXxY

        # skip connections
        h = h + zz  # residual connection
        
        # reshape h to H, W, C
        h = h.squeeze(0).permute(2, 1, 0)

        
        "save zz and h_g_w for visualization"
        # torch.save(zz, 'zz.pt')
        # torch.save(h_g_w, 'h_g_w.pt')

        h = h[cc_y, cc_x] 
        h= h.view(-1, self.C)  # Nxfeat_dim
        
        """Local"""
        A, h = self.attention_net(h)  # NxK  
              
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        
        A = F.softmax(A, dim=1)  # softmax over N
        
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        return logits, Y_prob, Y_hat, A_raw, {}
    
class ReverseUNetLSE_normal(nn.Module):
    def __init__(self, embed_dim=1024, size_arg="small", n_classes=2, features=[16, 8, 4, 2], dropout=0.2):
        super(ReverseUNetLSE, self).__init__()
        
        self.dropout = dropout

        # Fully connected layers to reduce embedding dimension (1024 -> 256 -> 64 -> 32)
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 32)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Upsampling part of the Reverse U-Net (adjusted features [16, 8, 4, 2])
        in_channels = 32  # Start with 32 channels after the fully connected layers
        for feature in features:
            self.ups.append(nn.ConvTranspose2d(in_channels, feature, kernel_size=2, stride=2))
            self.ups.append(self.conv_block(feature, feature))
            in_channels = feature  # Set the in_channels for the next layer

        # Bottleneck
        self.bottleneck = self.conv_block(features[-1], features[-1])

        # Downsampling part of the Reverse U-Net
        for feature in reversed(features):
            self.downs.append(self.conv_block(feature * 2, feature))  # Expect doubled channels after concatenation
            self.downs.append(nn.Conv2d(feature, feature * 2, kernel_size=3, padding=1))

        # Final Conv2d layer to output logits
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, h, coords, attention_only=False):
        device = h.device  # N x feat_dim    
        N, C = h.shape

        "input data with preserved spatial information"
         
        min_x = coords[:, 0].min() // 512  # 512 because of the 256x256 patch size at 20x magnification 
        max_x = coords[:, 0].max() // 512
        min_y = coords[:, 1].min() // 512
        max_y = coords[:, 1].max() // 512
                
        zz = torch.zeros((max_y - min_y + 1, max_x - min_x + 1,  C), device=device)  # XxYxfeat_dim
        coords[:, :] //= 512
        cc_x = coords[:, 0] - min_x
        cc_y = coords[:, 1] - min_y
        zz[cc_y, cc_x] = h
        zz = zz.permute(2, 0, 1).unsqueeze(0)  # 1xCxXxY
        
        # Assign zz to x (1xCxXxY)
        x = zz

        # Reshape x to (batch_size * height * width, C) for FC layers
        batch_size, C, height, width = x.shape
        x = x.view(batch_size, C, -1).permute(0, 2, 1)  # Reshape to (batch_size, height*width, C)
        x = x.view(-1, C)  # Flatten to (batch_size * height * width, C)
        
        # Apply fully connected layers to reduce embedding dimension
        x = F.relu(self.fc1(x))  # Fully connected 1: C (1024) -> 256
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout
        x = F.relu(self.fc2(x))  # Fully connected 2: 256 -> 64
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout
        # x = F.relu(self.fc3(x))  # Fully connected 3: 64 -> 32
        # x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout

        # Reshape x back to (batch_size, 32, height, width)
        x = x.view(batch_size, height, width, 32).permute(0, 3, 1, 2)  # Back to 1x32xXxY

        skip_connections = []

        # Upward path (Decoder-like)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # ConvTranspose to upsample
            x = self.ups[idx + 1](x)  # Conv block
            x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout in bottleneck

        # Downward path (Encoder-like)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.downs), 2):
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
                
            x = torch.cat((skip_connection, x), dim=1)  # Concatenate skip connection with current layer
            x = self.downs[idx](x)  # Conv block
            x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout
            x = self.downs[idx + 1](x)  # Conv layer for downsampling

        # Final convolution to produce logits (raw output before softmax)
        logits = self.final_conv(x)

        # Combine height and width dimensions
        batch_size, channels, height, width = logits.shape
        logits = logits.view(batch_size, channels, -1)  # Reshape to (batch_size, 2, height * width)

        # Apply Log-Sum-Exp pooling over the combined dimension (height * width)
        logits = torch.logsumexp(logits, dim=2)  # Perform LSE pooling over spatial dimensions

        # Apply softmax over the 2 output channels to get Y_prob
        Y_prob = F.softmax(logits, dim=1)

        # Predicted class (Y_hat) - taking the argmax over the channel dimension
        Y_hat = torch.argmax(Y_prob, dim=1)

        # Return the required outputs
        return logits, Y_prob, Y_hat, logits, {}  # logits returned as A_raw as well

    def conv_block(self, embed_dim, n_classes):
        return nn.Sequential(
            nn.Conv2d(embed_dim, n_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(inplace=True),
        )


# Testing the modified Reverse U-Net with embedding dimension reduction applied to x (1024 -> 32)
if __name__ == "__main__":
    model = ReverseUNetLSE(embed_dim=1024, n_classes=2, dropout=0.3)
    h = torch.randn((1000, 1024))  # Example input features
    coords = torch.randint(0, 512 * 20, (1000, 2))  # Example coordinates
    logits, Y_prob, Y_hat, A_raw, _ = model(h, coords)
    print(f"logits: {logits.shape}, Y_prob: {Y_prob.shape}, Y_hat: {Y_hat.shape}, A_raw: {A_raw.shape}")
