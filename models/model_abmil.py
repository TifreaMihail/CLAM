import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
import math

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
        

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class ABMIL(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, embed_dim=1024):
        super(ABMIL, self).__init__()
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
        self.n_classes = n_classes
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h, coords= None, attention_only=False):
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


"""Combined local & global"""
    
class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = int(num_features * expansion_factor)
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        return x

def window_partition(input, window_size=(7, 7)):
    """ Window partition function.
    Args:
        input (torch.Tensor): Input tensor of the shape [H, W, C].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)
    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    H, W, C = input.shape    
    # Unfold input
    windows = input.view(H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    # Permute and reshape to [windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 1, 3, 4).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, original_size, window_size= (7, 7)):
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [original_size[0], original_size[1], C].
    """
    # Get height and width
    H, W = original_size
    # Fold grid tensor
    output = windows.view(H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 2, 1, 3, 4).contiguous().view(H, W, -1)
    return output


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    win_size: window size for global attention
"""
class GABMIL(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, win_size= 2):
        super(GABMIL, self).__init__()
        self.win_size= win_size
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
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
        self.n_classes = n_classes
        
        self.mlp1 = MLP(num_features= self.win_size*self.win_size, expansion_factor= 1, dropout= 0.25)  
        
        self.window_partition = window_partition
        self.window_reverse = window_reverse
    
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.mlp1 = self.mlp1.to(device)
        
    def forward(self, h, coords= None, attention_only=False):
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
    
        """Global"""
        X, Y, C = zz.shape
        _S_x = int(np.ceil(X/self.win_size1))
        _S_y = int(np.ceil(Y/self.win_size1)) 
        add_length_s_x = _S_x*self.win_size1 - X
        add_length_s_y = _S_y*self.win_size1- Y
        zz_x = torch.zeros((add_length_s_x, Y, C), device= device)
        zz = torch.cat([zz, zz_x],dim = 0) # (X+x)xYxfeat_dim  
        zz_y = torch.zeros((X+add_length_s_x ,add_length_s_y, C), device= device)
        zz = torch.cat([zz, zz_y],dim = 1) # (X+x)x(Y+y)xfeat_dim  
                    
        H, W, C = zz.shape

        "Window"
        windows= self.window_partition(zz, (self.win_size1, self.win_size1)) # [windows,  window_size[0], window_size[1], feat_dim]
        windows = windows.view(-1, self.win_size1*self.win_size1, C) # [windows, window_size[0]*window_size[1], feat_dim]
        windows = windows.permute(0, 2, 1) #  [windows, feat_dim, window_size[0]*window_size[1]]
        h_g_w = windows
        h_g_w = self.mlp1(h_g_w) # [windows, feat_dim, window_size[0]*window_size[1]]
        
        h_g_w = h_g_w.permute(0, 2, 1) # [windows, window_size[0]*window_size[1], feat_dim]
        h_g_w = h_g_w.view(-1, self.win_size1, self.win_size1, C) # [windows,  window_size[0], window_size[1], feat_dim]
        h_g_w = self.window_reverse(h_g_w, (H, W), (self.win_size1, self.win_size1)) # [original_size[0], original_size[1], feat_dim]
         
        """Residual Connection"""
        h= zz + h_g_w
        h = h[cc_y, cc_x] 
        h= h.view(-1, C)
        
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
    
