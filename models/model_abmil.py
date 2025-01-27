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
    


def spatialy_rearange(coords, h):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, C = h.shape

    min_x = coords[:, 0].min() // 512
    max_x = coords[:, 0].max() // 512
    min_y = coords[:, 1].min() // 512
    max_y = coords[:, 1].max() // 512

    zz = torch.zeros((max_y - min_y + 1, max_x - min_x + 1, C), device=device)

    cc_x = coords[:, 0] // 512 - min_x
    cc_y = coords[:, 1] // 512 - min_y

    zz[cc_y, cc_x] = h

    return zz, cc_x, cc_y

def padd_for_windows(zz, win_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, C = zz.shape

    _S_x = int(np.ceil(X / win_size))
    _S_y = int(np.ceil(Y / win_size))

    zz_x = torch.zeros((_S_x * win_size - X, Y, C), device=device)
    zz_y = torch.zeros((_S_x * win_size, _S_y * win_size - Y, C), device=device)

    zz = torch.cat([zz, zz_x], dim=0)
    zz = torch.cat([zz, zz_y], dim=1)
    return zz

def spatially_rearange_and_respahe_for_windows(coords, h, win_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zz, cc_x, cc_y = spatialy_rearange(coords, h)
    zz = padd_for_windows(zz, win_size)
    return zz, cc_x, cc_y

def undo_rearangement(h, cc_x, cc_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h = h[cc_y, cc_x]
    return h

def apply_SIMM(h, coords, win_size, use_block, use_grid, mlp1, mlp2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zz, cc_x, cc_y = spatially_rearange_and_respahe_for_windows(coords, h, win_size)

    if use_block:
        windows = window_partition(zz, (win_size, win_size))
        h_g_w = mlp1(windows)
        h_g_w = window_reverse(h_g_w, zz.shape, (win_size, win_size))
    else:
        h_g_w = zz

    if use_grid:
        windows = grid_partition(h_g_w, (win_size, win_size))
        h_g_w = mlp2(windows)
        h_g_w = grid_reverse(h_g_w, zz.shape, (win_size, win_size))

    h = undo_rearangement(h_g_w, cc_x, cc_y)
    return h

class Attn_Net_Gated_v2(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1, win_size=2, use_layer_norm=False, use_weight_norm=False, use_block=True, use_grid=False):
        super(Attn_Net_Gated_v2, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.win_size = win_size
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.use_block = use_block
        self.use_grid = use_grid

        self.mlp1 = MLP(num_features=win_size * win_size, expansion_factor=1, dropout=0.25, use_layer_norm=False, use_weight_norm=False).to(device)
        self.mlp2 = MLP(num_features=win_size * win_size, expansion_factor=1, dropout=0.25, use_layer_norm=False, use_weight_norm=False).to(device)

        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        ).to(device)
        
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        ).to(device)

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_c = nn.Linear(D, n_classes).to(device)

    def forward(self, x, coords):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, coords = x.to(device), coords.to(device)

        a = self.attention_a(x)
        a = apply_SIMM(a, coords, self.win_size, self.use_block, self.use_grid, self.mlp1, self.mlp2)

        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
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


class SelfAttentionModule(nn.Module):
    def __init__(self, feat_dim = 1024, num_heads = 4):
        """
        Initializes the SelfAttentionModule.
        
        Args:
            feat_dim (int): Dimensionality of features (input size for attention).
            num_heads (int): Number of attention heads.
        """
        super(SelfAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, x):
        """
        Forward pass for the module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (windows, feat_dim, window_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (windows, feat_dim, window_size).
        """
        # Permute input to (windows, window_size, feat_dim)
        x = x.permute(0, 2, 1)
        
        # Apply self-attention
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Permute back to (windows, feat_dim, window_size)
        attn_output = attn_output.permute(0, 2, 1)
        return attn_output

"""Combined local & global"""
    
class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout, use_layer_norm=False, use_weight_norm= True):
        super().__init__()
        num_hidden = int(num_features * expansion_factor)
        self.fc1 = nn.Linear(num_features, num_hidden)
        if use_weight_norm:
            self.fc1 = nn.utils.weight_norm(self.fc1)
        self.dropout1 = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(num_features)  # LayerNorm applied on the input features

    def forward(self, x):
        if self.use_layer_norm:
            x = self.layer_norm(x)  # Applying LayerNorm before other operations
        x = self.dropout1(F.relu(self.fc1(x)))
        return x

def window_partition(input, window_size=(7, 7)):
    """ Window partition function.
    Args:
        input (torch.Tensor): Input tensor of the shape [H, W, C].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)
    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [windows, window_size[0], window_size[1], 
        C].
    """
    # Get size of input
    H, W, C = input.shape    
    # Unfold input
    windows = input.view(H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    # Permute and reshape to [windows, window_size[0], window_size[1], channels]

    windows = windows.permute(0, 2, 1, 3, 4).contiguous().view(-1, window_size[0], window_size[1], C)

    windows = windows.view(-1, window_size[0]*window_size[1], C)
    windows = windows.permute(0, 2, 1)

    return windows

def grid_partition(input, window_size=(7, 7)):
    """ Grid partition function.
    Args:
        input (torch.Tensor): Input tensor of the shape [H, W, C].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)
    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [windows, window_size[0], window_size[1],
         C].
    """
    # Get size of input
    H, W, C = input.shape    
    # Unfold input
    windows = input.view(window_size[0], H // window_size[0], window_size[1], W // window_size[1], C)
    # Permute and reshape to [windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 1, 3, 4).contiguous().view(window_size[0], window_size[1], -1, C)

    windows = windows.view(window_size[0]*window_size[1], -1, C)
    # windows = windows.permute(0, 2, 1)
    # windows = windows.permute(2, 1, 0)
    windows = windows.permute(1, 2, 0)

    return windows

def grid_reverse(windows, original_size, window_size= (7, 7)):
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [original_size[0], original_size[1], C].
    """
    # Get height and width
    H, W, C = original_size

    windows = windows.permute(2, 0, 1) # [windows, window_size[0]*window_size[1], feat_dim]
    windows = windows.view(window_size[0], window_size[1], -1, C) # [windows,  window_size[0], window_size[1], feat_dim]
    # Fold grid tensor
    output = windows.view(window_size[0], window_size[1], H // window_size[0], W // window_size[1], -1)
    output = output.permute(0, 2, 1, 3, 4).contiguous().view(H, W, -1)
    return output

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
    H, W, C = original_size

    windows = windows.permute(0, 2, 1) # [windows, window_size[0]*window_size[1], feat_dim]
    windows = windows.view(-1, window_size[0], window_size[1], C) # [windows,  window_size[0], window_size[1], feat_dim]
    # Fold grid tensor
    output = windows.view(H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 2, 1, 3, 4).contiguous().view(H, W, -1)
    return output


"""
THIS ONE USES TRANSFOrmeR ATTENTION
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    win_size: window size for global attention
"""
class GABMIL_ATTN(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, win_size= 2, embed_dim=
    1024, use_grid= True, use_skip= True, use_norm=False, use_block= True, use_weight_norm= True):
        super(GABMIL_ATTN, self).__init__()
        self.win_size= win_size
        self.use_grid= use_grid
        self.use_skip= use_skip
        self.use_norm= use_norm
        self.use_block= use_block
        self.use_weight_norm= use_weight_norm
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
        
        self.mlp1 = MLP(num_features= self.win_size*self.win_size, expansion_factor= 1, dropout= 0.25, use_layer_norm = use_norm, use_weight_norm= use_weight_norm)
        self.mlp2 = MLP(num_features= self.win_size*self.win_size, expansion_factor= 1, dropout= 0.25, use_layer_norm = use_norm, use_weight_norm= use_weight_norm)    
        
        self.shared_atttention1 =  SelfAttentionModule()
        self.shared_atttention2 = SelfAttentionModule()

        self.window_partition = window_partition
        self.window_reverse = window_reverse
    
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.mlp1 = self.mlp1.to(device)
        
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
        zz[cc_y, cc_x]= h # maybe zz[cc_x, cc_y]?
    
        """Global"""
        X, Y, C = zz.shape
        _S_x = int(np.ceil(X/self.win_size))
        _S_y = int(np.ceil(Y/self.win_size)) 
        add_length_s_x = _S_x*self.win_size - X
        add_length_s_y = _S_y*self.win_size- Y
        zz_x = torch.zeros((add_length_s_x, Y, C), device= device)
        zz = torch.cat([zz, zz_x],dim = 0) # (X+x)xYxfeat_dim  
        zz_y = torch.zeros((X+add_length_s_x ,add_length_s_y, C), device= device)
        zz = torch.cat([zz, zz_y],dim = 1) # (X+x)x(Y+y)xfeat_dim  
                    
        H, W, C = zz.shape

        "Window"

        # Block attention:
        if self.use_block:
            windows= window_partition(zz, (self.win_size, self.win_size))

            h_g_w = windows
            # h_g_w = self.mlp1(h_g_w)
            h_g_w = self.shared_atttention1(h_g_w)

            h_g_w = window_reverse(h_g_w, (H, W, C), (self.win_size, self.win_size))
        else:
            h_g_w= zz

        # grid attention:
        #  [windows, feat_dim, window_size[0]*window_size[1]]
        if self.use_grid:
            windows= grid_partition(h_g_w, (self.win_size, self.win_size))
            
            h_g_w = windows
            # h_g_w = self.mlp2(h_g_w)
            h_g_w = self.shared_atttention2(h_g_w)

            h_g_w = grid_reverse(h_g_w, (H, W, C), (self.win_size, self.win_size))

         
        """Residual Connection"""
        if self.use_skip:
            h= zz + h_g_w
        else:
            h= h_g_w
        
        "save zz and h_g_w for visualization"
        # torch.save(zz, 'zz.pt')
        # torch.save(h_g_w, 'h_g_w.pt')

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
    
    """
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes
    win_size: window size for global attention
"""
class GABMIL(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, win_size= 2, embed_dim=
    1024, use_grid= False, use_skip= True, use_norm=False, use_block= True, use_weight_norm= False):
        super(GABMIL, self).__init__()
        self.win_size= win_size
        self.use_grid= use_grid
        self.use_skip= use_skip
        self.use_norm= use_norm
        self.use_block= use_block
        self.use_weight_norm= use_weight_norm
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
        
        self.mlp1 = MLP(num_features= self.win_size*self.win_size, expansion_factor= 1, dropout= 0.25, use_layer_norm = use_norm, use_weight_norm= use_weight_norm)
        self.mlp2 = MLP(num_features= self.win_size*self.win_size, expansion_factor= 1, dropout= 0.25, use_layer_norm = use_norm, use_weight_norm= use_weight_norm)    
        
        self.window_partition = window_partition
        self.window_reverse = window_reverse
    
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.mlp1 = self.mlp1.to(device)
        
    def forward(self, h, coords, attention_only=False):
        device = h.device # Nxfeat_dim    
        N, C = h.shape
                        
        "input data with preserved spatial information"
         
        min_x= coords[:, 0].min()//512 # 512 because of the 256*256 patch size at *20 magnification 
        max_x= coords[:, 0].max()//512
        min_y= coords[:, 1].min()//512
        max_y= coords[:, 1].max()//512
                
        zz= torch.zeros((max_y-min_y+1, max_x-min_x+1,  C), device= device)
        coords[:, :] //= 512
        cc_x= coords[:, 0]-min_x
        cc_y= coords[:, 1]-min_y
        zz[cc_y, cc_x]= h
    
        """Global"""
        X, Y, C = zz.shape
        _S_x = int(np.ceil(X/self.win_size))
        _S_y = int(np.ceil(Y/self.win_size))
        add_length_s_x = _S_x*self.win_size - X
        add_length_s_y = _S_y*self.win_size- Y
        zz_x = torch.zeros((add_length_s_x, Y, C), device= device)
        zz = torch.cat([zz, zz_x],dim = 0) # (X+x)xYxfeat_dim  
        zz_y = torch.zeros((X+add_length_s_x ,add_length_s_y, C), device= device)
        zz = torch.cat([zz, zz_y],dim = 1) # (X+x)x(Y+y)xfeat_dim  
                    
        H, W, C = zz.shape

        "Window"

        # Block attention:
        if self.use_block:
            windows= window_partition(zz, (self.win_size, self.win_size))

            h_g_w = windows
            h_g_w = self.mlp1(h_g_w)

            h_g_w = window_reverse(h_g_w, (H, W, C), (self.win_size, self.win_size))
        else:
            h_g_w= zz

        # grid attention:
        #  [windows, feat_dim, window_size[0]*window_size[1]]
        if self.use_grid:
            windows= grid_partition(h_g_w, (self.win_size, self.win_size))
            
            h_g_w = windows
            h_g_w = self.mlp2(h_g_w)

            h_g_w = grid_reverse(h_g_w, (H, W, C), (self.win_size, self.win_size))

         
        """Residual Connection"""
        if self.use_skip:
            h= zz + h_g_w
        else:
            h= h_g_w
        
        "save zz and h_g_w for visualization"
        # torch.save(zz, 'zz.pt')
        # torch.save(h_g_w, 'h_g_w.pt')

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

class GABMIL_ws1(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, win_size= 2, embed_dim=
    1024, use_grid= False, use_skip= True, use_norm=False, use_block= True, use_weight_norm= False):
        super(GABMIL, self).__init__()
        self.win_size= win_size
        self.use_grid= use_grid
        self.use_skip= use_skip
        self.use_norm= use_norm
        self.use_block= use_block
        self.use_weight_norm= use_weight_norm
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
        
        self.mlp1 = MLP(num_features= self.win_size*self.win_size, expansion_factor= 1, dropout= 0.25, use_layer_norm = use_norm, use_weight_norm= use_weight_norm)
        self.mlp2 = MLP(num_features= self.win_size*self.win_size, expansion_factor= 1, dropout= 0.25, use_layer_norm = use_norm, use_weight_norm= use_weight_norm)    
        
        self.window_partition = window_partition
        self.window_reverse = window_reverse
    
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.mlp1 = self.mlp1.to(device)
        self.mlp2 = self.mlp2.to(device)
        
    def forward(self, h, coords, attention_only=False):
        device = h.device # Nxfeat_dim    
        N, C = h.shape
                        
        "input data with preserved spatial information"
         
        min_x= coords[:, 0].min()//512 # 512 because of the 256*256 patch size at *20 magnification 
        max_x= coords[:, 0].max()//512
        min_y= coords[:, 1].min()//512
        max_y= coords[:, 1].max()//512
                
        zz= torch.zeros((max_y-min_y+1, max_x-min_x+1,  C), device= device)
        coords[:, :] //= 512
        cc_x= coords[:, 0]-min_x
        cc_y= coords[:, 1]-min_y
        zz[cc_y, cc_x]= h
    
        """Global"""
        X, Y, C = zz.shape
        _S_x = int(np.ceil(X/self.win_size))
        _S_y = int(np.ceil(Y/self.win_size))
        add_length_s_x = _S_x*self.win_size - X
        add_length_s_y = _S_y*self.win_size- Y
        zz_x = torch.zeros((add_length_s_x, Y, C), device= device)
        zz = torch.cat([zz, zz_x],dim = 0) # (X+x)xYxfeat_dim  
        zz_y = torch.zeros((X+add_length_s_x ,add_length_s_y, C), device= device)
        zz = torch.cat([zz, zz_y],dim = 1) # (X+x)x(Y+y)xfeat_dim  
                    
        H, W, C = zz.shape

        "Window"

        # Block attention:
        if self.use_block:
            windows= window_partition(zz, (self.win_size, self.win_size))

            h_g_w = windows
            h_g_w = self.mlp1(h_g_w)

            h_g_w = window_reverse(h_g_w, (H, W, C), (self.win_size, self.win_size))
        else:
            h_g_w= zz

        # grid attention:
        #  [windows, feat_dim, window_size[0]*window_size[1]]
        if self.use_grid:
            windows= grid_partition(h_g_w, (self.win_size, self.win_size))
            
            h_g_w = windows
            h_g_w = self.mlp2(h_g_w)

            h_g_w = grid_reverse(h_g_w, (H, W, C), (self.win_size, self.win_size))

         
        """Residual Connection"""
        if self.use_skip:
            h= zz + h_g_w
        else:
            h= h_g_w
        
        "save zz and h_g_w for visualization"
        # torch.save(zz, 'zz.pt')
        # torch.save(h_g_w, 'h_g_w.pt')

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
    
class GABMIL_v2(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, win_size= 2, embed_dim=
    1024, use_grid= True, use_skip= True, use_norm=False, use_block= True, use_weight_norm= True):
        super(GABMIL_v2, self).__init__()
        self.win_size= win_size
        self.use_grid= use_grid
        self.use_skip= use_skip
        self.use_norm= use_norm
        self.use_block= use_block
        self.use_weight_norm= use_weight_norm
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        
        self.fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            self.fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*self.fc)

        self.attention_net = Attn_Net_Gated_v2(L = size[1], D = size[2], dropout = dropout, n_classes = 1, win_size= self.win_size, use_layer_norm = self.use_norm, use_weight_norm= self.use_weight_norm, use_block= self.use_block, use_grid= self.use_grid)
        
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes
        
        self.mlp1 = MLP(num_features= self.win_size*self.win_size, expansion_factor= 1, dropout= 0.25, use_layer_norm = use_norm, use_weight_norm= use_weight_norm)
        self.mlp2 = MLP(num_features= self.win_size*self.win_size, expansion_factor= 1, dropout= 0.25, use_layer_norm = use_norm, use_weight_norm= use_weight_norm)    
        
        self.window_partition = window_partition
        self.window_reverse = window_reverse
    
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.mlp1 = self.mlp1.to(device)
        
    def forward(self, h, coords, attention_only=False):
        device = h.device # Nxfeat_dim    
        N, C = h.shape
        
        h=self.fc(h)

        """Local"""
        A, h = self.attention_net(h, coords)  # NxK  
              
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
    
class PseudoConvolution(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dropout_prob=0.0):
        super(PseudoConvolution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.dropout_prob = dropout_prob
        
        # Fully connected layer that processes each patch (flattened)
        self.fc = nn.Linear(in_channels * kernel_size * kernel_size, in_channels)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Input shape: (N, C, H, W)
        N, C, H, W = x.shape
        
        # Step 1: Extract patches using unfold
        patches = nn.functional.unfold(
            x, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding
        )  # Shape: (N, C * kernel_size * kernel_size, L)
        
        L = patches.shape[-1]  # Number of patches
        
        # Step 2: Transpose and reshape to apply FC layer
        patches = patches.transpose(1, 2)  # Shape: (N, L, C * kernel_size * kernel_size)
        patches = patches.reshape(-1, C * self.kernel_size * self.kernel_size)  # Shape: (N * L, C * K * K)
        
        # Step 3: Apply the fully connected layer
        processed_patches = self.fc(patches)  # Shape: (N * L, C)
        
        # Step 4: Apply dropout
        if self.dropout_prob > 0.0:
            processed_patches = self.dropout(processed_patches)
        
        # Step 5: Reshape back to patch layout
        processed_patches = processed_patches.view(N, L, C)  # Shape: (N, L, C)
        processed_patches = processed_patches.transpose(1, 2)  # Shape: (N, C, L)
        
        # Step 6: Fold back to spatial dimensions
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = processed_patches.view(N, C, H_out, W_out)
        
        return output

    
class GABMIL_CONV(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, win_size= 2, embed_dim=
    1024, use_grid= True, use_skip= True, use_norm=False, use_block= True, use_weight_norm= True):
        super(GABMIL_CONV, self).__init__()
        self.win_size= win_size
        self.use_grid= use_grid
        self.use_skip= use_skip
        self.use_norm= use_norm
        self.use_block= use_block
        self.use_weight_norm= use_weight_norm
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
        
        # create a convolutional layer

        # self.conv_layer = nn.Conv2d(
        #     in_channels=1024,  # Input channels
        #     out_channels=1024,  # Output channels
        #     kernel_size=win_size,  # Kernel size
        #     stride=1,  # Stride
        #     padding=(win_size-1)//2,  # keep dimensions
        #     bias=True  # No bias for simplicity
        # )
        if dropout:
            dropout_prob = 0.36
        else:
            dropout_prob = 0.0
        self.conv_layer = PseudoConvolution(in_channels=1024, kernel_size=win_size, stride=1, padding=(win_size-1)//2, dropout_prob = dropout_prob)

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.mlp1 = self.mlp1.to(device)
        
    def forward(self, h, coords, attention_only=False):
        device = h.device # Nxfeat_dim    
        N, C = h.shape
                        
        "input data with preserved spatial information"
         
        min_x= coords[:, 0].min()//512 # 512 because of the 256*256 patch size at *20 magnification 
        max_x= coords[:, 0].max()//512
        min_y= coords[:, 1].min()//512
        max_y= coords[:, 1].max()//512
                
        zz= torch.zeros((max_y-min_y+1, max_x-min_x+1,  C), device= device)
        coords[:, :] //= 512
        cc_x= coords[:, 0]-min_x
        cc_y= coords[:, 1]-min_y
        zz[cc_y, cc_x]= h
    
        """Global"""
        X, Y, C = zz.shape
        _S_x = int(np.ceil(X/self.win_size))
        _S_y = int(np.ceil(Y/self.win_size))
        add_length_s_x = _S_x*self.win_size - X
        add_length_s_y = _S_y*self.win_size- Y
        zz_x = torch.zeros((add_length_s_x, Y, C), device= device)
        zz = torch.cat([zz, zz_x],dim = 0) # (X+x)xYxfeat_dim  
        zz_y = torch.zeros((X+add_length_s_x ,add_length_s_y, C), device= device)
        zz = torch.cat([zz, zz_y],dim = 1) # (X+x)x(Y+y)xfeat_dim  
                    
        H, W, C = zz.shape

        "Window"

        input_tensor = zz.permute(2, 0, 1)  # From (H, W, C) -> (C, H, W)

        # Step 2: Add batch dimension to make it (N, C, H, W)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension, now (1, C, H, W)

        
        # Step 3: Apply convolution
        output_tensor = self.conv_layer(input_tensor)

        h_g_w = output_tensor.squeeze(0).permute(1, 2, 0)  # From (1, C, H, W) -> (H, W, C)

         
        """Residual Connection"""
        if self.use_skip:
            h= zz + h_g_w
        else:
            h= h_g_w
        
        "save zz and h_g_w for visualization"
        # torch.save(zz, 'zz.pt')
        # torch.save(h_g_w, 'h_g_w.pt')

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
    
