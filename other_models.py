import torch
import torch.nn as nn
import torch.nn.functional as F
# from models import ReconNetwork, MLP, MLP_Lipschitz



def double_convLeon(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


# adapted from https://github.com/usuyama/pytorch-unet/tree/master
class UNetLeon(nn.Module):

    def __init__(self, n_channels, base_channel):
        super().__init__()

        self.dconv_down1 = double_convLeon(n_channels, base_channel)
        self.dconv_down2 = double_convLeon(base_channel, base_channel * 2)
        self.dconv_down3 = double_convLeon(base_channel * 2, base_channel * 4)
        self.dconv_down4 = double_convLeon(base_channel * 4, base_channel * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.dconv_up3 = double_convLeon(base_channel * 12, base_channel * 4)
        self.dconv_up2 = double_convLeon(base_channel * 6, base_channel * 2)
        self.dconv_up1 = double_convLeon(base_channel * 3, base_channel)

        self.conv_last = nn.Conv2d(base_channel, n_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)  # 256x256

        x = self.maxpool(conv1)  # 128x128
        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)  # 64x64
        conv3 = self.dconv_down3(x)

        x = self.maxpool(conv3)  # 32x32
        bootle = self.dconv_down4(x)

        x = self.upsample(bootle)  # 64x64
        x = torch.cat([x, conv3], dim=1)
        up1 = self.dconv_up3(x)

        x = self.upsample(up1)  # 128x128
        x = torch.cat([x, conv2], dim=1)
        up2 = self.dconv_up2(x)

        x = self.upsample(up2)  # 256x256
        x = torch.cat([x, conv1], dim=1)
        up3 = self.dconv_up1(x)

        out = self.conv_last(up3)

        return out






class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], args=None):
        super(UNet, self).__init__()
        self.exp_mode = args.exp_mode
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.R_linear_module = get_R_linear_module(args)

        # Encoder: cada etapa reduce la resolución y aumenta las características
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder: configuración según el modo de fusión de las skip connections
        rev_features = list(reversed(features))
        for i, feature in enumerate(rev_features):
            in_ch = features[-1] * 2 if i == 0 else rev_features[i - 1]
            up_conv = nn.ConvTranspose2d(in_ch, feature, kernel_size=2, stride=2)

            if self.exp_mode == "Residual":
                conv_block = nn.Sequential(
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            else:
                conv_block = nn.Sequential(
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            self.ups.append(up_conv)
            self.ups.append(conv_block)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if 'Identity' in self.exp_mode:
            self.initialize_identity()

    def initialize_identity(self):
        # Inicializa todas las capas convolucionales para que actúen como la identidad cuando sea posible.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Solo si el número de canales coincide y el kernel es impar.
                if m.in_channels == m.out_channels and m.kernel_size[0] % 2 == 1:
                    init_identity_conv(m)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            if self.exp_mode == "Residual":
                x = x + skip_connection
            else:
                x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](x)

        return self.R_linear_module(self.final_conv(x).reshape(x.shape[0], -1))  # Aplicamos la red lineal después de la convolución


class IndiSelfAttention(nn.Module):
    """
    A Self-Attention module implementing multi-headed attention mechanism.

    This module applies a multi-head attention mechanism on the input feature map,
    followed by layer normalization and a feedforward neural network.

    Attributes:
        channels (int): The number of channels in the input.
        size (int): The size of each attention head.
    """
    def __init__(self, channels, size):
        super(IndiSelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Normal convolution block, with 2d convolution -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    maxpool reduce size by half -> 2*DoubleConv -> Embedding layer
    
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear( # linear projection to bring the time embedding to the proper dimension
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # projection
        return x + emb


class Up(nn.Module):
    """
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class IndiUnet(nn.Module):
    def __init__(self, c_in=1, c_out=1, image_size=64, time_dim=256, device='cuda', latent=False, true_img_size=64, num_classes=None, args=None):
        super(IndiUnet, self).__init__()

        # Encoder
        self.true_img_size = true_img_size
        self.image_size = image_size
        self.time_dim = time_dim
        self.device = device
        self.inc = DoubleConv(c_in, self.image_size) # Wrap-up for 2 Conv Layers
        self.down1 = Down(self.image_size, self.image_size*2) # input and output channels
        # self.sa1 = IndiSelfAttention(self.image_size*2,int( self.true_img_size/2)) # 1st is channel dim, 2nd current image resolution
        self.down2 = Down(self.image_size*2, self.image_size*4)
        # self.sa2 = IndiSelfAttention(self.image_size*4, int(self.true_img_size/4))
        self.down3 = Down(self.image_size*4, self.image_size*4)
        # self.sa3 = IndiSelfAttention(self.image_size*4, int(self.true_img_size/8))
        
        # Bootleneck
        self.bot1 = DoubleConv(self.image_size*4, self.image_size*8)
        self.bot2 = DoubleConv(self.image_size*8, self.image_size*8)
        self.bot3 = DoubleConv(self.image_size*8, self.image_size*4)
        
        # Decoder: reverse of encoder
        self.up1 = Up(self.image_size*8, self.image_size*2)
        # self.sa4 = IndiSelfAttention(self.image_size*2, int(self.true_img_size/4))
        self.up2 = Up(self.image_size*4, self.image_size)
        # self.sa5 = IndiSelfAttention(self.image_size, int(self.true_img_size/2))
        self.up3 = Up(self.image_size*2, self.image_size)
        # self.sa6 = IndiSelfAttention(self.image_size, self.true_img_size)
        self.outc = nn.Conv2d(self.image_size, c_out, kernel_size=1) # projecting back to the output channel dimensions
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        if latent == True:
            self.latent = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256)).to(device)    
            
        self.R_linear_module = get_R_linear_module(args)
  
    def pos_encoding(self, t, channels):
        """
        Input noised images and the timesteps. The timesteps will only be
        a tensor with the integer timesteps values in it
        """
        inv_freq = 1.0 /  (
            10000 
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc 

    def forward(self, x, lab=None, t=None):
        # Pass the source image through the encoder network
        t = torch.tensor(t).unsqueeze(-1).type(torch.float).to(self.device)
        t = self.pos_encoding(t, self.time_dim) # Encoding timesteps is HERE, we provide the dimension we want to encode

        
        if lab is not None:
            t += self.label_emb(lab)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        # x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t) # We note that upsampling box that in the skip connections from encoder 
        # x = self.sa4(x)
        x = self.up2(x, x2, t)
        # x = self.sa5(x)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)

        return self.R_linear_module(output.reshape(x.shape[0], -1))  # Aplicamos la red lineal después de la convolución