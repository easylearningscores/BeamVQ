import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import scipy.io
import torch.utils.data
from backbone.FNO_2d import *
from backbone.UNet import U_net
from backbone.ResNet import ResNet
from backbone.CNO import CNO
from backbone.FNO_2d import FNO2d
from backbone.UNO_2d import UNO
from backbone.LSM_2d import Model as LSM
from backbone.ConvLSTM import CLSTM as convlstm

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5, top_k=3):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._top_k = top_k

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        top_k_indices = torch.topk(distances, self._top_k, dim=1, largest=False)[1]
        top_k_encodings = []
        top_k_quantized = []
        for i in range(self._top_k):
            encoding_indices = top_k_indices[:, i].unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            top_k_encodings.append(encodings)

            quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
            top_k_quantized.append(quantized)

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(top_k_quantized[0].detach(), inputs)  
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (top_k_quantized[0] - inputs).detach()  
        top_k_quantized = [inputs + (q - inputs).detach() for q in top_k_quantized] 

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, top_k_quantized


    def lookup(self, x):
            embeddings = F.embedding(x, self._embedding)
            return embeddings
    
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Projection(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Projection, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)
        
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)
        
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs) 
        x = F.relu(x)
        x = self._conv_2(x) 
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)


def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker,
                          stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class DST(nn.Module):
    def __init__(self,
                 in_channel=1,
                 num_hiddens=128,
                 res_layers=2,
                 res_units=32,
                 embedding_nums=1024,  # K
                 embedding_dim=128,  # D
                 top_k=3,
                 commitment_cost=0.25,
                 decay=0.99):
        super(DST, self).__init__()
        self.in_channel = in_channel
        self.num_hiddens = num_hiddens
        self.res_layers = res_layers
        self.res_units = res_units
        self.embedding_dim = embedding_dim
        self.embedding_nums = embedding_nums
        self.top_k = top_k
        self.decay = decay
        self.commitment_cost = commitment_cost
        self._projection = Projection(in_channel, num_hiddens,
                                res_layers, res_units)  #
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        # code book
        self._vq_vae = VectorQuantizerEMA(num_embeddings=self.embedding_nums,
                                  embedding_dim=self.embedding_dim,
                                  commitment_cost=self.commitment_cost,
                                  decay=self.decay,
                                  top_k=self.top_k)


        self._decoder = Decoder(in_channels=self.embedding_dim,
                        num_hiddens=self.num_hiddens,
                        num_residual_layers=self.res_layers,
                        num_residual_hiddens=self.res_units,
                        out_channels=self.in_channel)
    def forward(self, x):
        '''
        Process input x through a neural network model.
        The shape of the input is [B, C, W, H], i.e. [batch size, number of channels, width, height].
        The input x is projected into the hidden unit space by means of the _projection function and the output has the shape [B, hidden_units, W//4, H//4].
        Convolution is then performed before applying VQ-VAE, and z is transformed into coded form.
        After VQ-VAE processing, the output consists of the loss, quantized encoding, perplexity, and other possible outputs (ignored here with _).
        The quantized code is reconstructed from the original input x by the decoder.
        The return value consists of the loss, the reconstructed x, and the perplexity.
        quantized -> embedding, quantized is equivalent to the encoder output in videoGPT.
        '''
        z = self._projection(x) 
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

    def get_embedding(self, x):
        return self._pre_vq_conv(self._encoder(x))

    def get_quantization(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        _, quantized, _, _ = self._vq_vae(z)
        return quantized

    def reconstruct_img_by_embedding(self, embedding):
        loss, quantized, perplexity, _, quantized_list = self._vq_vae(embedding)
        return self._decoder(quantized)

    def reconstruct_img(self, q):
        return self._decoder(q)

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder


class DynamicPropagation(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(DynamicPropagation, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(
            channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(
                channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid //
                          2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(
            channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(
                2*channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid //
                          2, channel_in, incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, input_state):
        B, T, C, H, W = input_state.shape
        input_state = input_state.reshape(B, T*C, H, W)
        # encoder
        skips = []
        hidden_embed = input_state
        for i in range(self.N_T):
            hidden_embed = self.enc[i](hidden_embed)
            if i < self.N_T - 1:
                skips.append(hidden_embed)

        # decoder
        hidden_embed = self.dec[0](hidden_embed)
        for i in range(1, self.N_T):
            hidden_embed = self.dec[i](torch.cat([hidden_embed, skips[-i]], dim=1))

        output_state = hidden_embed.reshape(B, T, C, H, W)
        return output_state

class BeamVQ(nn.Module):
    def __init__(self, 
                 shape_in, 
                 backbone_name,
                 load_pred_train = 0,
                 freeze_vqvae = 0,
                 hid_T=256, 
                 N_T=8, 
                 incep_ker=[3, 5, 7, 11], 
                 groups=8, 
                 res_units=64, 
                 res_layers=2, 
                 embedding_nums=512, 
                 embedding_dim=64,
                 top_k=10,
                 complete=True):
        super(BeamVQ, self).__init__()
        self.complete = complete
        T, C, H, W = shape_in
        self.DST_module = DST(in_channel=C,
                             res_units=res_units,
                             res_layers=res_layers,
                             embedding_dim=embedding_dim,
                             embedding_nums=embedding_nums,
                             top_k=top_k)
        
        self.load_pred_train = load_pred_train
        self.freeze_vqvae = freeze_vqvae
        self.backbone_name = backbone_name
        
        if self.backbone_name == 'unet':
            self.backbone = U_net(input_channels = 10, output_channels = 10, kernel_size = 3, dropout_rate = 0.1)
            print("U_Net has been launched!")
        elif self.backbone_name == 'resnet':
            self.backbone = ResNet(input_channels = 10, output_channels = 10, kernel_size = 3)
            print("ResNet has been launched!")
        elif self.backbone_name == 'cno':
            self.backbone = CNO(in_size = 64, N_layers = 3, in_dim = 10)
            print("CNO has been launched!")
        elif self.backbone_name == 'fno':
            self.backbone = FNO2d(modes1 = 12, modes2 = 12, pred_len = 10, width = 20)
            print("FNO has been launched!")
        elif self.backbone_name == 'lsm':
            self.backbone = LSM(in_dim = 10, out_dim = 1)
            print("LSM has been launched!")
        elif self.backbone_name == 'uno':
            self.backbone = UNO(in_width = 14, width = 32)
            print("UNO has been launched!")
        elif self.backbone_name == 'convlstm':
            self.backbone = convlstm(input_size=(H, W), channels=C, pred_len = T, hidden_dim=[64], num_layers=1)
            print("ConvLSTM has been launched!")
            
            
            
            
        if  self.load_pred_train:
            print_log("Load Pre-trained Model.")
            self.vq_vae.load_state_dict(torch.load("./models/vqvae.ckpt"), strict=False)

        if  self.freeze_vqvae:
            print_log(f"Params of VQVAE is freezed.")
            for p in self.vq_vae.parameters():
                p.requires_grad = False
        self.DynamicPro = DynamicPropagation(T*64, hid_T, N_T, incep_ker, groups)

    def forward(self, input_frames):
        
        latent_embedding = self.backbone(input_frames)

        if not self.complete:
            return latent_embedding, []

        B, T, C, H, W = input_frames.shape 
        latent_embedding_ = latent_embedding.reshape([B * T, C, H, W])
        encoder_embed = self.DST_module._projection(latent_embedding_)

        z = self.DST_module._pre_vq_conv(encoder_embed)
        vq_loss, Latent_embed, _, _, Latent_embed_list = self.DST_module._vq_vae(z)
        _, C_, H_, W_ = Latent_embed.shape
        Latent_embed = Latent_embed.reshape(B, T, C_, H_, W_)
        hidden_dim = self.DynamicPro(Latent_embed)
        B_, T_, C_, H_, W_ = hidden_dim.shape
        hid = hidden_dim.reshape([B_ * T_, C_, H_, W_])

        # Decoder
        predicti_feature = self.DST_module._decoder(hid)
        predicti_feature = predicti_feature.reshape([B, T, C, H, W])
        top_k_features = []
        for quantized_top_k in Latent_embed_list:
            quantized_top_k = quantized_top_k.reshape(B, T, C_, H_, W_)
            hidden_dim_k = self.DynamicPro(quantized_top_k)
            B_, T_, C_, H_, W_ = hidden_dim_k.shape
            hid_k = hidden_dim_k.reshape([B_ * T_, C_, H_, W_])
            predicti_feature_k = self.DST_module._decoder(hid_k)
            predicti_feature_k = predicti_feature_k.reshape([B, T, C, H, W])
            top_k_features.append(predicti_feature_k)

        return predicti_feature, top_k_features

  
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = 'convlstm'
    model = BeamVQ(shape_in=(10,1,64,64), backbone_name=backbone).to(device)
    print(model)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    inputs_tensor = torch.randn(1, 10, 1, 64, 64).to(device)
    outputs_tensor, top_k_features = model(inputs_tensor)
    print(outputs_tensor.shape, len(top_k_features))
    