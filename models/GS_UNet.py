# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 22:17:25 2025

@author: MaxGr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

import math

# =================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class GaussianUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, out_ch=8):
        super().__init__()

        self.inc = DoubleConv(in_ch, base_ch)
        self.out_ch = out_ch

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch, base_ch * 2),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 2, base_ch * 4),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 4, base_ch * 8),
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 8, base_ch * 8),
        )

        # ---------- Bottleneck compress & expand ----------
        k_dim = 16
        bottleneck_ch = base_ch * 8     # e.g. 256

        self.bottleneck_reduce = nn.Conv2d(bottleneck_ch, k_dim, kernel_size=1)
        self.bottleneck_act = nn.ReLU(inplace=True)
        self.bottleneck_expand = nn.Conv2d(k_dim, bottleneck_ch, kernel_size=1)
        
        # ---------- FiLM(K) ----------
        self.k_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True)
        )
        self.k_to_gamma = nn.Linear(128, k_dim)
        self.k_to_beta  = nn.Linear(128, k_dim)

        # ---------- Decoder ----------
        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_ch * 8 + base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_ch * 4 + base_ch * 4, base_ch * 2)

        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_ch * 2 + base_ch * 2, base_ch)

        self.up4 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_ch + base_ch, base_ch)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 6, kernel_size=1)
        )

    def film(self, feat, k):
        """
        feat: [B, C, Hc, Wc]
        k: [B] or [B, 1]
        """
        B, C, H, W = feat.shape

        K_max = 100000
        k = torch.log1p(k.float()) / math.log(K_max + 1)
        
        k_emb = self.k_embed(k)           # [B, 128]
        gamma = self.k_to_gamma(k_emb)    # [B, 16]
        beta  = self.k_to_beta(k_emb)     # [B, 16]

        gamma = gamma.view(-1, C, 1, 1)
        beta  = beta.view(-1, C, 1, 1)

        # feature wise linear modulation
        return feat * gamma + beta

    def forward(self, x, k):
        B, C, H, W = x.shape

        # encoder
        x1 = self.inc(x)          # B, C, H, W
        x2 = self.down1(x1)       # /2
        x3 = self.down2(x2)       # /4
        x4 = self.down3(x3)       # /8
        x5 = self.down4(x4)       # /16

        # ----- reduced bottleneck -----
        bottleneck = self.bottleneck_reduce(x5)       # [B, k_dim, Hc, Wc]
        bottleneck = self.bottleneck_act(bottleneck)

        # FiLM
        bottleneck = self.film(bottleneck, k)

        # expand
        bottleneck = self.bottleneck_expand(bottleneck) # [B, 256, Hc, Wc]
        # ----- reduced bottleneck -----

        # decoder
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        feat = self.conv4(u4)
        
        out = self.out_conv(feat)   # B, out_ch, H, W
        # params = params.reshape(B, self.out_ch, N).permute(0,2,1)
        
        scale = out[:, 0:2, ...] # >= 0
        color = out[:, 2:5, ...] # [0, 1]
        rot   = out[:, 5:6, ...] # [0, 1]

        scale = F.softplus(scale)
        color = torch.tanh(color) + x
        rot   = torch.sigmoid(rot) * 2 * torch.pi

        return [scale, color, rot]#, heatmap

# model = GaussianUNet().to('cuda')
# summary(model, input_size=(3, 512, 512))



class HeatmapUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, out_ch=1):
        super().__init__()

        self.inc = DoubleConv(in_ch, base_ch)
        self.out_ch = out_ch

        # ---------- Encoder ----------
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch, base_ch * 2),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 2, base_ch * 4),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 4, base_ch * 8),
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 8, base_ch * 8),
        )

        # ---------- Bottleneck compress & expand ----------
        k_dim = 16
        bottleneck_ch = base_ch * 8     # e.g. 256

        self.bottleneck_reduce = nn.Conv2d(bottleneck_ch, k_dim, kernel_size=1)
        self.bottleneck_act = nn.ReLU(inplace=True)
        self.bottleneck_expand = nn.Conv2d(k_dim, bottleneck_ch, kernel_size=1)
        
        # ---------- FiLM(K) ----------
        self.k_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True)
        )
        self.k_to_gamma = nn.Linear(128, k_dim)
        self.k_to_beta  = nn.Linear(128, k_dim)

        # ---------- Decoder ----------
        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_ch * 8 + base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_ch * 4 + base_ch * 4, base_ch * 2)

        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_ch * 2 + base_ch * 2, base_ch)

        self.up4 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_ch + base_ch, base_ch)

        self.pos_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, kernel_size=1)
        )

    def film(self, feat, k):
        """
        feat: [B, C, Hc, Wc]
        k: [B] or [B, 1]
        """
        B, C, H, W = feat.shape

        K_max = 100000
        k = torch.log1p(k.float()) / math.log(K_max + 1)
        
        k_emb = self.k_embed(k)           # [B, 128]
        gamma = self.k_to_gamma(k_emb)    # [B, 16]
        beta  = self.k_to_beta(k_emb)     # [B, 16]

        gamma = gamma.view(-1, C, 1, 1)
        beta  = beta.view(-1, C, 1, 1)

        # feature wise linear modulation
        return feat * gamma + beta

    def forward(self, x, k):
        B, C, H, W = x.shape

        # ----- encoder -----
        x1 = self.inc(x)          # B, C, H, W
        x2 = self.down1(x1)       # /2
        x3 = self.down2(x2)       # /4
        x4 = self.down3(x3)       # /8
        x5 = self.down4(x4)       # /16
        
        # ----- reduced bottleneck -----
        bottleneck = self.bottleneck_reduce(x5)       # [B, k_dim, Hc, Wc]
        bottleneck = self.bottleneck_act(bottleneck)

        # FiLM
        bottleneck = self.film(bottleneck, k)

        # expand
        bottleneck = self.bottleneck_expand(bottleneck) # [B, 256, Hc, Wc]

        # ----- decoder -----
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        feat = self.conv4(u4)
        
        heatmap = self.pos_conv(feat)
      
        return heatmap




# =================================================================

class GaussianUNet_Plus(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()

        self.inc = DoubleConv(in_ch, base_ch)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch, base_ch * 2),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 2, base_ch * 4),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 4, base_ch * 8),
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 8, base_ch * 8),
        )

        # ---------- Bottleneck compress & expand ----------
        k_dim = 64
        bottleneck_ch = base_ch * 8     # e.g. 256

        self.bottleneck_reduce = nn.Conv2d(bottleneck_ch, k_dim, kernel_size=1)
        self.bottleneck_act = nn.ReLU(inplace=True)
        self.bottleneck_expand = nn.Conv2d(k_dim, bottleneck_ch, kernel_size=1)
        
        # ---------- FiLM(K) ----------
        self.k_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True)
        )
        self.k_to_gamma = nn.Linear(128, k_dim)
        self.k_to_beta  = nn.Linear(128, k_dim)

        # ---------- Decoder ----------
        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_ch * 8 + base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_ch * 4 + base_ch * 4, base_ch * 2)

        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_ch * 2 + base_ch * 2, base_ch)

        self.up4 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_ch + base_ch, base_ch)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 8, kernel_size=1)
        )

    def film(self, feat, k):
        """
        feat: [B, C, Hc, Wc]
        k: [B] or [B, 1]
        """
        B, C, H, W = feat.shape
        # k = torch.full((B, 1), float(k), device=feat.device)

        K_max = 100000
        k = torch.log1p(k.float()) / math.log(K_max + 1)
        
        k_emb = self.k_embed(k)           # [B, 128]
        gamma = self.k_to_gamma(k_emb)    # [B, 16]
        beta  = self.k_to_beta(k_emb)     # [B, 16]

        gamma = gamma.view(-1, C, 1, 1)
        beta  = beta.view(-1, C, 1, 1)

        # feature wise linear modulation
        return feat * gamma + beta

    def forward(self, x, k=50000):
        B, C, H, W = x.shape
        # k = torch.full((B, 1), float(k), device=x.device)

        # encoder
        x1 = self.inc(x)          # B, C, H, W
        x2 = self.down1(x1)       # /2
        x3 = self.down2(x2)       # /4
        x4 = self.down3(x3)       # /8
        x5 = self.down4(x4)       # /16

        # ----- reduced bottleneck -----
        bottleneck = self.bottleneck_reduce(x5)       # [B, k_dim, Hc, Wc]
        bottleneck = self.bottleneck_act(bottleneck)

        # FiLM
        bottleneck = self.film(bottleneck, k)

        # expand
        bottleneck = self.bottleneck_expand(bottleneck) # [B, 256, Hc, Wc]
        # ----- reduced bottleneck -----

        # decoder
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        feat = self.conv4(u4)
        
        out = self.out_conv(feat)   # B, out_ch, H, W
        
        offset = out[:, 0:2, ...] # [-1, 1]
        scale = out[:, 2:4, ...] # >= 0
        color = out[:, 4:7, ...] # [0, 1]
        rot   = out[:, 7:8, ...] # [0, 1]
        
        # print(offset.shape)
        # print(scale.shape)
        # print(color.shape)
        # print(rot.shape)

        offset = torch.tanh(offset)
        scale = F.softplus(scale)
        color = torch.tanh(color) + x
        rot   = torch.sigmoid(rot) * 2 * torch.pi

        return [offset, scale, color, rot]#, heatmap


# model = GaussianUNet_Plus().to('cuda')
# summary(model, input_size=(3, 512, 512))












