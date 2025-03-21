"""
本代码改编自 https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py
实现了带自注意力机制的条件GAN（SAGAN）生成器和判别器，使用谱归一化（spectral normalization）
对网络进行正则化以提高训练稳定性。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_


# -----------------------------
# 权重初始化函数：对于线性层和卷积层，采用Xavier均匀初始化，并将偏置初始化为0
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)


# -----------------------------
# 封装一个带谱归一化的卷积层，调用spectral_norm包装nn.Conv2d
def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


# 封装一个带谱归一化的全连接层
def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))


# -----------------------------
# 自注意力模块
class Self_Attn(nn.Module):
    """自注意力层，用于捕捉长程依赖和全局特征"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        # 通过1x1卷积生成Theta特征，输出通道数为 in_channels // 8
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        # 通过1x1卷积生成Phi特征
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        # 通过1x1卷积生成g特征，输出通道数为 in_channels // 2
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        # 通过1x1卷积将自注意力加权后的特征映射回原通道数
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        # 使用最大池化降低空间分辨率，便于计算注意力矩阵
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        # 在最后一个维度上进行softmax归一化
        self.softmax  = nn.Softmax(dim=-1)
        # 可学习的参数sigma，初始值为0，控制自注意力加权特征的影响程度
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        输入：
            x: 输入特征图，形状 (B, C, H, W)
        输出：
            out: 与输入x相加后的自注意力增强特征图
        """
        _, ch, h, w = x.size()
        # Theta路径：1x1卷积后reshape为 (B, C//8, H*W)
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi路径：1x1卷积后池化，reshape为 (B, C//8, H*W//4)
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # 计算注意力矩阵：对Theta和Phi进行矩阵乘法，结果形状为 (B, H*W, H*W//4)
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g路径：1x1卷积后池化，reshape为 (B, C//2, H*W//4)
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # 对g和注意力矩阵进行矩阵乘法，得到注意力加权特征，reshape回 (B, C//2, H, W)
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        # 通过1x1卷积将加权特征映射回原始通道数
        attn_g = self.snconv1x1_attn(attn_g)
        # 最终输出为输入x与加权特征的和，权重由sigma控制
        out = x + self.sigma * attn_g
        return out


"""
生成器部分
"""

# 条件批归一化层：根据标签嵌入信息调整归一化结果
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        """
        参数：
            num_features: 输入特征图通道数
            dim_embed: 标签嵌入维度
        """
        super().__init__()
        self.num_features = num_features
        # 标准批归一化，但不使用可学习参数（affine=False）
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        # 两个全连接层分别映射标签嵌入到gamma和beta
        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        """
        输入：
            x: 特征图 (B, C, H, W)
            y: 标签嵌入 (B, dim_embed)
        输出：
            条件归一化后的特征图
        """
        out = self.bn(x)
        # 计算缩放因子gamma和偏置beta，并reshape为 (B, C, 1, 1)
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        # 采用公式：out = bn(x) + gamma * bn(x) + beta
        out = out + gamma * out + beta
        return out


# 生成器残差块（GenBlock），包含条件批归一化、ReLU激活、卷积和上采样
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_embed):
        super(GenBlock, self).__init__()
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, dim_embed)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, dim_embed)
        self.snconv2d2 = snconv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels):
        """
        前向传播：
            x: 输入特征图 (B, in_channels, H, W)
            labels: 标签嵌入 (B, dim_embed)
        输出：
            输出特征图 (B, out_channels, 2H, 2W) —— 经过上采样后尺寸加倍
        """
        x0 = x  # 旁路分支保存原始输入

        # 主分支：先条件归一化，再ReLU激活，然后上采样
        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 上采样（nearest插值）
        x = self.snconv2d1(x)  # 进行卷积
        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.snconv2d2(x)  # 第二次卷积

        # 旁路分支：对原始输入进行上采样和1x1卷积
        x0 = F.interpolate(x0, scale_factor=2, mode='nearest')
        x0 = self.snconv2d0(x0)

        # 将主分支和旁路分支相加得到最终输出
        out = x + x0
        return out

class CcGAN_SAGAN_Generator(nn.Module):
    """生成器，根据随机噪声和标签生成图像 (教师模型)"""

    def __init__(self, dim_z, dim_embed=128, nc=3, gene_ch=64):
        super(CcGAN_SAGAN_Generator, self).__init__()
        self.dim_z = dim_z
        self.gene_ch = gene_ch
        # 全连接，把 z -> (gene_ch*16)*4*4
        self.snlinear0 = snlinear(in_features=dim_z, out_features=gene_ch*16*4*4)

        # 依次4个生成块 + 自注意力
        self.block1 = GenBlock(gene_ch*16, gene_ch*8, dim_embed)
        self.block2 = GenBlock(gene_ch*8, gene_ch*4, dim_embed)
        self.block3 = GenBlock(gene_ch*4, gene_ch*2, dim_embed)
        self.self_attn = Self_Attn(gene_ch*2)
        self.block4 = GenBlock(gene_ch*2, gene_ch, dim_embed)

        # 最后的 BN + ReLU + conv
        self.bn = nn.BatchNorm2d(gene_ch, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=gene_ch, out_channels=nc, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # 权重初始化
        self.apply(init_weights)

    def forward(self, z, labels, return_features):
        """
        参数:
            z: (B, dim_z)
            labels: (B, dim_embed)
            return_features: 如果为 True, 返回 [block1_out, block2_out, block3_out, block4_out] 等中间特征
        """
        # 特征收集列表
        feat_list = []

        out = self.snlinear0(z)  # (B, gene_ch*16*4*4)
        out = out.view(-1, self.gene_ch*16, 4, 4)

        out = self.block1(out, labels)   # -> (B, gene_ch*8, 8, 8)
        if return_features:
            feat_list.append(out)

        out = self.block2(out, labels)   # -> (B, gene_ch*4, 16, 16)
        if return_features:
            feat_list.append(out)

        out = self.block3(out, labels)   # -> (B, gene_ch*2, 32, 32)
        if return_features:
            feat_list.append(out)

        out = self.self_attn(out)        # 仍是 (B, gene_ch*2, 32, 32)
        out = self.block4(out, labels)   # -> (B, gene_ch, 64, 64)
        if return_features:
            feat_list.append(out)

        out = self.bn(out)
        out = self.relu(out)
        out = self.snconv2d1(out)
        out = self.tanh(out)             # -> (B, nc, 64, 64)

        if return_features:
            return out, feat_list
        else:
            return out


class CcGAN_SAGAN_Generator_Student(nn.Module):
    """学生生成器，通道减半；同时在 forward 中返回若干层特征，并用 1x1 conv 做通道对齐"""

    def __init__(self, dim_z, dim_embed=128, nc=3, gene_ch=32):
        super(CcGAN_SAGAN_Generator_Student, self).__init__()
        self.dim_z = dim_z
        self.gene_ch = gene_ch

        # 线性层把 z -> (gene_ch*16)*4*4
        self.snlinear0 = snlinear(dim_z, gene_ch*16*4*4)

        # 4个生成块
        self.block1 = GenBlock(gene_ch*16, gene_ch*8, dim_embed)
        self.block2 = GenBlock(gene_ch*8, gene_ch*4, dim_embed)
        self.block3 = GenBlock(gene_ch*4, gene_ch*2, dim_embed)
        self.self_attn = Self_Attn(gene_ch*2)
        self.block4 = GenBlock(gene_ch*2, gene_ch, dim_embed)

        self.bn = nn.BatchNorm2d(gene_ch, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(gene_ch, nc, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z, labels, return_features):
        """
        若 return_features=True, 则返回:
          final_out,
          [feat_s1, feat_s2, feat_s3, feat_s4],  # 学生各层特征
          [feat_s1_aligned, feat_s2_aligned, feat_s3_aligned, feat_s4_aligned]  # 适配后(对齐教师通道)的特征
        """
        feat_list = []

        out = self.snlinear0(z)
        out = out.view(-1, self.gene_ch*16, 4, 4)

        out1 = self.block1(out, labels)    # (B, gene_ch*8, 8, 8)
        out2 = self.block2(out1, labels)   # (B, gene_ch*4, 16, 16)
        out3 = self.block3(out2, labels)   # (B, gene_ch*2, 32, 32)
        out3_attn = self.self_attn(out3)   # (B, gene_ch*2, 32, 32)
        out4 = self.block4(out3_attn, labels)  # (B, gene_ch, 64, 64)

        out_bn = self.bn(out4)
        out_relu = self.relu(out_bn)
        out_final = self.snconv2d1(out_relu)
        out_final = self.tanh(out_final)

        if return_features:
            feat_list = [out1, out2, out3, out3_attn, out4]  # 返回原始特征
            return out_final, feat_list
        else:
            return out_final


# # SAGAN生成器（条件GAN版本），根据随机噪声和标签生成图像
# class CcGAN_SAGAN_Generator(nn.Module):
#     """生成器，根据随机噪声和标签生成图像"""

#     def __init__(self, dim_z, dim_embed=128, nc=3, gene_ch=32):
#         """
#         参数：
#             dim_z: 噪声向量的维度
#             dim_embed: 标签嵌入的维度
#             nc: 输出图像通道数（例如3表示RGB）
#             gene_ch: 生成器基础通道数
#         """
#         super(CcGAN_SAGAN_Generator, self).__init__()
#         self.dim_z = dim_z
#         self.gene_ch = gene_ch

#         # 将噪声向量通过全连接层映射为初始特征图，尺寸为 (gene_ch*16) x 4 x 4
#         self.snlinear0 = snlinear(in_features=dim_z, out_features=gene_ch*16*4*4)
#         # 依次经过4个生成器块，每个块完成上采样和特征变换
#         self.block1 = GenBlock(gene_ch*16, gene_ch*8, dim_embed)
#         self.block2 = GenBlock(gene_ch*8, gene_ch*4, dim_embed)
#         self.block3 = GenBlock(gene_ch*4, gene_ch*2, dim_embed)
#         # 在特征图较大时加入自注意力模块以捕捉长程依赖
#         self.self_attn = Self_Attn(gene_ch*2)
#         self.block4 = GenBlock(gene_ch*2, gene_ch, dim_embed)
#         # 最后批归一化和ReLU激活
#         self.bn = nn.BatchNorm2d(gene_ch, eps=1e-5, momentum=0.0001, affine=True)
#         self.relu = nn.ReLU(inplace=True)
#         # 最终卷积层，将特征图转换为目标通道数nc，采用3x3卷积
#         self.snconv2d1 = snconv2d(in_channels=gene_ch, out_channels=nc, kernel_size=3, stride=1, padding=1)
#         # Tanh激活函数将像素值映射到[-1, 1]
#         self.tanh = nn.Tanh()

#         # 对所有子模块进行权重初始化
#         self.apply(init_weights)

#     def forward(self, z, labels):
#         """
#         输入：
#             z: 随机噪声向量 (B, dim_z)
#             labels: 标签嵌入 (B, dim_embed)
#         输出：
#             生成的图像 (B, nc, H, W)
#         """
#         # 先将噪声通过全连接层转换为初始特征图，并reshape为 (B, gene_ch*16, 4, 4)
#         out = self.snlinear0(z)
#         out = out.view(-1, self.gene_ch*16, 4, 4)
#         # 依次经过各个生成块进行上采样和特征转换
#         out = self.block1(out, labels)    # 从4x4上采样到8x8
#         out = self.block2(out, labels)    # 8x8 -> 16x16
#         out = self.block3(out, labels)    # 16x16 -> 32x32
#         out = self.self_attn(out)         # 自注意力模块（保持32x32）
#         out = self.block4(out, labels)    # 32x32 -> 64x64
#         out = self.bn(out)
#         out = self.relu(out)
#         out = self.snconv2d1(out)           # 最终卷积转换到目标通道数
#         out = self.tanh(out)                # 归一化到[-1, 1]
#         return out


"""
判别器部分
"""

# 判别器优化块，用于第一层处理，包含卷积、ReLU、下采样及旁路连接
class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        参数：
            in_channels: 输入图像通道数
            out_channels: 输出通道数
        """
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)  # 平均池化下采样
        # 1x1卷积用于旁路分支，匹配通道数
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        输入：
            x: 输入特征图 (B, in_channels, H, W)
        输出：
            输出特征图 (B, out_channels, H/2, W/2)
        """
        x0 = x  # 旁路分支保存原始输入

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)  # 主分支进行下采样

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)  # 旁路分支进行1x1卷积

        out = x + x0  # 两个分支相加
        return out


# 判别器块，可选下采样，且当输入输出通道不匹配时使用1x1卷积调整
class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        # 标记输入与输出通道是否不匹配
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        # 1x1卷积用于调整通道数
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        """
        输入：
            x: 输入特征图 (B, in_channels, H, W)
            downsample: 是否执行下采样操作（默认为True）
        输出：
            融合后的特征图 (B, out_channels, H/2, W/2)（若下采样）
        """
        x0 = x  # 旁路分支

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


# SAGAN判别器（条件GAN版本），利用投影方法将标签信息融入判别
class CcGAN_SAGAN_Discriminator(nn.Module):
    """判别器，通过条件投影融合标签嵌入，实现条件判别"""

    def __init__(self, dim_embed=128, nc=3, disc_ch=64):
        """
        参数：
            dim_embed: 标签嵌入维度
            nc: 输入图像通道数
            disc_ch: 判别器基础通道数
        """
        super(CcGAN_SAGAN_Discriminator, self).__init__()
        self.disc_ch = disc_ch
        # 第一个优化块，直接处理原始图像
        self.opt_block1 = DiscOptBlock(nc, disc_ch)
        # 自注意力模块，在较低分辨率下捕捉全局信息
        self.self_attn = Self_Attn(disc_ch)
        # 后续判别器块，逐步增加通道数并下采样
        self.block1 = DiscBlock(disc_ch, disc_ch*2)
        self.block2 = DiscBlock(disc_ch*2, disc_ch*4)
        self.block3 = DiscBlock(disc_ch*4, disc_ch*8)
        self.block4 = DiscBlock(disc_ch*8, disc_ch*16)
        self.relu = nn.ReLU(inplace=True)
        # 线性层将卷积特征展平后映射为标量输出
        self.snlinear1 = snlinear(in_features=disc_ch*16*4*4, out_features=1)
        # 条件投影层：将标签嵌入映射到与展平特征相同的维度
        self.sn_embedding1 = snlinear(dim_embed, disc_ch*16*4*4, bias=False)

        # 对所有模块进行权重初始化
        self.apply(init_weights)
        xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, x, labels):
        """
        输入：
            x: 输入图像 (B, nc, H, W)
            labels: 标签嵌入 (B, dim_embed)
        输出：
            判别器输出分数 (B, 1)
        """
        # 通过第一个优化块，下采样并提取特征，输出尺寸例如32x32（假设输入64x64）
        out = self.opt_block1(x)
        # 自注意力模块，保持尺寸不变
        out = self.self_attn(out)
        out = self.block1(out)    # 例如下采样到16x16
        out = self.block2(out)    # 16x16 -> 8x8
        out = self.block3(out)    # 8x8 -> 4x4
        # 第四个块不进行下采样（保持4x4）
        out = self.block4(out, downsample=False)
        out = self.relu(out)
        # 将特征图展平为向量
        out = out.view(-1, self.disc_ch*16*4*4)
        # 通过线性层得到无条件的判别分数
        output1 = torch.squeeze(self.snlinear1(out))
        # 条件投影：将标签嵌入经过全连接层后与展平特征逐元素相乘，再求和得到条件分数
        h_labels = self.sn_embedding1(labels)  # (B, disc_ch*16*4*4)
        proj = torch.mul(out, h_labels)
        output2 = torch.sum(proj, dim=[1])
        # 最终输出为两部分分数之和
        output = output1 + output2
        return output


# -----------------------------
# 测试部分：构建生成器和判别器，并进行前向传播测试与参数统计
if __name__ == "__main__":
    # 辅助函数，统计网络中参数总数及可训练参数数目
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    # 初始化生成器，噪声维度为256，标签嵌入维度128，生成器基础通道64
    netG = CcGAN_SAGAN_Generator(dim_z=256, dim_embed=128, gene_ch=64).cuda()  # 参数约131M

    netG_student = CcGAN_SAGAN_Generator_Student(dim_z=256, dim_embed=128, gene_ch=32).cuda()  # 参数约33M
    # 初始化判别器，标签嵌入维度128，判别器基础通道64
    netD = CcGAN_SAGAN_Discriminator(dim_embed=128, disc_ch=64).cuda()  # 参数约162M

    # 若需要使用多GPU训练，可取消以下注释
    # netG = nn.DataParallel(netG)
    # netD = nn.DataParallel(netD)

    # N = 4  # 测试批次大小
    # z = torch.randn(N, 256).cuda()  # 随机生成噪声向量
    # y = torch.randn(N, 128).cuda()   # 随机生成标签嵌入
    # x = netG(z, y,return_features=False)  # 生成图像
    # o = netG_student(z, y,return_features=False)
    # print(x.size())  # 输出生成图像尺寸
    # print(o.size())  # 输出判别器输出尺寸
    # # 打印生成器和判别器的参数统计信息
    # print('G:', get_parameter_number(netG))
    # print('G_s:', get_parameter_number(netG_student))
# 测试代码
# gene_ch = 32
# dim_z = 256
# dim_embed = 128
# netG_student = CcGAN_SAGAN_Generator_Student(dim_z, dim_embed, gene_ch=gene_ch)

# # 随机输入
# z = torch.randn(1, dim_z)
# labels = torch.randn(1, dim_embed)

# # 前向传播
# fake, features = netG_student(z, labels, return_features=True)
# print(f"Block1 输出通道数: {features[0].shape[1]}")  # 应为 512 → 256（错误时显示 512）