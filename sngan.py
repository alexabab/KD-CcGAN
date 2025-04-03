'''
引用的GitHub项目链接：
https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

chainer实现参考：
https://github.com/pfnet-research/sngan_projection
'''

# ResNet生成器和判别器的实现

import torch
from torch import nn
import torch.nn.functional as F
# 导入numpy库
import numpy as np
# 导入谱归一化函数，用于稳定GAN训练
from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_
# 设置图像的通道数（例如RGB图像为3）和卷积层是否使用偏置
channels = 3
bias = True

######################################################################################################################
# 生成器部分
######################################################################################################################
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)

# 定义条件批归一化层，用于根据条件（如标签）调整归一化的缩放和平移参数
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        """
        参数:
            num_features: 输入特征的通道数
            dim_embed: 条件向量（例如标签）的嵌入维度
        """
        super().__init__()
        self.num_features = num_features
        # 创建不带可学习参数的标准批归一化层
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        # 通过全连接层根据条件向量生成缩放因子gamma
        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        # 通过全连接层根据条件向量生成偏移量beta
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)
    def forward(self, x, y):
        """
        前向传播：
            x: 输入特征图，形状为 [batch, num_features, H, W]
            y: 条件嵌入向量，形状为 [batch, dim_embed]
        """
        # 对输入x进行批归一化
        out = self.bn(x)
        # 计算gamma因子，并reshape为 [batch, num_features, 1, 1]
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        # 计算beta偏置，并reshape为相同形状
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        # 将批归一化后的结果按条件调整（注意这里采用的公式为 out + out * gamma + beta）
        out = out + out * gamma + beta
        return out

# 定义生成器中的残差块（ResBlock）
class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, dim_embed, bias=True):
        """
        参数:
            in_channels: 输入特征图的通道数
            out_channels: 输出特征图的通道数
            dim_embed: 条件嵌入向量的维度
            bias: 卷积层是否使用偏置
        """
        super(ResBlockGenerator, self).__init__()

        # 定义两个3x3卷积层，步幅为1，填充1，保持特征图尺寸
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        # 使用Xavier均匀初始化卷积层的权重，增进训练稳定性
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        # 条件归一化：分别为第一个和第二个卷积层构建条件批归一化模块
        self.condgn1 = ConditionalBatchNorm2d(in_channels, dim_embed)
        self.condgn2 = ConditionalBatchNorm2d(out_channels, dim_embed)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 上采样层，将特征图尺寸扩大2倍
        self.upsample = nn.Upsample(scale_factor=2)

        # 无条件分支：采用标准批归一化、ReLU和上采样再加卷积构成
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )

        # bypass分支：使用1x1卷积匹配通道数，并进行上采样，构成残差连接
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=bias) # 保持高宽不变
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x, y):
        """
        前向传播：
            x: 输入特征图
            y: 条件嵌入向量，如果使用条件则不为None，否则为None
        """
        if y is not None:
            # 条件分支：先进行条件批归一化，再ReLU激活，上采样，然后卷积
            out = self.condgn1(x, y)
            out = self.relu(out)
            out = self.upsample(out)
            out = self.conv1(out)
            # 第二次条件批归一化和卷积处理
            out = self.condgn2(out, y)
            out = self.relu(out)
            out = self.conv2(out)
            # 与旁路分支相加，构成残差连接
            out = out + self.bypass(x)
        else:
            # 无条件情况：直接用预定义的模型处理，并加上旁路分支
            out = self.model(x) + self.bypass(x)
        return out

# 定义SNGAN生成器，基于ResNet结构和条件信息
class sngan_generator(nn.Module):
    #教师生成器
    def __init__(self, nz=256, dim_embed=128, gen_ch=64):
        """
        参数:
            nz: 噪声向量的维度
            dim_embed: 条件嵌入向量的维度
            gen_ch: 生成器基础通道数
        """
        super(sngan_generator, self).__init__()
        self.z_dim = nz
        self.dim_embed = dim_embed
        self.gen_ch = gen_ch

        # 全连接层，将噪声向量映射为4x4大小、通道数为gen_ch*16的特征图
        self.dense = nn.Linear(self.z_dim, 4 * 4 * gen_ch * 16, bias=True)
        # 最终卷积层，用于将特征图映射到目标通道数（例如3通道RGB）
        self.final = nn.Conv2d(gen_ch, channels, 3, stride=1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        # 定义4个残差块，逐步上采样并减少通道数
        self.genblock0 = ResBlockGenerator(gen_ch * 16, gen_ch * 8, dim_embed=dim_embed)  # 4x4 -> 8x8
        self.genblock1 = ResBlockGenerator(gen_ch * 8, gen_ch * 4, dim_embed=dim_embed)   # 8x8 -> 16x16
        self.genblock2 = ResBlockGenerator(gen_ch * 4, gen_ch * 2, dim_embed=dim_embed)   # 16x16 -> 32x32
        self.genblock3 = ResBlockGenerator(gen_ch * 2, gen_ch, dim_embed=dim_embed)       # 32x32 -> 64x64

        # 最终模块：先批归一化，ReLU激活，再卷积和Tanh激活，将输出映射到[-1,1]之间
        self.final = nn.Sequential(
            nn.BatchNorm2d(gen_ch),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z, y, return_features):
        """
        前向传播：
            z: 随机噪声向量，形状为 [batch, nz]
            y: 条件嵌入向量，形状为 [batch, dim_embed]
            return_features: 如果为 True, 返回 [view_out, block0_out, block1_out, block2_out, block3_out, final_out] 等中间特征
        """
        # 特征收集列表
        feat_list = []
        
        # 将噪声向量通过全连接层转换为初始特征图，尺寸为4x4
        z = z.view(z.size(0), z.size(1))
        out = self.dense(z)
        out = out.view(-1, self.gen_ch * 16, 4, 4)
        if return_features:
            feat_list.append(out)
        # 依次经过各个残差块，利用条件信息进行上采样和特征提取
        out = self.genblock0(out, y)
        if return_features:
            feat_list.append(out)
        out = self.genblock1(out, y)
        if return_features:
            feat_list.append(out)
        out = self.genblock2(out, y)
        if return_features:
            feat_list.append(out)
        out = self.genblock3(out, y)
        if return_features:
            feat_list.append(out)
        # 经过最终模块生成图像
        out = self.final(out)
        if return_features:
            feat_list.append(out)

        if return_features:
            return out, feat_list
        else:
            return out


class sngan_generator_student(nn.Module):  # 确保继承自nn.Module
    def __init__(self, nz=128, dim_embed=128, gen_ch=64):
        super().__init__()
        """
        参数:
            nz: 噪声向量的维度
            dim_embed: 条件嵌入向量的维度
            gen_ch: 生成器基础通道数
        """
        self.z_dim = nz
        self.dim_embed = dim_embed
        self.gen_ch = gen_ch

        # 全连接层，将噪声向量映射为4x4大小、通道数为gen_ch*16的特征图
        self.dense = nn.Linear(self.z_dim, 4 * 4 * gen_ch * 16, bias=True)
        # 最终卷积层，用于将特征图映射到目标通道数（例如3通道RGB）
        self.final = nn.Conv2d(gen_ch, channels, 3, stride=1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        # 定义4个残差块，逐步上采样并减少通道数
        self.genblock0 = ResBlockGenerator(gen_ch * 16, gen_ch * 8, dim_embed=dim_embed)  # 4x4 -> 8x8
        self.genblock1 = ResBlockGenerator(gen_ch * 8, gen_ch * 4, dim_embed=dim_embed)   # 8x8 -> 16x16
        self.genblock2 = ResBlockGenerator(gen_ch * 4, gen_ch * 2, dim_embed=dim_embed)   # 16x16 -> 32x32
        self.genblock3 = ResBlockGenerator(gen_ch * 2, gen_ch, dim_embed=dim_embed)       # 32x32 -> 64x64

        # 最终模块：先批归一化，ReLU激活，再卷积和Tanh激活，将输出映射到[-1,1]之间
        self.final = nn.Sequential(
            nn.BatchNorm2d(gen_ch),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )
        # adapter用于对齐教师特征
        self.feat_adapters = nn.ModuleDict()

        # 权重初始化
        self.apply(init_weights)

    def forward(self, z, y, return_features):
        """
        前向传播：
            z: 随机噪声向量，形状为 [batch, nz]
            y: 条件嵌入向量，形状为 [batch, dim_embed]
            return_features: 如果为 True, 返回 [view_out, block0_out, block1_out, block2_out, block3_out, final_out] 等中间特征
        """
        # 特征收集列表
        feat_list = []
        
        # 将噪声向量通过全连接层转换为初始特征图，尺寸为4x4
        z = z.view(z.size(0), z.size(1))
        out = self.dense(z)
        out = out.view(-1, self.gen_ch * 16, 4, 4)
        if return_features:
            feat_list.append(out)
        # 依次经过各个残差块，利用条件信息进行上采样和特征提取
        out = self.genblock0(out, y)
        if return_features:
            feat_list.append(out)
        out = self.genblock1(out, y)
        if return_features:
            feat_list.append(out)
        out = self.genblock2(out, y)
        if return_features:
            feat_list.append(out)
        out = self.genblock3(out, y)
        if return_features:
            feat_list.append(out)
        # 经过最终模块生成图像
        out = self.final(out)
        if return_features:
            feat_list.append(out)

        if return_features:
            return out, feat_list
        else:
            return out

    def match_teacher_feat(self, student_feat, teacher_feat):
            """
            使用1x1卷积对学生特征进行通道变换，以匹配教师特征的通道数，同时保持反向传播可学习。
            """
            teacher_channels = teacher_feat.size(1)
            student_channels = student_feat.size(1)
            key = f"{student_channels}_to_{teacher_channels}"

            if key not in self.feat_adapters:
                self.feat_adapters[key] = nn.Conv2d(
                    student_channels, teacher_channels, kernel_size=1, stride=1, padding=0
                ).to(student_feat.device)

            return self.feat_adapters[key](student_feat)
######################################################################################################################
# 判别器部分
######################################################################################################################

# 定义判别器中的残差块
class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        """
        参数:
            in_channels: 输入特征图的通道数
            out_channels: 输出特征图的通道数
            stride: 卷积步幅，stride>1时实现下采样
        """
        super(ResBlockDiscriminator, self).__init__()

        # 定义两个3x3卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        # 根据stride设置主分支模型
        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

        # bypass分支：通过1x1卷积匹配通道数，并在需要时进行下采样
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=bias)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if stride != 1:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
            )

    def forward(self, x):
        # 主分支与旁路分支相加，构成残差连接
        return self.model(x) + self.bypass(x)

# 专门为判别器第一层设计的残差块，不在输入前使用ReLU激活（因为原始图像不适合先激活）
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        """
        参数:
            in_channels: 输入通道数（通常为图像通道数，如3）
            out_channels: 输出通道数
            stride: 是否下采样（一般设为2）
        """
        super(FirstResBlockDiscriminator, self).__init__()

        # 定义两个3x3卷积层和一个1x1旁路卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        # 不对原始图像进行ReLU预激活，直接进入卷积操作后池化
        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# 定义SNGAN判别器，采用ResNet结构，并融入条件信息
class sngan_discriminator(nn.Module):
    def __init__(self, dim_embed=128, disc_ch=64):
        """
        参数:
            dim_embed: 条件嵌入向量的维度
            disc_ch: 判别器的基础通道数
        """
        super(sngan_discriminator, self).__init__()
        self.dim_embed = dim_embed
        self.disc_ch = disc_ch

        # 第一部分判别器：多层残差块逐步下采样输入图像
        self.discblock1 = nn.Sequential(
            FirstResBlockDiscriminator(channels, disc_ch, stride=2),    # 从64x64下采样到32x32
            ResBlockDiscriminator(disc_ch, disc_ch * 2, stride=2),         # 32x32 -> 16x16
            ResBlockDiscriminator(disc_ch * 2, disc_ch * 4, stride=2),       # 16x16 -> 8x8
        )
        # 第二部分判别器：继续下采样
        self.discblock2 = ResBlockDiscriminator(disc_ch * 4, disc_ch * 8, stride=2)  # 8x8 -> 4x4
        # 第三部分判别器：不改变尺寸，提取高维特征后ReLU激活
        self.discblock3 = nn.Sequential(
            ResBlockDiscriminator(disc_ch * 8, disc_ch * 16, stride=1),  # 保持4x4
            nn.ReLU(),
        )

        # 全连接层，将判别器输出映射到标量输出
        self.linear1 = nn.Linear(disc_ch * 16 * 4 * 4, 1, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        self.linear1 = spectral_norm(self.linear1)
        # 条件映射层，将条件信息转换为与判别器特征同维度的向量
        self.linear2 = nn.Linear(self.dim_embed, disc_ch * 16 * 4 * 4, bias=False)
        nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        self.linear2 = spectral_norm(self.linear2)

    def forward(self, x, y):
        """
        前向传播：
            x: 输入图像，形状为 [batch, channels, H, W]
            y: 条件嵌入向量，形状为 [batch, dim_embed]
        """
        output = self.discblock1(x)
        output = self.discblock2(output)
        output = self.discblock3(output)

        # 将特征图展平为向量
        output = output.view(-1, self.disc_ch * 16 * 4 * 4)
        # 计算条件内积，将条件信息融入判别器输出
        output_y = torch.sum(output * self.linear2(y), 1, keepdim=True)
        output = self.linear1(output) + output_y

        return output.view(-1, 1)

######################################################################################################################
# 测试代码
######################################################################################################################
if __name__ == "__main__":
    
    # 初始化生成器和判别器，并转移到GPU
    netG = sngan_generator(nz=256, dim_embed=128).cuda()
    netD = sngan_discriminator(dim_embed=128).cuda()

    # 定义批次大小为4
    N = 4
    # 随机生成噪声向量（生成器输入）
    z = torch.randn(N, 256).cuda()
    # 随机生成条件向量
    y = torch.randn(N, 128).cuda()
    # 通过生成器生成图像
    x = netG(z, y)
    # 将生成图像与条件向量输入判别器
    o = netD(x, y)
    print(x.size())  # 输出生成图像的尺寸
    print(o.size())  # 输出判别器的结果尺寸

    # 辅助函数：统计网络中的参数数量
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(netG))
    print(get_parameter_number(netD))
