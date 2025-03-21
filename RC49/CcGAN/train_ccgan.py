# 导入必要的库
import torch                                  # PyTorch，深度学习框架
import numpy as np                            # 用于数值计算的库
import os                                     # 操作系统接口，用于文件和目录操作
import timeit                                 # 用于计时，评估代码运行时间
from PIL import Image                         # 图像处理库，支持打开、操作和保存图像
from torchvision.utils import save_image      # 用于保存生成的图像
import torch.cuda as cutorch                  # CUDA相关操作，可直接调用GPU资源
import torch.nn as nn                         # PyTorch的神经网络模块
# 导入自定义模块和函数
from utils import SimpleProgressBar, IMGs_dataset  # 自定义进度条和数据集加载工具
from opts import parse_opts                        # 自定义参数解析模块，用于获取程序运行参数
from DiffAugment_pytorch import DiffAugment        # 数据增强模块，用于对图像进行DiffAugment增强
from torch.nn import functional as F               # PyTorch的函数模块
''' Settings '''
# 解析命令行参数或配置文件，获取运行时的参数设置
args = parse_opts()

# 从参数中提取GAN相关的参数设置
gan_arch = args.GAN_arch                   # GAN的架构类型
loss_type = args.loss_type_gan             # 损失函数类型（例如 vanilla 或 hinge）
niters = args.niters_gan                   # 训练总迭代次数
resume_niters = args.resume_niters_gan     # 恢复训练时的起始迭代次数
dim_gan = args.dim_gan                     # GAN生成器输入的随机噪声向量的维度
lr_g = args.lr_g_gan                       # 生成器学习率
lr_d = args.lr_d_gan                       # 判别器学习率
save_niters_freq = args.save_niters_freq   # 模型保存的频率（每迭代多少次保存一次）
batch_size_disc = args.batch_size_disc     # 判别器每个批次处理的样本数
batch_size_gene = args.batch_size_gene     # 生成器每个批次生成的样本数
# batch_size_max = max(batch_size_disc, batch_size_gene)  # 可选：计算批次中较大的样本数
num_D_steps = args.num_D_steps             # 每次迭代中判别器的训练步数

visualize_freq = args.visualize_freq       # 可视化生成图像的频率（每迭代多少次保存一次图片）

num_workers = args.num_workers             # 数据加载时的子进程数

threshold_type = args.threshold_type       # 阈值类型（例如 "hard" 或 "soft"）
nonzero_soft_weight_threshold = args.nonzero_soft_weight_threshold  # 软权重非零的阈值

num_channels = args.num_channels           # 图像的通道数，例如RGB图像为3
img_size = args.img_size                   # 图像尺寸（边长）
max_label = args.max_label                 # 标签的最大值

# 是否使用DiffAugment数据增强以及所采用的策略
use_DiffAugment = args.gan_DiffAugment
policy = args.gan_DiffAugment_policy

## 梯度累积设置，用于在内存不足的情况下模拟大批量训练
num_grad_acc_d = args.num_grad_acc_d       # 判别器梯度累积步数
num_grad_acc_g = args.num_grad_acc_g       # 生成器梯度累积步数


## 定义图像归一化函数
def normalize_images(batch_images):
    """
    将输入图像归一化到[-1,1]的范围。
    参数:
        batch_images: 输入的图像数组（假设像素值范围为0-255）
    返回:
        归一化后的图像张量
    """
    batch_images = batch_images / 255.0             # 先归一化到[0,1]
    batch_images = (batch_images - 0.5) / 0.5         # 再映射到[-1,1]
    return batch_images

class RKDLoss(nn.Module):
    """Relational Knowledge Distillation, 来自 CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d  # distance 项权重
        self.w_a = w_a  # angle   项权重

    def forward(self, f_s, f_t):
        # 先把学生和教师特征打平到 (batch_size, num_features)
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # 1. 计算教师网络特征两两之间的 pairwise distance，并做归一化
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)   # pairwise dist
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # 2. 计算教师网络特征之间的两两向量角度
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            # 两个向量之间的内积可以反映角度的余弦值
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        # 最终 RKD 的 loss 同时考虑距离和角度
        loss = self.w_d * loss_d + self.w_a * loss_a
        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        """
        计算向量 e 的两两距离。若 squared=True 则返回平方距离，否则返回欧式距离。
        """
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        # 对角线元素(自身距离自己)重置为0
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

## 定义CCGAN（条件连续GAN）训练函数
def train_ccgan_with_kd(kernel_sigma, kappa, train_images, train_labels,netG, netD,
                        netG_student, KD_rate,net_y2h, save_images_folder,
                        save_models_folder=None,clip_label=False):
    """
    训练条件连续GAN的函数
    参数:
        kernel_sigma: 高斯核标准差，用于对标签添加噪声
        kappa: 标签附近的窗口宽度参数，用于选择真实样本
        train_images: 训练图像数据（未归一化到[-1,1]）
        train_labels: 与图像对应的标签（连续值）
        netG: 生成器网络
        netD: 判别器网络
        netD_student: 学生判别器网络
        KD_rate: KD损失的权重
        net_y2h: 标签到隐向量映射网络，用于将标签转换为隐含特征
        save_images_folder: 保存生成图像的文件夹路径
        save_models_folder: 模型保存的文件夹路径（可选）
        clip_label: 是否对标签进行裁剪（将标签限制在一定范围内）
    返回:
        训练后的生成器和判别器
    """
    # 将网络转移到GPU
    netG = netG.cuda()
    netD = netD.cuda()
    net_y2h = net_y2h.cuda()
    net_y2h.eval()  # 映射网络在训练过程中不更新参数

    # ---- 教师生成器固定，不更新参数 ----
    netG.eval()
    for param in netG.parameters():
        param.requires_grad = False

    # ---- 学生生成器要训练 ----
    netG_student = netG_student.cuda()
    netG_student.train()

    # 优化器：判别器和学生生成器各自一个
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizerG_s = torch.optim.Adam(netG_student.parameters(), lr=lr_g, betas=(0.5, 0.999))

    criterion_mse = nn.MSELoss()

    # 如果提供了模型保存文件夹，并且设置了恢复训练的起始迭代次数，则加载模型和优化器状态
    if save_models_folder is not None and resume_niters > 0:
        save_file = save_models_folder + "/ckpts_in_train/ckpt_niter_{}.pth".format(resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        #optimizerG_s.load_state_dict(checkpoint['optimizerG_s_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    # end if

    #################
    # 获取训练集中所有独一无二的标签，并进行排序
    unique_train_labels = np.sort(np.array(list(set(train_labels))))

    # 为了可视化，选取标签分布中第5百分位到第95百分位之间的标签
    n_row = 10  # 可视化图像的行数
    n_col = n_row  # 可视化图像的列数
    z_fixed = torch.randn(n_row * n_col, dim_gan, dtype=torch.float).cuda()  # 固定的噪声向量，用于生成固定图像便于比较
    start_label = np.quantile(train_labels, 0.05)  # 计算第5百分位标签
    end_label = np.quantile(train_labels, 0.95)    # 计算第95百分位标签
    selected_labels = np.linspace(start_label, end_label, num=n_row)  # 在线性区间内均匀选取标签
    y_fixed = np.zeros(n_row * n_col)  # 初始化固定标签数组
    # 对每一行填充相同的标签，方便观察生成器对不同标签的生成效果
    for i in range(n_row):
        curr_label = selected_labels[i]
        for j in range(n_col):
            y_fixed[i * n_col + j] = curr_label
    print(y_fixed)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1, 1).cuda()

    start_time = timeit.default_timer()  # 记录训练开始时间
    # 主训练循环，从resume_niters开始迭代到总迭代次数niters
    for niter in range(resume_niters, niters):

        ''' 训练判别器 '''
        # 对于每次迭代，判别器可能需要多次更新
        for step_D_index in range(num_D_steps):

            optimizerD.zero_grad()  # 清空判别器的梯度

            # 梯度累积，分多步计算梯度再反向传播
            for accumulation_index in range(num_grad_acc_d):

                ## 随机从训练标签中抽取一批样本的标签（大小为batch_size_disc）
                batch_target_labels_in_dataset = np.random.choice(unique_train_labels, size=batch_size_disc, replace=True)
                ## 对抽取的标签添加高斯噪声，用于估计条件图像分布
                batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_disc)
                batch_target_labels = batch_target_labels_in_dataset + batch_epsilons

                ## 根据添加噪声后的标签，在训练集里寻找标签接近的真实样本，并同时生成用于伪造图像的标签
                batch_real_indx = np.zeros(batch_size_disc, dtype=int)  # 存储满足条件的真实图像在数据集中的索引
                batch_fake_labels = np.zeros(batch_size_disc)             # 用于生成伪造图像的标签

                # 对于每个样本，寻找合适的真实图像索引和对应的伪造标签
                for j in range(batch_size_disc):
                    ## 根据阈值类型（硬阈值或软阈值）确定标签的邻域
                    if threshold_type == "hard":
                        indx_real_in_vicinity = np.where(np.abs(train_labels - batch_target_labels[j]) <= kappa)[0]
                    else:
                        # 对于soft阈值，使用一个反转的权重函数来确定符合条件的样本
                        indx_real_in_vicinity = np.where((train_labels - batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold) / kappa)[0]

                    ## 如果当前噪声导致没有找到任何真实样本（例如当标签间隔较大时），则重新生成噪声
                    while len(indx_real_in_vicinity) < 1:
                        batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                        batch_target_labels[j] = batch_target_labels_in_dataset[j] + batch_epsilons_j
                        if clip_label:
                            batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)
                        ## 重新根据阈值类型选择邻域内的真实样本
                        if threshold_type == "hard":
                            indx_real_in_vicinity = np.where(np.abs(train_labels - batch_target_labels[j]) <= kappa)[0]
                        else:
                            indx_real_in_vicinity = np.where((train_labels - batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold) / kappa)[0]
                    # end while

                    # 确保至少找到了一个样本
                    assert len(indx_real_in_vicinity) >= 1

                    # 从邻域内随机选择一个真实样本的索引
                    batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]

                    ## 为伪造图像生成标签，标签取值在(batch_target_labels[j]-kappa, batch_target_labels[j]+kappa)范围内，
                    ## 若为soft阈值，则使用相应的区间
                    if threshold_type == "hard":
                        lb = batch_target_labels[j] - kappa
                        ub = batch_target_labels[j] + kappa
                    else:
                        lb = batch_target_labels[j] - np.sqrt(-np.log(nonzero_soft_weight_threshold) / kappa)
                        ub = batch_target_labels[j] + np.sqrt(-np.log(nonzero_soft_weight_threshold) / kappa)
                    lb = max(0.0, lb); ub = min(ub, 1.0)  # 保证标签在[0,1]范围内
                    assert lb <= ub
                    assert lb >= 0 and ub >= 0
                    assert lb <= 1 and ub <= 1
                    batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
                # end for j

                ## 根据上面选定的真实样本索引，从训练集中提取真实图像和对应的标签
                batch_real_images = torch.from_numpy(normalize_images(train_images[batch_real_indx]))
                batch_real_images = batch_real_images.type(torch.float).cuda()
                batch_real_labels = train_labels[batch_real_indx]
                batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).cuda()

                ## 将生成伪造图像时用到的标签转换为张量
                batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).cuda()
                # 生成随机噪声，作为生成器的输入
                z = torch.randn(batch_size_disc, dim_gan, dtype=torch.float).cuda()
                # 使用学生生成器生成伪造图像
                batch_fake_images = netG_student(z, net_y2h(batch_fake_labels), return_features=False)

                ## 将批次目标标签转换为张量，作为判别器的条件输入
                batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).cuda()

                ## 计算权重向量：若使用soft阈值，则对真实和伪造标签计算指数衰减权重，否则均为1
                if threshold_type == "soft":
                    real_weights = torch.exp(-kappa * (batch_real_labels - batch_target_labels)**2).cuda()
                    fake_weights = torch.exp(-kappa * (batch_fake_labels - batch_target_labels)**2).cuda()
                else:
                    real_weights = torch.ones(batch_size_disc, dtype=torch.float).cuda()
                    fake_weights = torch.ones(batch_size_disc, dtype=torch.float).cuda()
                # end if threshold type

                # forward前向传播判别器
                if use_DiffAugment:
                    # 使用DiffAugment对图像进行数据增强后再送入判别器
                    real_dis_out = netD(DiffAugment(batch_real_images, policy=policy), net_y2h(batch_target_labels))
                    fake_dis_out = netD(DiffAugment(batch_fake_images.detach(), policy=policy), net_y2h(batch_target_labels))
                else:
                    real_dis_out = netD(batch_real_images, net_y2h(batch_target_labels))
                    fake_dis_out = netD(batch_fake_images.detach(), net_y2h(batch_target_labels))

                # 根据损失类型计算判别器的损失
                if loss_type == "vanilla":
                    # vanilla GAN的交叉熵损失
                    real_dis_out = torch.nn.Sigmoid()(real_dis_out)
                    fake_dis_out = torch.nn.Sigmoid()(fake_dis_out)
                    d_loss_real = - torch.log(real_dis_out + 1e-20)
                    d_loss_fake = - torch.log(1 - fake_dis_out + 1e-20)
                elif loss_type == "hinge":
                    # hinge损失
                    d_loss_real = torch.nn.ReLU()(1.0 - real_dis_out)
                    d_loss_fake = torch.nn.ReLU()(1.0 + fake_dis_out)
                else:
                    raise ValueError('Not supported loss type!!!')

                # 将真实样本损失和伪造样本损失按照权重进行加权平均，并考虑梯度累积步数
                d_loss = torch.mean(real_weights.view(-1) * d_loss_real.view(-1)) + torch.mean(fake_weights.view(-1) * d_loss_fake.view(-1)) / float(num_grad_acc_d)

                d_loss.backward()  # 反向传播计算梯度
            # end for accumulation_index

            optimizerD.step()  # 更新判别器参数

        # end for step_D_index



        ''' 训练生成器 '''
        netG_student.train()
        optimizerG_s.zero_grad()

        # 梯度累积，分多步计算生成器的梯度
        for accumulation_index in range(num_grad_acc_g):

            # 生成伪造图像的标签，类似于判别器的处理
            ## 随机抽取一批标签，并添加高斯噪声
            batch_target_labels_in_dataset = np.random.choice(unique_train_labels,
                                                              size=batch_size_gene,
                                                              replace=True)
            batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_gene)
            batch_target_labels = batch_target_labels_in_dataset + batch_epsilons
            batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).cuda()

            # 生成随机噪声作为输入
            z = torch.randn(batch_size_gene, dim_gan, dtype=torch.float).cuda()
            # 1) 用教师网络得到 teacher_out, teacher_feats
            with torch.no_grad():
                teacher_out, teacher_feats = netG(z, net_y2h(batch_target_labels),
                                                  return_features=True)

            # 2) 用学生网络得到 student_out, student_feats
            # 生成学生输出及特征用于 KD 损失计算
            student_out, student_feats = netG_student(
            z, net_y2h(batch_target_labels), return_features=True)


            # 3) GAN损失 (看你用 hinge 还是 vanilla)
            dis_out = netD(student_out, net_y2h(batch_target_labels))
            if loss_type == "vanilla":
                dis_out = torch.sigmoid(dis_out)
                g_loss = -torch.mean(torch.log(dis_out+1e-20))
            elif loss_type == "hinge":
                g_loss = -dis_out.mean()

        # 4) 计算RKD损失
        rkd_loss_total = 0.0
        layer_weights = [1, 2, 3, 4]  # 各层权重可调整

        # 遍历各层特征 (假设teacher_feats和student_feats层数相同)
        for layer_idx in range(len(teacher_feats)):
            # 教师特征:
            teacher_feats[layer_idx]
            # 学生特征:
            student_feats[layer_idx]

            # 展平特征图 (保留batch维度)
            t_feat = teacher_feats[layer_idx].flatten(start_dim=1)
            s_feat = student_feats[layer_idx].flatten(start_dim=1)

            # 计算RKD损失并加权
            rkd_loss_layer = RKDLoss(w_d=25, w_a=50)(s_feat, t_feat)
            rkd_loss_total += layer_weights[layer_idx] * rkd_loss_layer

        # 5) 合并损失
        g_loss_total = g_loss + KD_rate * rkd_loss_total
        g_loss_total = g_loss_total / float(num_grad_acc_g)
        g_loss_total.backward()

        # 生成伪造图像
        batch_fake_images = netG_student(z, net_y2h(batch_target_labels), return_features=False)
        # end for accumulation_index

        optimizerG_s.step()  # 更新生成器参数

        # 每20次迭代打印一次训练状态，包括损失、真实和伪造样本的判别器输出平均值以及消耗时间
        if (niter + 1) % 20 == 0:
            print("CcGAN,%s: [Iter %d/%d] [RKD loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4f]" %
                  (gan_arch, niter + 1, niters, rkd_loss_total.item(), g_loss.item(),
                   real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer() - start_time))

        # 按设定频率进行生成图像的可视化
        if (niter + 1) % visualize_freq == 0:
            netG.eval()  # 设置生成器为评估模式
            with torch.no_grad():
                gen_imgs = netG_student(z_fixed, net_y2h(y_fixed),return_features=False)
                gen_imgs = gen_imgs.detach().cpu()
                # 保存生成的图像到指定文件夹中，n_row参数用于控制图片拼接的行数
                save_image(gen_imgs.data, save_images_folder + '/{}.png'.format(niter + 1), nrow=n_row, normalize=True)

        # 根据设定频率保存模型的参数和优化器状态
        if save_models_folder is not None and ((niter + 1) % save_niters_freq == 0 or (niter + 1) == niters):
            save_file = save_models_folder + "/ckpts_in_train/ckpt_niter_{}.pth".format(niter + 1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG_s.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'rng_state': torch.get_rng_state()
            }, save_file)
    # end for niter
    return netG_student

## 定义根据给定标签采样伪造图像的函数
def sample_ccgan_given_labels(netG, net_y2h, labels, batch_size=500, to_numpy=True, denorm=True, verbose=True):
    """
    使用预训练生成器根据给定的标签采样生成图像
    参数:
        netG: 预训练生成器网络
        net_y2h: 标签到隐向量映射网络
        labels: 目标标签数组（浮点数，归一化后的标签）
        batch_size: 每个批次生成图像的数量,默认500
        to_numpy: 是否将生成的图像转换为numpy数组
        denorm: 是否对生成的图像进行反归一化操作（将[-1,1]范围转换回[0,255]），用于节省内存
        verbose: 是否显示进度条
    返回:
        fake_images: 生成的图像数组
        fake_labels: 对应的标签数组
    """
    nfake = len(labels)  # 需要生成的图像总数
    if batch_size > nfake:
        batch_size = nfake

    fake_images = []
    # 为方便批次生成，将labels扩充，防止最后一个批次不足batch_size
    fake_labels = np.concatenate((labels, labels[0:batch_size]))
    netG = netG.cuda()
    netG.eval()  # 设置生成器为评估模式
    net_y2h = net_y2h.cuda()
    net_y2h.eval()
    with torch.no_grad():
        if verbose:
            pb = SimpleProgressBar()  # 初始化自定义进度条
        n_img_got = 0
        while n_img_got < nfake:
            # 每次生成一批随机噪声作为输入
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).cuda()
            # 从fake_labels中取出当前批次对应的标签，并转换为张量
            y = torch.from_numpy(fake_labels[n_img_got:(n_img_got + batch_size)]).type(torch.float).view(-1, 1).cuda()
            # 生成图像
            batch_fake_images = netG(z, net_y2h(y),return_features=False)
            if denorm:  # 如果需要反归一化，转换生成图像到[0,255]
                # 断言生成的图像值在[-1,1]范围内
                assert batch_fake_images.max().item() <= 1.0 and batch_fake_images.min().item() >= -1.0
                batch_fake_images = batch_fake_images * 0.5 + 0.5  # 映射到[0,1]
                batch_fake_images = batch_fake_images * 255.0      # 映射到[0,255]
                batch_fake_images = batch_fake_images.type(torch.uint8)
                # 可选：可以添加断言检查转换后的图像是否正确
            fake_images.append(batch_fake_images.cpu())
            n_img_got += batch_size
            if verbose:
                pb.update(min(float(n_img_got) / nfake, 1) * 100)
        # end while

    # 将所有批次拼接在一起
    fake_images = torch.cat(fake_images, dim=0)
    # 去除多余生成的样本，使生成图像数量精确等于nfake
    fake_images = fake_images[0:nfake]
    fake_labels = fake_labels[0:nfake]

    if to_numpy:
        fake_images = fake_images.numpy()  # 转换为numpy数组便于后续处理
    else:
        fake_labels = torch.from_numpy(fake_labels).type(torch.float)

    return fake_images, fake_labels

def train_ccgan(kernel_sigma, kappa, train_images, train_labels, netG, netD, net_y2h, save_images_folder, save_models_folder=None, clip_label=False):
    """
    训练条件连续GAN的函数
    参数:
        kernel_sigma: 高斯核标准差，用于对标签添加噪声
        kappa: 标签附近的窗口宽度参数，用于选择真实样本
        train_images: 训练图像数据（未归一化到[-1,1]）
        train_labels: 与图像对应的标签（连续值）
        netG: 生成器网络
        netD: 判别器网络
        net_y2h: 标签到隐向量映射网络，用于将标签转换为隐含特征
        save_images_folder: 保存生成图像的文件夹路径
        save_models_folder: 模型保存的文件夹路径（可选）
        clip_label: 是否对标签进行裁剪（将标签限制在一定范围内）
    返回:
        训练后的生成器和判别器
    """
    # 将网络转移到GPU
    netG = netG.cuda()
    netD = netD.cuda()
    net_y2h = net_y2h.cuda()
    net_y2h.eval()  # 映射网络在训练过程中不更新参数

    # 定义生成器和判别器的优化器（Adam优化器）
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # 如果提供了模型保存文件夹，并且设置了恢复训练的起始迭代次数，则加载模型和优化器状态
    if save_models_folder is not None and resume_niters > 0:
        save_file = save_models_folder + "/ckpts_in_train/ckpt_niter_{}.pth".format(resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    # end if

    #################
    # 获取训练集中所有独一无二的标签，并进行排序
    unique_train_labels = np.sort(np.array(list(set(train_labels))))

    # 为了可视化，选取标签分布中第5百分位到第95百分位之间的标签
    n_row = 10  # 可视化图像的行数
    n_col = n_row  # 可视化图像的列数
    z_fixed = torch.randn(n_row * n_col, dim_gan, dtype=torch.float).cuda()  # 固定的噪声向量，用于生成固定图像便于比较
    start_label = np.quantile(train_labels, 0.05)  # 计算第5百分位标签
    end_label = np.quantile(train_labels, 0.95)    # 计算第95百分位标签
    selected_labels = np.linspace(start_label, end_label, num=n_row)  # 在线性区间内均匀选取标签
    y_fixed = np.zeros(n_row * n_col)  # 初始化固定标签数组
    # 对每一行填充相同的标签，方便观察生成器对不同标签的生成效果
    for i in range(n_row):
        curr_label = selected_labels[i]
        for j in range(n_col):
            y_fixed[i * n_col + j] = curr_label
    print(y_fixed)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1, 1).cuda()

    start_time = timeit.default_timer()  # 记录训练开始时间
    # 主训练循环，从resume_niters开始迭代到总迭代次数niters
    for niter in range(resume_niters, niters):

        ''' 训练判别器 '''
        # 对于每次迭代，判别器可能需要多次更新
        for step_D_index in range(num_D_steps):

            optimizerD.zero_grad()  # 清空判别器的梯度

            # 梯度累积，分多步计算梯度再反向传播
            for accumulation_index in range(num_grad_acc_d):

                ## 随机从训练标签中抽取一批样本的标签（大小为batch_size_disc）
                batch_target_labels_in_dataset = np.random.choice(unique_train_labels, size=batch_size_disc, replace=True)
                ## 对抽取的标签添加高斯噪声，用于估计条件图像分布
                batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_disc)
                batch_target_labels = batch_target_labels_in_dataset + batch_epsilons


                ## 根据添加噪声后的标签，在训练集里寻找标签接近的真实样本，并同时生成用于伪造图像的标签
                batch_real_indx = np.zeros(batch_size_disc, dtype=int)  # 存储满足条件的真实图像在数据集中的索引
                batch_fake_labels = np.zeros(batch_size_disc)             # 用于生成伪造图像的标签

                # 对于每个样本，寻找合适的真实图像索引和对应的伪造标签
                for j in range(batch_size_disc):
                    ## 根据阈值类型（硬阈值或软阈值）确定标签的邻域
                    if threshold_type == "hard":
                        indx_real_in_vicinity = np.where(np.abs(train_labels - batch_target_labels[j]) <= kappa)[0]
                    else:
                        # 对于soft阈值，使用一个反转的权重函数来确定符合条件的样本
                        indx_real_in_vicinity = np.where((train_labels - batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold) / kappa)[0]

                    ## 如果当前噪声导致没有找到任何真实样本（例如当标签间隔较大时），则重新生成噪声
                    while len(indx_real_in_vicinity) < 1:
                        batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                        batch_target_labels[j] = batch_target_labels_in_dataset[j] + batch_epsilons_j
                        if clip_label:
                            batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)
                        ## 重新根据阈值类型选择邻域内的真实样本
                        if threshold_type == "hard":
                            indx_real_in_vicinity = np.where(np.abs(train_labels - batch_target_labels[j]) <= kappa)[0]
                        else:
                            indx_real_in_vicinity = np.where((train_labels - batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold) / kappa)[0]
                    # end while

                    # 确保至少找到了一个样本
                    assert len(indx_real_in_vicinity) >= 1

                    # 从邻域内随机选择一个真实样本的索引
                    batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]

                    ## 为伪造图像生成标签，标签取值在(batch_target_labels[j]-kappa, batch_target_labels[j]+kappa)范围内，
                    ## 若为soft阈值，则使用相应的区间
                    if threshold_type == "hard":
                        lb = batch_target_labels[j] - kappa
                        ub = batch_target_labels[j] + kappa
                    else:
                        lb = batch_target_labels[j] - np.sqrt(-np.log(nonzero_soft_weight_threshold) / kappa)
                        ub = batch_target_labels[j] + np.sqrt(-np.log(nonzero_soft_weight_threshold) / kappa)
                    lb = max(0.0, lb); ub = min(ub, 1.0)  # 保证标签在[0,1]范围内
                    assert lb <= ub
                    assert lb >= 0 and ub >= 0
                    assert lb <= 1 and ub <= 1
                    batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
                # end for j

                ## 根据上面选定的真实样本索引，从训练集中提取真实图像和对应的标签
                batch_real_images = torch.from_numpy(normalize_images(train_images[batch_real_indx]))
                batch_real_images = batch_real_images.type(torch.float).cuda()
                batch_real_labels = train_labels[batch_real_indx]
                batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).cuda()

                ## 将生成伪造图像时用到的标签转换为张量
                batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).cuda()
                # 生成随机噪声，作为生成器的输入
                z = torch.randn(batch_size_disc, dim_gan, dtype=torch.float).cuda()
                # 使用生成器生成伪造图像，条件是通过net_y2h对标签进行映射后的结果
                batch_fake_images = netG(z, net_y2h(batch_fake_labels))

                ## 将批次目标标签转换为张量，作为判别器的条件输入
                batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).cuda()

                ## 计算权重向量：若使用soft阈值，则对真实和伪造标签计算指数衰减权重，否则均为1
                if threshold_type == "soft":
                    real_weights = torch.exp(-kappa * (batch_real_labels - batch_target_labels)**2).cuda()
                    fake_weights = torch.exp(-kappa * (batch_fake_labels - batch_target_labels)**2).cuda()
                else:
                    real_weights = torch.ones(batch_size_disc, dtype=torch.float).cuda()
                    fake_weights = torch.ones(batch_size_disc, dtype=torch.float).cuda()
                # end if threshold type

                # forward前向传播判别器
                if use_DiffAugment:
                    # 使用DiffAugment对图像进行数据增强后再送入判别器
                    real_dis_out = netD(DiffAugment(batch_real_images, policy=policy), net_y2h(batch_target_labels))
                    fake_dis_out = netD(DiffAugment(batch_fake_images.detach(), policy=policy), net_y2h(batch_target_labels))
                else:
                    real_dis_out = netD(batch_real_images, net_y2h(batch_target_labels))
                    fake_dis_out = netD(batch_fake_images.detach(), net_y2h(batch_target_labels))

                # 根据损失类型计算判别器的损失
                if loss_type == "vanilla":
                    # vanilla GAN的交叉熵损失
                    real_dis_out = torch.nn.Sigmoid()(real_dis_out)
                    fake_dis_out = torch.nn.Sigmoid()(fake_dis_out)
                    d_loss_real = - torch.log(real_dis_out + 1e-20)
                    d_loss_fake = - torch.log(1 - fake_dis_out + 1e-20)
                elif loss_type == "hinge":
                    # hinge损失
                    d_loss_real = torch.nn.ReLU()(1.0 - real_dis_out)
                    d_loss_fake = torch.nn.ReLU()(1.0 + fake_dis_out)
                else:
                    raise ValueError('Not supported loss type!!!')

                # 将真实样本损失和伪造样本损失按照权重进行加权平均，并考虑梯度累积步数
                d_loss = torch.mean(real_weights.view(-1) * d_loss_real.view(-1)) + torch.mean(fake_weights.view(-1) * d_loss_fake.view(-1)) / float(num_grad_acc_d)

                d_loss.backward()  # 反向传播计算梯度
            # end for accumulation_index

            optimizerD.step()  # 更新判别器参数

        # end for step_D_index



        ''' 训练生成器 '''
        netG.train()

        optimizerG.zero_grad()  # 清空生成器梯度

        # 梯度累积，分多步计算生成器的梯度
        for accumulation_index in range(num_grad_acc_g):

            # 生成伪造图像的标签，类似于判别器的处理
            ## 随机抽取一批标签，并添加高斯噪声
            batch_target_labels_in_dataset = np.random.choice(unique_train_labels, size=batch_size_gene, replace=True)
            batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_gene)
            batch_target_labels = batch_target_labels_in_dataset + batch_epsilons
            batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).cuda()

            # 生成随机噪声作为输入
            z = torch.randn(batch_size_gene, dim_gan, dtype=torch.float).cuda()
            # 生成伪造图像
            batch_fake_images = netG(z, net_y2h(batch_target_labels))

            # 计算生成器的损失：判别器的输出作为依据
            if use_DiffAugment:
                dis_out = netD(DiffAugment(batch_fake_images, policy=policy), net_y2h(batch_target_labels))
            else:
                dis_out = netD(batch_fake_images, net_y2h(batch_target_labels))
            if loss_type == "vanilla":
                dis_out = torch.nn.Sigmoid()(dis_out)
                g_loss = - torch.mean(torch.log(dis_out + 1e-20))
            elif loss_type == "hinge":
                g_loss = - dis_out.mean()

            g_loss = g_loss / float(num_grad_acc_g)

            # 反向传播计算生成器梯度
            g_loss.backward()

        # end for accumulation_index

        optimizerG.step()  # 更新生成器参数

        # 每20次迭代打印一次训练状态，包括损失、真实和伪造样本的判别器输出平均值以及消耗时间
        if (niter + 1) % 20 == 0:
            print("CcGAN,%s: [Iter %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4f]" %
                  (gan_arch, niter + 1, niters, d_loss.item(), g_loss.item(),
                   real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer() - start_time))

        # 按设定频率进行生成图像的可视化
        if (niter + 1) % visualize_freq == 0:
            netG.eval()  # 设置生成器为评估模式
            with torch.no_grad():
                gen_imgs = netG(z_fixed, net_y2h(y_fixed))
                gen_imgs = gen_imgs.detach().cpu()
                # 保存生成的图像到指定文件夹中，n_row参数用于控制图片拼接的行数
                save_image(gen_imgs.data, save_images_folder + '/{}.png'.format(niter + 1), nrow=n_row, normalize=True)

        # 根据设定频率保存模型的参数和优化器状态
        if save_models_folder is not None and ((niter + 1) % save_niters_freq == 0 or (niter + 1) == niters):
            save_file = save_models_folder + "/ckpts_in_train/ckpt_niter_{}.pth".format(niter + 1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'rng_state': torch.get_rng_state()
            }, save_file)
    # end for niter
    return netG, netD