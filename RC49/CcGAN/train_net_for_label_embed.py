import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit
from PIL import Image

"""
_note_

..为什么要训练一个网络去做“图像->标签”的映射？

.1映射本身并非显式可得，需要模型学习
对于简单的场景，也许“标签”可以是图像某些已知的简单统计量(比如平均像素值)。这时候你可能凭经验就能写出一个公式完成映射，不需要神经网络。
但是，在大多数实际深度学习任务中，图像和标签之间关系相当复杂，例如边缘、纹理、形状、轮廓等。这些特征远比肉眼描述的“某些像素 > xx”要复杂得多,没有一个显而易见的显式函数可用。
只能依靠“学习”来让模型在大量样本上逐渐调整参数，找到一个合适的映射。

.2网络的学习能力
神经网络的优势在于它有极强的表示能力，通过大量参数(权重+偏置)可以学习非常灵活且复杂的映射关系。
在训练之后，网络不但能记住训练集中的“图像->标签”关系，还能在新图像上做出合理的预测（即泛化）。

.3保证对未见图像也能预测
如果只是把训练集“图像->标签”硬编码成查表(lookup table)，那确实不需要损失函数，但它无法对“新图像”做任何预测。
训练一个网络的核心目标是学到一套能够对新数据也产生正确映射的规则，这就是深度学习的“泛化”能力所在。

###################################################################################

训练网络的步骤

在深度学习模型中，所谓“学习转化（映射）过程”通常指的是通过反复的前向传播和反向传播来调整网络中的可学习参数（权重和偏置），
从而逐渐逼近理想的输入到输出（图像到标签、或者图像到 embedding）的映射关系。ta是如何“学习”的呢？：

1. 前向传播（Forward Pass）
1.1输入数据
你会将一小批图像（batch）以及它们对应的目标标签（或数值）输入到网络中。

1.2网络推断输出
神经网络（例如卷积神经网络 ResNet 或者自定义结构）包含若干层，层与层之间用可学习的权重和偏置连接。
数据依次经过各层，进行卷积、非线性激活、池化等操作，最后得到一个向量或标量输出，用于与标签对比。

1.3计算损失

将网络输出y^pred与真实标签y^true使用MSELoss计算误差(损失函数)。(MSELoss 是均方误差，即预测值和真实值的平方差的均值。)
这个损失值越大，说明预测和真实值差距越大；损失值越小，表示网络学得更好。

2. 反向传播（Backward Pass）和参数更新

2.1反向传播
得到损失值后，PyTorch 内部会根据“自动求导”机制，对该损失关于网络所有参数的梯度进行计算（这一步就是反向传播）。
反向传播的核心是“链式法则”：从输出层开始，一层层将误差传回去，每一层根据梯度信息知道该怎么修正自己的权重，以使得下次预测更准确。

2.2更新可学习参数
一旦计算出梯度，就可以使用优化器（如 SGD、Adam 等）对网络权重和偏置进行更新。
例如，在 SGD 中，参数更新公式一般为：w←w-η*(∂Loss/∂w)，其中 η 是学习率（learning rate）。
PyTorch 根据你指定的 lr_base（初始学习率）、lr_decay_factor 等参数，动态调整 η 的数值。,

2.3迭代重复
训练过程中，上述 “前向 -> 计算损失 -> 反向传播 -> 更新参数” 的流程会在每个 batch（小批量数据）上反复进行，直至完成全部 epoch（训练轮数）。
随着训练的进行，网络会逐步减少预测误差，从而“学会”如何把图像映射到相应标签或嵌入。

3. 学到什么？（从随机初始化到可用映射）
初始状态是随机的
在训练开始时，网络权重通常是随机初始化的，随机预测自然与真实标签偏差很大，导致损失值也很大。
学习阶段网络自适应调节权重
每次反向传播都会把预测误差的信息“传播”到各层权重，使其朝着“让输出更贴近真实标签”的方向修正。
逐渐逼近目标映射
经过若干 epoch，网络可以找到一种在训练集上较为准确、同时对测试集也有良好泛化的映射方式，即“如何把图像特征投射到和标签/embedding 对应的低维空间上”。

"""
# -------------------------------------------------------------
def train_net_embed(net,net_name,trainloader,testloader,epochs=200,resume_epoch=0,lr_base=0.01,
                    lr_decay_factor=0.1,lr_decay_epochs=[80, 140],weight_decay=1e-4,path_to_ckpt=None
):
    """
    该函数用于训练一个“图像 -> 标签”(或与标签对齐的embedding)的网络, 通过MSELoss来衡量预测结果与真实标签的偏差。
    训练过程中可动态调整学习率, 也支持断点续训和保存中间结果。

    参数释义:
    ----------
    net: 需要训练的网络(例如ResNet变体),
    net_name: 网络名称, 仅用于日志或区分模型,
    trainloader: 包含训练数据的DataLoader,
    testloader: 包含测试数据的DataLoader, 若为None则不进行测试评估,
    epochs: 训练总轮数(默认200),
    resume_epoch: 若从中间某个epoch恢复训练, 传入该epoch的数字(>0),
    lr_base: 初始学习率(默认0.01),
    lr_decay_factor: 学习率每次衰减的因子(默认0.1),
    lr_decay_epochs: 需要衰减学习率的里程碑列表(默认[80,140]),
    weight_decay: 权重衰减系数(默认1e-4),
    path_to_ckpt: 模型检查点的文件夹路径, 若为None则不做保存和加载。

    返回值:
    ----------
    返回训练好的网络net(可能也会在函数内保存到指定path下的pth文件).
    """

    # 辅助函数: 动态调整学习率
    def adjust_learning_rate_1(optimizer, epoch):
        """
        根据当前epoch判断是否需要衰减学习率:
        当epoch >= lr_decay_epochs中的某个值时,
        学习率在原基础上乘以lr_decay_factor。
        """
        lr = lr_base  # 初始化为最初的学习率

        # 对每个衰减里程碑进行检查
        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                # 若当前epoch达到或超过该里程碑, 学习率乘以衰减因子
                lr = lr * lr_decay_factor

        # 将衰减后的学习率设置回optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 1. 将模型迁移到GPU
    net = net.cuda()

    # 2. 定义损失函数和优化器
    #    这里用MSELoss表示希望网络输出的数值与真实标签之间的L2距离最小
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr_base,
        momentum=0.9,           # SGD动量
        weight_decay=weight_decay
    )

    # 3. 如果需要断点续训, 加载此前保存的checkpoint
    if path_to_ckpt is not None and resume_epoch > 0:
        save_file = path_to_ckpt + "/embed_x2y_ckpt_in_train/embed_x2y_checkpoint_epoch_{}.pth".format(resume_epoch)
        checkpoint = torch.load(save_file)
        # 加载模型参数
        net.load_state_dict(checkpoint['net_state_dict'])
        # 加载优化器状态(如动量、学习率等)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 恢复随机数生成器的状态, 以保证训练过程可复现
        torch.set_rng_state(checkpoint['rng_state'])

    # 记录起始时间, 便于计算训练时长
    start_tmp = timeit.default_timer()

    # 4. 进入训练循环, 从resume_epoch开始直到epochs结束
    for epoch in range(resume_epoch, epochs):
        net.train()        # 切换到训练模式, 启用Dropout、BN的统计等
        train_loss = 0.0   # 用于累加当前epoch所有batch的训练损失

        # 根据当前epoch调整学习率
        adjust_learning_rate_1(optimizer, epoch)

        # 逐批训练
        for _, (batch_train_images, batch_train_labels) in enumerate(trainloader):
            # batch_train_images, batch_train_labels 分别是当前批次的图像和标签

            # 将图像/标签转为float并迁移到GPU;
            # 同时将标签reshape为(N, 1)形状, 便于MSELoss计算
            batch_train_images = batch_train_images.type(torch.float).cuda()
            batch_train_labels = batch_train_labels.type(torch.float).view(-1, 1).cuda()

            # 前向传播: net(batch_train_images)返回(outputs, some_features),
            # 其中 outputs通常是对标签的预测或与标签对齐的嵌入
            outputs, _ = net(batch_train_images)

            # 计算损失: MSE(outputs, batch_train_labels)
            loss = criterion(outputs, batch_train_labels)

            # 反向传播并更新参数
            optimizer.zero_grad()  # 梯度清零
            loss.backward()        # 反向传播, 计算梯度
            optimizer.step()       # 参数更新

            # 将当前batch的损失累加到train_loss, 并转移到CPU以避免不必要的显存占用
            train_loss += loss.cpu().item()

        # 计算本epoch的平均训练损失
        train_loss = train_loss / len(trainloader)

        # 5. 如果没有testloader, 只打印训练损失
        if testloader is None:
            print('Train net_x2y for embedding: [epoch %d/%d] train_loss:%f Time:%.4f' % (
                epoch + 1,
                epochs,
                train_loss,
                timeit.default_timer() - start_tmp
            ))
        else:
            # 如果有测试集, 评估测试损失
            net.eval()  # 切换到eval模式(不使用BN统计、Dropout等)
            with torch.no_grad():  # 不追踪梯度, 节省显存加速
                test_loss = 0.0
                for batch_test_images, batch_test_labels in testloader:
                    # 测试阶段同理, 迁移到GPU并计算MSE
                    batch_test_images = batch_test_images.type(torch.float).cuda()
                    batch_test_labels = batch_test_labels.type(torch.float).view(-1, 1).cuda()

                    outputs, _ = net(batch_test_images)
                    loss = criterion(outputs, batch_test_labels)
                    test_loss += loss.cpu().item()

                test_loss = test_loss / len(testloader)

                # 打印当前epoch的训练与测试损失, 以及耗时
                print('Train net_x2y for label embedding: [epoch %d/%d] train_loss:%f test_loss:%f Time:%.4f' % (
                    epoch + 1,
                    epochs,
                    train_loss,
                    test_loss,
                    timeit.default_timer() - start_tmp
                ))

        # 6. 定期保存checkpoint(如每50个epoch一次, 以及最后一个epoch)
        if path_to_ckpt is not None and (((epoch + 1) % 50 == 0) or (epoch + 1 == epochs)):
            save_file = path_to_ckpt + "/embed_x2y_ckpt_in_train/embed_x2y_checkpoint_epoch_{}.pth".format(epoch + 1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)  # 若目录不存在则创建
            torch.save({
                'epoch': epoch,                        # 当前epoch数
                'net_state_dict': net.state_dict(),     # 模型参数
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
                'rng_state': torch.get_rng_state()      # 随机数生成器状态
            }, save_file)
    # end for epoch

    # 训练完成后, 返回训练好的模型(也可能已在中途保存了多次checkpoint)
    return net

    # end for epoch

    return net


###################################################################################
"""""
为什么这样又要训练"标签->embedding"映射？

1. 机器不“理解”标签本身的语义，只能处理数值向量
标签并非“一看就懂”
从人的角度看，“标签 0.7” 或 “类别 3” 可能有明确含义，但对机器而言，其实都是数字符号。若只是做简单的监督训练，网络可以直接输入标签(比如做回归或分类)，的确不需要再映射。
embedding 是神经网络可处理的形式
当系统需要让“图像特征”和“标签特征”在同一个空间中相互对比或计算距离时，就会倾向于把标签也转成某个向量。这样网络就能直接用向量空间的操作（如内积、距离度量）来处理它们。
2. 对齐图像和标签的“隐空间”有更多可能性

    度量学习和相似度检索
如果图像和标签都在同一个向量空间里，你可以很容易计算它们的相似度/距离。
例如：对于某个新标签，想找到与它最接近的已知图像或特征；或反之，根据某些图像特征找与之相邻的标签等。

    插值和生成
有了共同的 embedding 空间，就可以对标签向量做插值，看在这个隐空间里对应哪些图像特征，从而产生新的合成数据。
举个简单例子：如果标签是 0 和 1，可以在 0→1 中间取 0.2, 0.4, 0.6, 0.8 并经过“标签->embedding->图像”过程，得到不同程度的变化。这在传统“标签直接是一个类ID”的方式下并不好办。

    多模态融合
有时还不只是图像和标签，还可能加入文本描述、音频或其他模态。将它们都嵌入到一个公共的高维空间，便于多模态互相检索、对比或生成。

3. 为什么要专门训练“标签->embedding”？

    网络需要“学会”如何对数值标签进行表征
虽然标签本身是一个数字（比如 0.7 或类别索引 3），但要跟图像的高维特征对应起来并非“天生就能明白”。
通过训练，可以让网络学到怎样将标签映射到与图像 embedding 对应的特征点上，比如：某标签 0.7 对应到一个隐藏向量ℎ，并且这个ℎ能和“图像->embedding”网络输出的隐向量对齐或匹配。

    在标签扰动时保持连贯
代码里给标签加噪声再映射，实际上是在训练一个平滑的函数：当标签有所微调（从 0.7 调到 0.75）时，对应的 embedding 也能平滑改变，从而让整个系统更具有“可插值”和“泛化”能力。
如果直接把数字标签“当作向量”用，不做任何学习映射，模型对标签的变化并不敏感，也无法保证在 embedding 空间有足够的表达能力。

    更灵活的后续操作
一旦学好“标签->embedding”，就能在隐空间里做更多操作（插值、生成、检索、多模态对齐等）。这跟“原始数字标签”在某种程度上是两回事。
训练这种映射网络，本质上是在让模型“学到”标签在高维语义上的分布和结构。


"""""
class label_dataset(torch.utils.data.Dataset):
    def __init__(self, labels):
        super(label_dataset, self).__init__()

        self.labels = labels
        self.n_samples = len(self.labels)

    def __getitem__(self, index):
        y = self.labels[index]
        return y

    def __len__(self):
        return self.n_samples


def train_net_y2h(
    unique_labels_norm,  # 归一化后的唯一标签数组，数值范围为[0,1]
    net_y2h,             # 将标签映射到embedding空间的网络
    net_embed,           # 已训练好的 embedding 网络(从图像到embedding)
    epochs=500,          # 训练net_y2h的总轮数(默认500)
    lr_base=0.01,        # 初始学习率
    lr_decay_factor=0.1, # 学习率衰减因子
    lr_decay_epochs=[150, 250, 350], # 哪些epoch进行学习率衰减
    weight_decay=1e-4,   # 权重衰减系数, 用于防止过拟合
    batch_size=128       # 训练batch大小(默认128)
):
    """
    该函数用于训练 "标签 -> embedding" 这个映射网络net_y2h，使得在标签加入噪声后,
    通过 net_y2h 得到的 embedding 再经过 net_embed 中的 h2y 模块映射回标签时,
    能够与原始(带噪声的)标签尽量接近。

    参数说明:
    ----------------
    unique_labels_norm:
        - 一个归一化后的标签数组, 元素范围在[0,1]之间,
        - 这些标签也会在训练时加入一定噪声, 验证模型对扰动的鲁棒性.
    net_y2h:
        - 网络模块, 输入是[0,1]区间的数值标签, 输出是embedding向量(与图像的embedding空间对齐).
    net_embed:
        - 已经训练或加载好的网络(从图像到embedding), 其中包含 h2y 子模块,
          可把embedding映射回标签空间.
    epochs:
        - 训练net_y2h的轮数(默认500).
    lr_base:
        - 初始学习率(默认0.01).
    lr_decay_factor:
        - 学习率衰减因子(默认0.1).
    lr_decay_epochs:
        - 在这些epoch(如[150,250,350])之后, 学习率乘以lr_decay_factor.
    weight_decay:
        - 权重衰减系数(默认1e-4), 用于L2正则化, 缓解过拟合.
    batch_size:
        - 每个训练批次的大小(默认128).

    返回值:
    ----------------
    返回训练完成的 net_y2h, 即"标签->embedding"映射网络.
    """

    # 内部函数: 用于在特定epoch时动态衰减学习率
    def adjust_learning_rate_2(optimizer, epoch):
        """
        根据当前epoch判断是否需要衰减学习率:
        当 epoch >= lr_decay_epochs中的某个值时,
        将学习率乘以lr_decay_factor。
        """
        lr = lr_base
        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 1. 首先确保标签在[0,1]之间(提前断言判断)
    assert np.max(unique_labels_norm) <= 1 and np.min(unique_labels_norm) >= 0

    # 2. 构建用于训练的标签数据集 (label_dataset是自定义的Dataset,只含标签)
    trainset = label_dataset(unique_labels_norm)
    # 通过DataLoader每次随机抓取batch_size个标签
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # 3. 将net_embed设为eval模式, 表示此网络在此处只做前向推断, 不再训练
    net_embed.eval()
    #   net_embed可能是多卡并行(DataParallel)的包装
    #   因此通过 net_embed.module.h2y 获取其内部的 h2y 子模块
    net_h2y = net_embed.module.h2y

    # 4. 构建一个优化器来训练 net_y2h
    optimizer_y2h = torch.optim.SGD(
        net_y2h.parameters(),
        lr=lr_base,
        momentum=0.9,
        weight_decay=weight_decay
    )

    # 记录起始时间以便输出时长
    start_tmp = timeit.default_timer()

    # 5. 开始训练循环
    for epoch in range(epochs):
        net_y2h.train()  # 使 net_y2h 进入训练模式(启用Dropout等)

        train_loss = 0.0  # 用于累加本epoch的训练损失

        # 调整当前epoch的学习率
        adjust_learning_rate_2(optimizer_y2h, epoch)

        # 6. 遍历当前epoch所有批次
        for _, batch_labels in enumerate(trainloader):
            # (a) 数据准备: 将标签搬到GPU, 并reshape为(N,1)
            batch_labels = batch_labels.type(torch.float).view(-1, 1).cuda()

            # (b) 生成随机噪声, 并添加到标签上
            batch_size_curr = len(batch_labels)
            batch_gamma = np.random.normal(0, 0.2, batch_size_curr)  # 均值0,方差0.2的噪声
            batch_gamma = torch.from_numpy(batch_gamma).view(-1, 1).type(torch.float).cuda()

            # 用 clamp 限制到[0,1]区间, 避免噪声后标签越界
            batch_labels_noise = torch.clamp(batch_labels + batch_gamma, 0.0, 1.0)

            # (c) 前向传播:
            #     先用 net_y2h(标签+噪声) -> embedding表征
            batch_hiddens_noise = net_y2h(batch_labels_noise)
            #     再通过 net_h2y(embedding) -> 重建回标签(带噪声的版本)
            batch_rec_labels_noise = net_h2y(batch_hiddens_noise)

            # (d) 计算 MSE 损失: 重建的标签 vs 原始(带噪声的)标签
            loss = nn.MSELoss()(batch_rec_labels_noise, batch_labels_noise)

            # (e) 反向传播与参数更新
            optimizer_y2h.zero_grad()
            loss.backward()
            optimizer_y2h.step()

            # (f) 累加本批次损失到 train_loss
            train_loss += loss.cpu().item()

        # end for batch_idx

        # 计算该epoch的平均损失
        train_loss = train_loss / len(trainloader)

        # 打印日志, 包括当前epoch、平均损失、耗时
        print('\n Train net_y2h: [epoch %d/%d] train_loss:%f Time:%.4f' % (
            epoch + 1,
            epochs,
            train_loss,
            timeit.default_timer() - start_tmp
        ))
    # end for epoch

    # 训练完成后返回训练好的 net_y2h 模型
    return net_y2h
