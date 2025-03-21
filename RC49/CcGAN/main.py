
print("\n===================================================================================================")

import copy  # 用于深拷贝数据

import matplotlib.pyplot as plt  # 绘图工具
import numpy as np  # 数值计算库

plt.switch_backend('agg')           # 后端设置为无界面模式，便于在服务器上绘图
import h5py             # 用于读取HDF5格式数据文件
import os               # 操作系统相关操作，如路径、目录等
import random           # 随机数生成
from tqdm import tqdm  # 进度条显示工具
import torch.backends.cudnn as cudnn  # CUDA相关设置
import timeit           # 计时模块
from PIL import Image   # 图像处理库

# ================================================================================
# 导入自定义模块和函数
from opts import parse_opts     # 解析命令行参数的自定义模块
args = parse_opts()             # 获取参数
wd = args.root_path             # 设置工作目录为参数指定的路径
os.chdir(wd)                    # 切换当前工作目录
from utils import IMGs_dataset, compute_entropy, predict_class_labels  # 自定义数据集加载、熵计算和标签预测函数
from models import *            # 导入所有定义好的网络模型
from train_ccgan import train_ccgan, sample_ccgan_given_labels,train_ccgan_with_kd  # 导入条件GAN训练与采样函数
from train_net_for_label_embed import train_net_embed, train_net_y2h  # 导入标签嵌入网络训练函数
from eval_metrics import cal_FID, cal_labelscore, inception_score  # 导入FID、标签得分、Inception Score等评价指标计算函数
from models.CcGAN_SAGAN import CcGAN_SAGAN_Generator, CcGAN_SAGAN_Discriminator, CcGAN_SAGAN_Generator_Student  # 导入CcGAN的生成器和判别器
# ================================================================================
'''                                   Settings                                      '''
# ================================================================================
# -------------------------------
# 设置随机种子，保证实验结果的可复现性
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True  # 保证每次运行结果一致，但可能牺牲一定效率
cudnn.benchmark = False
np.random.seed(args.seed)

# -------------------------------
# 嵌入网络的基础学习率设置
base_lr_x2y = 0.01  # 用于训练从图像到嵌入向量的网络（x2y）
base_lr_y2h = 0.01  # 用于训练将标签映射到嵌入空间的网络（y2h）

# -------------------------------
# 采样参数设置
assert args.eval_mode in [1,2,3,4]  # 评估模式必须为1,2,3或4
if args.data_split == "all":
    args.eval_mode != 1  # 如果数据集划分为全部数据，则eval_mode不能为1

# -------------------------------
# 定义标签归一化和反归一化的函数
def fn_norm_labels(labels):
    '''
    输入：未归一化的标签
    返回：归一化后的标签（除以最大标签值）
    '''
    return labels / args.max_label

def fn_denorm_labels(labels):
    '''
    输入：归一化后的标签
    返回：反归一化后的标签（乘以最大标签值）
    '''
    if isinstance(labels, np.ndarray):
        return labels * args.max_label
    elif torch.is_tensor(labels):
        return labels * args.max_label
    else:
        return labels * args.max_label

# ================================================================================
'''                                Data loader                                    '''
# ================================================================================
# 构造数据文件的完整路径，文件名格式为RC-49_<img_size>x<img_size>.h5
data_filename = args.data_path + '/RC-49_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')  # 打开HDF5文件，读取模式
labels_all = hf['labels'][:]  # 读取所有标签
labels_all = labels_all.astype(float)  # 转换为浮点数
images_all = hf['images'][:]  # 读取所有图像
indx_train = hf['indx_train'][:]  # 训练集的索引
hf.close()  # 关闭文件
print("\n RC-49 dataset shape: {}x{}x{}x{}".format(images_all.shape[0], images_all.shape[1], images_all.shape[2], images_all.shape[3]))

# 数据集划分：根据参数data_split来确定使用全部数据还是仅使用训练集数据
if args.data_split == "train":
    images_train = images_all[indx_train]
    labels_train_raw = labels_all[indx_train]
else:
    images_train = copy.deepcopy(images_all)
    labels_train_raw = copy.deepcopy(labels_all)

# 筛选标签在(q1, q2)范围内的图像和标签
q1 = args.min_label
q2 = args.max_label
indx = np.where((labels_train_raw > q1) * (labels_train_raw < q2) == True)[0]
labels_train_raw = labels_train_raw[indx]
images_train = images_train[indx]
assert len(labels_train_raw) == len(images_train)

#images_all->全部图像，labels_all->全部标签（未归一化）
#images_train->训练集图像，labels_train_raw->训练集标签（未归一化）
#labels_train->训练集标签（归一化）

# 如果需要计算FID，则对所有数据进行类似的筛选
if args.comp_FID:
    indx = np.where((labels_all > q1) * (labels_all < q2) == True)[0]
    labels_all = labels_all[indx]
    images_all = images_all[indx]
    assert len(labels_all) == len(images_all)
    print(labels_all)



# 针对每个角度（标签值）最多保留args.max_num_img_per_label张图像
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each angle, take no more than {} images>>>".format(len(images_train), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels_train_raw))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels_train_raw == unique_labels_tmp[i])[0]
    if len(indx_i) > image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images_train = images_train[sel_indx]
labels_train_raw = labels_train_raw[sel_indx]
print("{} images left and there are {} unique labels".format(len(images_train), len(set(labels_train_raw))))

# 显示归一化前标签的范围
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels_train_raw), np.max(labels_train_raw)))
# 对训练标签进行归一化
labels_train = fn_norm_labels(labels_train_raw)
print(labels_train.shape)
print('**************************:::::',labels_train[0:200:20])
print("\n Range of normalized labels: ({},{})".format(np.min(labels_train), np.max(labels_train)))

# 获取归一化后的唯一标签集合
unique_labels_norm = np.sort(np.array(list(set(labels_train))))
print(">>>>>>>>>>>>>>>><<<<<<<><><><><>>>>>>>>>>",unique_labels_norm)

# 如果kernel_sigma小于0，则使用经验公式计算其值
if args.kernel_sigma < 0:
    std_label = np.std(labels_train)
    args.kernel_sigma = 1.06 * std_label * (len(labels_train))**(-1/5)
    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\n The std of {} labels is {} so the kernel sigma is {}".format(len(labels_train), std_label, args.kernel_sigma))
# 如果kappa小于0，则根据标签间隔计算基础kappa
if args.kappa < 0:
    n_unique = len(unique_labels_norm)
    diff_list = []
    for i in range(1, n_unique):
        diff_list.append(unique_labels_norm[i] - unique_labels_norm[i-1])
    kappa_base = np.abs(args.kappa) * np.max(np.array(diff_list))
    if args.threshold_type == "hard":
        args.kappa = kappa_base
    else:
        args.kappa = 1 / kappa_base**2

# ================================================================================
'''                                Output folders                                 '''
# ================================================================================
# 定义输出路径，生成文件夹用于保存训练过程中的模型和图像
path_to_output = os.path.join(wd, 'output/CcGAN_{}_{}_si{:.3f}_ka{:.3f}_{}_nDs{}_nDa{}_nGa{}_Dbs{}_Gbs{}'.format(
    args.GAN_arch, args.threshold_type, args.kernel_sigma, args.kappa, args.loss_type_gan,
    args.num_D_steps, args.num_grad_acc_d, args.num_grad_acc_g, args.batch_size_disc, args.batch_size_gene))
if args.gan_DiffAugment:
    path_to_output = path_to_output + "_DiAu"
os.makedirs(path_to_output, exist_ok=True)
save_models_folder = os.path.join(path_to_output, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(path_to_output, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)

# 嵌入模型保存文件夹
path_to_embed_models = os.path.join(wd, 'output/embed_models')
os.makedirs(path_to_embed_models, exist_ok=True)

# ================================================================================
'''               Pre-trained CNN and GAN for label embedding                   '''
# ================================================================================
# 构造预训练嵌入模型和映射模型的文件名
net_embed_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_{}_epoch_{}_seed_2020.pth'.format(args.net_embed, args.epoch_cnn_embed))
net_y2h_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_net_y2h_epoch_{}_seed_2020.pth'.format(args.epoch_net_y2h))

print("\n " + net_embed_filename_ckpt)
print("\n " + net_y2h_filename_ckpt)

# 创建用于训练嵌入网络的数据集和数据加载器
trainset = IMGs_dataset(images_train, labels_train, normalize=True)
trainloader_embed_net = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_embed, shuffle=True)

# 根据配置选择不同的嵌入网络结构
if args.net_embed == "ResNet18_embed":
    net_embed = ResNet18_embed(dim_embed=args.dim_embed)
elif args.net_embed == "ResNet34_embed":
    net_embed = ResNet34_embed(dim_embed=args.dim_embed)
elif args.net_embed == "ResNet50_embed":
    net_embed = ResNet50_embed(dim_embed=args.dim_embed)
net_embed = net_embed.cuda()
# 可选：使用nn.DataParallel进行多GPU并行
# net_embed = nn.DataParallel(net_embed)

# 创建标签到嵌入空间的映射网络
net_y2h = model_y2h(dim_embed=args.dim_embed)
net_y2h = net_y2h.cuda()
# net_y2h = nn.DataParallel(net_y2h)

# (1) 训练嵌入网络（net_embed），如果预训练模型不存在则开始训练，否则直接加载
if not os.path.isfile(net_embed_filename_ckpt):
    print("\n Start training CNN for label embedding >>>")
    net_embed = train_net_embed(net=net_embed, net_name=args.net_embed, trainloader=trainloader_embed_net,
                                testloader=None, epochs=args.epoch_cnn_embed, resume_epoch=args.resumeepoch_cnn_embed,
                                lr_base=base_lr_x2y, lr_decay_factor=0.1, lr_decay_epochs=[80, 140],
                                weight_decay=1e-4, path_to_ckpt=path_to_embed_models)
    # 保存模型
    torch.save({
        'net_state_dict': net_embed.state_dict(),
    }, net_embed_filename_ckpt)
else:
    print("\n net_embed ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_embed_filename_ckpt, weights_only=True)
    net_embed.load_state_dict(checkpoint['net_state_dict'])

# (2) 训练将标签映射到嵌入空间的网络net_y2h，如果不存在则训练
if not os.path.isfile(net_y2h_filename_ckpt):
    print("\n Start training net_y2h >>>")
    net_y2h = train_net_y2h(unique_labels_norm, net_y2h, net_embed, epochs=args.epoch_net_y2h,
                             lr_base=base_lr_y2h, lr_decay_factor=0.1, lr_decay_epochs=[150, 250, 350],
                             weight_decay=1e-4, batch_size=128)
    # 保存模型
    torch.save({
        'net_state_dict': net_y2h.state_dict(),
    }, net_y2h_filename_ckpt)
else:
    print("\n net_y2h ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_y2h_filename_ckpt, weights_only=True)
    net_y2h.load_state_dict(checkpoint['net_state_dict'])

# 对训练好的嵌入网络进行“标签往返映射”测试
indx_tmp = np.arange(len(unique_labels_norm))
np.random.shuffle(indx_tmp)
indx_tmp = indx_tmp[:10]
labels_tmp = unique_labels_norm[indx_tmp].reshape(-1, 1)
labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).cuda()
epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1, 1).type(torch.float).cuda()
labels_tmp = torch.clamp(labels_tmp + epsilons_tmp, 0.0, 1.0)
net_embed.eval()
net_h2y = net_embed.h2y
net_y2h.eval()
with torch.no_grad():
    labels_rec_tmp = net_h2y(net_y2h(labels_tmp)).cpu().numpy().reshape(-1, 1)
results = np.concatenate((labels_tmp.cpu().numpy(), labels_rec_tmp), axis=1)
print("\n labels vs reconstructed labels")
print(results)

# ========================================================================
#                            GAN + KD main
# ========================================================================
print("CcGAN: {}, {}, Sigma is {}, Kappa is {}.".format(args.GAN_arch, args.threshold_type, args.kernel_sigma, args.kappa))
save_images_in_train_folder = save_images_folder + '/images_in_train'
os.makedirs(save_images_in_train_folder, exist_ok=True)

start = timeit.default_timer()
print("\n Begin Training:")

# 教师模型保存文件名 (已存在则表示教师已训练过)
Filename_GAN = save_models_folder + '/ckpt_niter_{}.pth'.format(args.teacher_gan)
print('\r', Filename_GAN)


print("Loading pre-trained teacher generator >>>")

# 1) 读取整个检查点
checkpoint = torch.load(Filename_GAN)

# 2) 对 netG_state_dict 进行去除 "module." 前缀处理
old_state_dict_g = checkpoint['netG_state_dict']
new_state_dict_g = {}
for k, v in old_state_dict_g.items():
    if k.startswith("module."):
        new_k = k[len("module."):]  # 去掉 "module."
    else:
        new_k = k
    new_state_dict_g[new_k] = v

checkpoint['netG_state_dict'] = new_state_dict_g

# 3) 对 netD_state_dict 同样进行处理
old_state_dict_d = checkpoint['netD_state_dict']
new_state_dict_d = {}
for k, v in old_state_dict_d.items():
    if k.startswith("module."):
        new_k = k[len("module."):]
    else:
        new_k = k
    new_state_dict_d[new_k] = v
checkpoint['netD_state_dict'] = new_state_dict_d


if args.GAN_arch == "SAGAN":
    netG_teacher = CcGAN_SAGAN_Generator(dim_z=args.dim_gan, dim_embed=args.dim_embed)
    #netD = CcGAN_SAGAN_Discriminator(dim_embed=args.dim_embed)
else:
    raise Exception("Not supported architecture in KD example...")

#netG_teacher = nn.DataParallel(netG_teacher)
#netD_teacher = nn.DataParallel(netD_teacher)

netG_teacher.load_state_dict(checkpoint['netG_state_dict'])
#netD.load_state_dict(checkpoint['netD_state_dict'])
print("Teacher model loaded from:", Filename_GAN)

# ---------------------------
# 3) 定义“学生生成器”和新的判别器，并进行蒸馏训练
# ---------------------------
#   学生仍用 SAGAN 结构，但通道数减半 (gene_ch=32)

netG_student = CcGAN_SAGAN_Generator_Student(
    dim_z=args.dim_gan,
    dim_embed=args.dim_embed,
    gene_ch=32  # 比教师的64小
)
#netG_student = nn.DataParallel(netG_student)

netD = CcGAN_SAGAN_Discriminator(
    dim_embed=args.dim_embed,
    disc_ch=32  # 比教师的64小
)

# “带KD”的训练函数 (注意把 netG_teacher 传入)
netG_student = train_ccgan_with_kd(
    kernel_sigma=args.kernel_sigma,
    kappa=args.kappa,
    train_images=images_train,
    train_labels=labels_train,
    netG=netG_teacher,
    netD=netD,
    netG_student=netG_student,
    KD_rate=1e-4,# KD损失系数
    net_y2h=net_y2h,
    save_images_folder=save_images_in_train_folder,
    save_models_folder=save_models_folder,
    clip_label=False
)

# 将学生生成器保存到一个独立文件 (方便将来单独使用)
Filename_Student = save_models_folder + '/ckpt_student_niter_{}.pth'.format(args.niters_gan)
torch.save({
    'netG_student_state_dict': netG_student.state_dict(),
    'netD_student_state_dict': netD.state_dict(),
}, Filename_Student)
print("Student model training done and saved to:", Filename_Student)

# ---------------------------
# 4) 定义一个简单封装函数，根据给定标签采样生成图像
# ---------------------------

def fn_sampleGAN_given_labels(labels, batch_size, to_numpy=True, denorm=True, verbose=True):
    fake_images, fake_labels = sample_ccgan_given_labels(netG_student, net_y2h, labels, batch_size=batch_size,
                                                          to_numpy=to_numpy, denorm=denorm, verbose=verbose)
    return fake_images, fake_labels

stop = timeit.default_timer()
print("KD training (teacher+student) finished; Time elapses: {}s".format(stop - start))


# ================================================================================
'''                                  Evaluation                                     '''
# ================================================================================
if args.comp_FID:
    print("\n Evaluation in Mode {}...".format(args.eval_mode))

    # -------------------------------
    # 1) 加载FID评估的预训练编码器
    PreNetFID = encoder(dim_bottleneck=512).cuda()
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = args.eval_ckpt_path + '/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs, weights_only=True)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # 2) 加载用于多样性评估的分类网络
    PreNetDiversity = ResNet34_class_eval(num_classes=49, ngpu=torch.cuda.device_count()).cuda()
    Filename_PreCNNForEvalGANs_Diversity = args.eval_ckpt_path + '/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity, weights_only=True)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # 3) 加载用于标签得分（LS）评估的回归网络
    PreNetLS = ResNet34_regre_eval(ngpu=torch.cuda.device_count()).cuda()
    Filename_PreCNNForEvalGANs_LS = args.eval_ckpt_path + '/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS, weights_only=True)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # -------------------------------
    # 以下代码段可用于dump出部分生成图像进行可视化（此处已注释）
    # sel_labels = np.array([0.1, 45, 89])
    # n_per_label = 6
    # for i in range(len(sel_labels)):
    #     curr_label = sel_labels[i]
    #     if i == 0:
    #         fake_labels_assigned = np.ones(n_per_label) * curr_label
    #     else:
    #         fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(n_per_label) * curr_label))
    # images_show, _ = fn_sampleGAN_given_labels(fn_norm_labels(fake_labels_assigned), batch_size=10, to_numpy=False, denorm=False, verbose=True)
    # filename_images_show = save_images_folder + '/visualization_images_grid.png'
    # save_image(images_show.data, filename_images_show, nrow=n_per_label, normalize=True)
    # sys.exit()

    # -------------------------------
    # 构造评估用标签并进行批量采样
    print("\r Start sampling {} fake images per label from Student GAN >>>".format(args.nfake_per_label))
    start_time = timeit.default_timer()

    if args.eval_mode == 1:
        # Mode 1：评估在训练集出现过的唯一标签上
        eval_labels = np.sort(np.array(list(set(labels_train_raw))))
    elif args.eval_mode in [2, 3]:
        # Mode 2和3：评估在数据集所有唯一标签上
        eval_labels = np.sort(np.array(list(set(labels_all))))
    else:
        # Mode 4：在[min_label, max_label]之间均匀采样args.num_eval_labels个标签
        eval_labels = np.linspace(np.min(labels_all), np.max(labels_all), args.num_eval_labels)

    unique_eval_labels = np.sort(np.array(list(set(eval_labels))))
    print("\r There are {} unique eval labels.".format(len(unique_eval_labels)))

    fake_labels_assigned = []
    for i in range(len(eval_labels)):
        curr_label = eval_labels[i]
        if i == 0:
            fake_labels_assigned = np.ones(args.nfake_per_label) * curr_label
        else:
            fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(args.nfake_per_label) * curr_label))

    # 采样学生模型假图
    fake_images, _ = fn_sampleGAN_given_labels(
        fn_norm_labels(fake_labels_assigned),
        args.samp_batch_size
    )
    assert len(fake_images) == args.nfake_per_label * len(eval_labels)
    assert len(fake_labels_assigned) == args.nfake_per_label * len(eval_labels)

    print("\r End sampling! We got {} fake images, which takes {:.3f} sec.".format(
        len(fake_images), timeit.default_timer()-start_time
    ))
    # -------------------------------
    # 如果需要dump生成的假图像以供NIQE计算，则保存图像到指定文件夹
    if args.dump_fake_for_NIQE:
        print("\n Dumping fake images for NIQE...")
        if args.niqe_dump_path == "None":
            dump_fake_images_folder = save_images_folder + '/fake_images'
        else:
            dump_fake_images_folder = args.niqe_dump_path + '/fake_images'
        os.makedirs(dump_fake_images_folder, exist_ok=True)
        for i in tqdm(range(len(fake_images))):
            # 获取每个假图像对应的标签（未归一化）
            label_i = fake_labels_assigned[i]
            filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i].astype(np.uint8)
            # PIL要求图像格式为H x W x C，需转置
            image_i_pil = Image.fromarray(image_i.transpose(1, 2, 0))
            image_i_pil.save(filename_i)
        #sys.exit()

    # -------------------------------
    # 根据评估模式选取真实图像及标签
    if args.eval_mode in [1, 3]:
        real_images = images_train  # 使用训练集图像（未归一化）
        real_labels = labels_train_raw  # 使用训练集标签（未归一化）
    else:
        real_images = images_all  # 使用全部数据
        real_labels = labels_all
    # -------------------------------
    # 为每个真实标签取固定数量的图像
    unique_labels_real = np.sort(np.array(list(set(real_labels))))
    indx_subset = []
    for i in range(len(unique_labels_real)):
        label_i = unique_labels_real[i]
        indx_i = np.where(real_labels == label_i)[0]
        np.random.shuffle(indx_i)
        if args.nreal_per_label > 1:
            indx_i = indx_i[0:args.nreal_per_label]
        indx_subset.append(indx_i)
    indx_subset = np.concatenate(indx_subset)
    real_images = real_images[indx_subset]
    real_labels = real_labels[indx_subset]

    nfake_all = len(fake_images)
    nreal_all = len(real_images)

    if args.comp_IS_and_FID_only:
        # -------------------------------
        # FID: 在所有假图像上计算FID指标
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake],
                      batch_size=500, resize=None, norm_img=True)
        print("\n FID of {} fake images: {}.".format(nfake_all, FID))

        # -------------------------------
        # IS: 在所有假图像上计算Inception Score
        IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=49,
                                     net=PreNetDiversity, cuda=True, batch_size=500, splits=10,
                                     normalize_img=True)
        print("\n IS of {} fake images: {}({}).".format(nfake_all, IS, IS_std))

    else:
        # -------------------------------
        # 对于每个评估中心（滑动窗口中心）计算FID、标签得分和多样性（熵）
        if args.eval_mode == 1:
            center_start = np.min(labels_train_raw) + args.FID_radius
            center_stop = np.max(labels_train_raw) - args.FID_radius
        else:
            center_start = np.min(labels_all) + args.FID_radius
            center_stop = np.max(labels_all) - args.FID_radius

        if args.FID_num_centers <= 0 and args.FID_radius == 0:  # 全部重叠
            centers_loc = eval_labels
        elif args.FID_num_centers > 0:
            centers_loc = np.linspace(center_start, center_stop, args.FID_num_centers)
        else:
            print("\n Error.")

        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc))  # 每个中心的熵
        labelscores_over_centers = np.zeros(len(centers_loc))  # 每个中心的标签得分
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = center - args.FID_radius
            interval_stop  = center + args.FID_radius
            indx_real = np.where((real_labels >= interval_start) * (real_labels <= interval_stop) == True)[0]
            assert len(indx_real) > 0
            np.random.shuffle(indx_real)
            real_images_curr = real_images[indx_real]
            real_images_curr = (real_images_curr / 255.0 - 0.5) / 0.5
            num_realimgs_over_centers[i] = len(real_images_curr)
            indx_fake = np.where((fake_labels_assigned >= interval_start) * (fake_labels_assigned <= interval_stop) == True)[0]
            assert len(indx_fake) > 0
            np.random.shuffle(indx_fake)
            fake_images_curr = fake_images[indx_fake]
            fake_images_curr = (fake_images_curr / 255.0 - 0.5) / 0.5
            fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
            # 计算FID
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size=200, resize=None)
            # 计算预测类别标签的熵（多样性指标）
            predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=200, num_workers=args.num_workers)
            entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            # 计算标签得分（Label Score）
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr,
                                                            fn_norm_labels(fake_labels_assigned_curr),
                                                            min_label_before_shift=0, max_label_after_shift=args.max_label,
                                                            batch_size=200, resize=None, num_workers=args.num_workers)
            # 输出当前中心评估结果
            print("\r [{}/{}] Center:{}; Real:{}; Fake:{}; FID:{:.3f}; LS:{:.3f}; ET:{:.3f}. \n".format(
                i+1, len(centers_loc), center, len(real_images_curr), len(fake_images_curr),
                FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))
        # 输出所有中心的平均、标准差和极值
        print("\n SFID: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(np.mean(FID_over_centers),
              np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\r LS over centers: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(np.mean(labelscores_over_centers),
              np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\r Entropy over centers: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(np.mean(entropies_over_centers),
              np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

        # 保存各中心的评估结果到文件
        dump_fid_ls_entropy_over_centers_filename = os.path.join(path_to_output, 'fid_ls_entropy_over_centers')
        np.savez(dump_fid_ls_entropy_over_centers_filename,
                 fids=FID_over_centers, labelscores=labelscores_over_centers,
                 entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

        # -------------------------------
        # 在所有假图像上计算FID指标
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake],
                      batch_size=200, resize=None, norm_img=True)
        print("\n {}: FID of {} fake images: {:.3f}.".format(args.GAN_arch, nfake_all, FID))

        # -------------------------------
        # 计算整体的标签得分（LS）
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fn_norm_labels(fake_labels_assigned),
                                                         min_label_before_shift=0, max_label_after_shift=args.max_label,
                                                         batch_size=200, resize=None, norm_img=True, num_workers=args.num_workers)
        print("\n {}: overall LS of {} fake images: {:.3f}({:.3f}).".format(args.GAN_arch, nfake_all, ls_mean_overall, ls_std_overall))

        # -------------------------------
        # 将评估结果写入日志文件
        eval_results_logging_fullpath = os.path.join(path_to_output, 'eval_results_{}.txt'.format(args.GAN_arch))
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n Radius: {}; # Centers: {}.  \n".format(args.FID_radius, args.FID_num_centers))
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n SFID: {:.3f}({:.3f}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            eval_results_logging_file.write("\n LS: {:.3f}({:.3f}).".format(ls_mean_overall, ls_std_overall))
            eval_results_logging_file.write("\n Diversity: {:.3f}({:.3f}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
            eval_results_logging_file.write("\n FID: {:.3f}.".format(FID))

print("\n===================================================================================================")
