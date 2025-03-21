#!/bin/bash

ROOT_PATH="D:/local_wd/tmp/CcGAN_tutorial/RC49/CcGAN"
DATA_PATH="C:/Users/DX/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/RC-49"
EVAL_PATH="D:/local_wd/tmp/CcGAN_tutorial/RC49/evaluation"
niqe_dump_path="D:/local_wd/tmp/CcGAN_tutorial/CcGAN_TPAMI_NIQE/RC-49/NIQE_64x64/fake_data"

SEED=2024
NUM_WORKERS=0
MIN_LABEL=0
MAX_LABEL=90.0
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=25
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

BATCH_SIZE_G=256
BATCH_SIZE_D=256
NUM_D_STEPS=2
SIGMA=-1.0
KAPPA=-2.0
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=1
NUM_ACC_G=1

GAN_ARCH="SAGAN"
LOSS_TYPE="hinge"

DIM_GAN=256
DIM_EMBED=128

NITERS=30000
Setting="niters${NITERS}"

python main.py \
    --setting_name $Setting \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan 30000 --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 5000 --visualize_freq 2000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --num_grad_acc_d $NUM_ACC_D --num_grad_acc_g $NUM_ACC_G \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --comp_FID --samp_batch_size 500 \
    2>&1 | tee output_${GAN_ARCH}_${SETTING}.txt

    # --dump_fake_for_NIQE --niqe_dump_path $niqe_dump_path