#!/bin/bash
#SBATCH -p palamut-cuda     # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ogam3            # Kullanici adi
#SBATCH -J mot20_1          # Gonderilen isin ismi
#SBATCH -o mot20_1-train.out       # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 16  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=72:00:00     # Sure siniri koyun.

eval "$(/truba/home/hbilgi/miniconda3/bin/conda shell.bash hook)"
conda activate py_cuda

RUN=mot20_private_train
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH='/truba/home/hbilgi/dev/thesis/mot_data'
OUT_PATH='/truba/home/hbilgi/dev/simple-model/outputs'
SEQ_MAP1='mot20-train-all'
SEQ_MAP2='mot20-train-all'
MODEL_PATH='/truba/home/hbilgi/dev/simple-model/outputs/experiments/mot17_private_train_02-28_11:04:32.330382/models/hiclnet_epoch_138_iteration12006.pth'

python main.py --experiment_mode train --cuda --train_splits ${SEQ_MAP1} --val_splits ${SEQ_MAP2} --run_id mot20_private_train \
    --save_cp --lr 0.0003 --weight_decay 0.0001 --interpolate_motion --linear_center_only --det_file byte065 \
    --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} \
    --reid_arch $REID_ARCH --output_path ${OUT_PATH} --rounding_method greedy --node_level_embed \
    --depth_pretrain_iteration 500 --num_epoch 100 --num_batch 4 --start_eval 40 --num_workers 4 \
    --load_train_ckpt --hicl_model_path ${MODEL_PATH} --load_transformers_only