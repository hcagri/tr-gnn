#!/bin/bash
#SBATCH -p palamut-cuda     # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ogam3            # Kullanici adi
#SBATCH -J mot17-512-26martP        # Gonderilen isin ismi
#SBATCH -o mot17-512-26martP.out       # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 16  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=72:00:00     # Sure siniri koyun.

eval "$(/truba/home/hbilgi/miniconda3/bin/conda shell.bash hook)"
conda activate py_cuda

RUN=mot17_public_train
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH='/truba/home/hbilgi/dev/thesis/mot_data'
OUT_PATH='/truba/home/hbilgi/dev/simple-model/outputs'
SEQ_MAP1='mot17-train-all'
SEQ_MAP2='mot17-val-split2'
# MODEL_PATH='outputs/experiments/mot17_public_train_10-03_17:51:56.652446/models/hiclnet_epoch_25_iteration4350.pth'

python main.py --experiment_mode train --cuda --train_splits ${SEQ_MAP1} --val_splits ${SEQ_MAP2} --run_id mot17_public_train \
    --save_cp --lr 0.0003 --weight_decay 0.0001 --interpolate_motion --linear_center_only --det_file det \
    --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} \
    --reid_arch $REID_ARCH --output_path ${OUT_PATH} --rounding_method greedy --node_level_embed \
    --depth_pretrain_iteration 500 --num_epoch 300 --num_batch 4 --start_eval 70 --num_workers 4