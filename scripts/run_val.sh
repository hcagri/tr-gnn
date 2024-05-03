#!/bin/bash
#SBATCH -p palamut-cuda      # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ogam3            # Kullanici adi
#SBATCH -J without_vis          # Gonderilen isin ismi
#SBATCH -o without_vis.out    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 16  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=1:00:00     # Sure siniri koyun.

eval "$(/truba/home/hbilgi/miniconda3/bin/conda shell.bash hook)"
conda activate py_cuda

REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH='/truba/home/hbilgi/dev/thesis/mot_data'
OUT_PATH='/truba/home/hbilgi/dev/simple-model/outputs'
SEQ_MAP1='mot17-train-split2'
SEQ_MAP2='mot17-val-split2'

PRETRAINED_MODEL_PATH=outputs/experiments/mot17_private_train_03-26_09:30:02.056639/models/hiclnet_epoch_163_iteration28362.pth

python main.py --hicl_model_path ${PRETRAINED_MODEL_PATH} --experiment_mode val --cuda --train_splits  ${SEQ_MAP1}  --val_splits  ${SEQ_MAP2}  \
    --run_id mot17_public_train --rounding_method hungarian_full --interpolate_motion --linear_center_only  \
    --det_file byte065 --data_path /truba/home/hbilgi/dev/thesis/mot_data --output_path /truba/home/hbilgi/dev/simple-model/outputs --reid_embeddings_dir reid_fastreid_msmt_BOT_R50_ibn --node_embeddings_dir node_fastreid_msmt_BOT_R50_ibn \
    --reid_arch ${REID_ARCH} --save_cp --node_level_embed 
