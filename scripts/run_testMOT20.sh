#!/bin/bash
#SBATCH -p palamut-cuda     # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ogam3            # Kullanici adi
#SBATCH -J inference          # Gonderilen isin ismi
#SBATCH -o inference20.out    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 16  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=24:00:00     # Sure siniri koyun.

eval "$(/truba/home/hbilgi/miniconda3/bin/conda shell.bash hook)"
conda activate py_cuda

RUN=mot20_private_test
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH='/truba/home/hbilgi/dev/thesis/mot_data'
OUT_PATH='/truba/home/hbilgi/dev/simple-model/outputs'

PRETRAINED_MODEL_PATH=outputs/experiments/mot20_private_train_02-29_11:56:27.332701/models/hiclnet_epoch_7_iteration749.pth
python main.py --hicl_model_path ${PRETRAINED_MODEL_PATH} --experiment_mode test --cuda --test_splits mot20-test-all \
    --run_id ${RUN} --rounding_method hungarian_full --interpolate_motion --linear_center_only  \
    --det_file byte065 --data_path ${DATA_PATH} --output_path ${OUT_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} \
    --reid_arch ${REID_ARCH} --save_cp --node_level_embed --load_transformers_only

