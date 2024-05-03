## Truba 

### Activate Conda environment 
``` 
eval "$(/truba/home/hbilgi/miniconda3/bin/conda shell.bash hook)"
```
### To work with GPU
- Use 
    ``` 
    ssh palamut-ui
    ```

```
#!/bin/bash
#SBATCH -p palamut-cuda     # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A hbilgi          # Kullanici adi
#SBATCH -J print_gpu        # Gonderilen isin ismi
#SBATCH -o print_gpu.out    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 10  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=12:00:00      # Sure siniri koyun.


eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate dl-env
python print_gpu.py
```
## Interactif 

```
srun -p palamut-cuda -N 1 -n 1 -c 16 --gres=gpu:1 --time 1:00:00 --pty /bin/bash
```

```
RUN=mot17_public_train
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
python scripts/main.py --experiment_mode train --cuda --train_splits mot17-train-all --val_splits mot17-train-all --run_id ${RUN} --interpolate_motion --linear_center_only --det_file aplift --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp

```