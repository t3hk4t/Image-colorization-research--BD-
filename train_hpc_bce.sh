#!/bin/sh -v
#PBS -e /mnt/beegfs2/home/leo01/Bachelor_thesis/logs/
#PBS -o /mnt/beegfs2/home/leo01/Bachelor_thesis/logs/
#PBS -q batch
#PBS -N first_hpc_run
#PBS -p 1000
#PBS -l nodes=1:ppn=8:gpus=1:shared,feature=v100
#PBS -l mem=40gb
#PBS -l walltime=96:00:00

module load conda
eval "$(conda shell.bash hook)"
source activate leonards_elksnis
export LD_LIBRARY_PATH=~/.conda/envs/leonards_elksnis/lib:$LD_LIBRARY_PATH
#mkdir /scratch/evalds
#mkdir /scratch/evalds/tmp
#mkdir /scratch/evalds/data
#export TMPDIR=/scratch/evalds/tmp
#export TEMP=/scratch/evalds/tmp
export SDL_AUDIODRIVER=waveout
export SDL_VIDEODRIVER=x11

#ulimit -n 500000

cd /mnt/beegfs2/home/leo01/Bachelor_thesis/


python taskgen.py \
-sequence_name conv2d_realu_radam \
-template template_hpc.sh \
-is_force_start True \
-num_repeat 1 \
-num_cuda_devices_per_task 1 \
-num_tasks_in_parallel 1 \
-learning_rate 3e-4 \
-batch_size 10 \
-path_train /mnt/beegfs2/home/leo01/image_data/video_framed_memmap_dataset/train \
-path_test /mnt/beegfs2/home/leo01/image_data/video_framed_memmap_dataset/test \
-epochs 50 \
-is_debug False \
-conv3d_depth 1 \
-expansion_rate 3 \


#python taskgen.py \
#-sequence_name unet_3plus_data_256_median_background_loss_compare \
#-template template_hpc.sh \
#-is_force_start True \
#-num_repeat 1 \
#-num_cuda_devices_per_task 1 \
#-num_tasks_in_parallel 1 \
#-is_cuda true \
#-optimizer radam \
#-loss_function ce \
#-combined_loss_weights 1. \
#-is_dataset_weighted_loss False \
#-is_per_sample_weighted_loss False \
#-batch_size 32 \
#-learning_rate 3e-4 3e-3 \
#-epochs 200 \
#-model model_4_unet3plus_multichannel \
#-datasource datasource_4_train_balance \
#-data_features rgb \
#-model_in_channels 3 \
#-is_log_first_layer_grad False \
#-is_filter_damage False \
#-filter_damage_min 0.05 \
#-filter_damage_max 1.0 \
#-is_filter_panel_percent False \
#-panel_part_min 0.2 \
#-panel_part_max 1.0 \
#-is_windows_test False \
#-is_cross_entropy True \
#-data_workers 1 \
#-path_train \
#/mnt/home/valtersve/scopetech_data/data_ready_256_v4/dents_Infolks_developer_fixed_dent \
#/mnt/home/valtersve/scopetech_data/data_ready_256_v4/dents_Infolks_developer_fixed_undamaged \
#-path_test \
#/mnt/home/valtersve/scopetech_data/data_ready_256_v4/val_PSA_Undamaged_undamaged \
#/mnt/home/valtersve/scopetech_data/data_ready_256_v4/val_Infolks_Undamaged_undamaged \
#/mnt/home/valtersve/scopetech_data/data_ready_256_v4/val_Infolks_Real_undamaged \
#/mnt/home/valtersve/scopetech_data/data_ready_256_v4/val_Infolks_Real_dent \
#/mnt/home/valtersve/scopetech_data/data_ready_256_v4/val_Audi_Undamaged_undamaged \
#-path_data_args /media/hdd_data2/datasets/data_args.json \
#-early_stopping_patience 200 \
#-is_deep_supervision True \
#-is_cgm False \
#-unet_depth 4 \
#-expansion_rate 2 \
#-first_conv_channel_count 8 \
#-channels_for_concat 8