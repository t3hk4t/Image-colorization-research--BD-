#!/bin/sh -v
#PBS -e /mnt/beegfs2/home/leo01/Bachelor_thesis/logs/
#PBS -o /mnt/beegfs2/home/leo01/Bachelor_thesis/logs/
#PBS -q long
#PBS -N preprocessor
#PBS -p 1000
#PBS -l nodes=1:ppn=36:feature=vasara
#PBS -l mem=40gb
#PBS -l walltime=48:00:00

module load conda
eval "$(conda shell.bash hook)"
source activate leonards_elksnis
export LD_LIBRARY_PATH=~/.conda/envs/leonards_elksnis/lib:$LD_LIBRARY_PATH
export SDL_AUDIODRIVER=waveout
export SDL_VIDEODRIVER=x11

#ulimit -n 500000

cd /mnt/beegfs2/home/leo01/Bachelor_thesis/


python datasets/scripts/Frames2MemmapsMultithreaded.py
#!/bin/sh