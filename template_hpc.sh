#!/bin/sh
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

#!/bin/sh