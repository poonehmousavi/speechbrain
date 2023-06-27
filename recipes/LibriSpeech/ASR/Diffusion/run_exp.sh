# !/bin/bash
module load python/3.10.2
source $HOME/diff_env/bin/activate

scp $HOME/projects/def-ravanelm/datasets/librispeech/train-clean-360.tar.gz $SLURM_TMPDIR
scp $HOME/projects/def-ravanelm/datasets/librispeech/train-clean-100.tar.gz $SLURM_TMPDIR
scp $HOME/projects/def-ravanelm/datasets/librispeech/train-other-500.tar.gz $SLURM_TMPDIR
scp $HOME/projects/def-ravanelm/datasets/librispeech/test-clean.tar.gz $SLURM_TMPDIR
scp $HOME/projects/def-ravanelm/datasets/librispeech/test-other.tar.gz $SLURM_TMPDIR
scp $HOME/projects/def-ravanelm/datasets/librispeech/dev-clean.tar.gz $SLURM_TMPDIR


cd $SLURM_TMPDIR
tar -zxf train-clean-360.tar.gz
tar -zxf train-clean-100.tar.gz
tar -zxf train-other-500.tar.gz
tar -zxf test-clean.tar.gz
tar -zxf test-other.tar.gz
tar -zxf dev-clean.tar.gz



cd $HOME/diffusion_asr_sb/recipes/LibriSpeech/ASR/Diffusion/
python train_diffusion_bert.py   hparams/train_with_diffusion_bert.yaml  --lm_folder=$HOME/scratch/hf/   --data_folder=$SLURM_TMPDIR/LibriSpeech/ "$@"


