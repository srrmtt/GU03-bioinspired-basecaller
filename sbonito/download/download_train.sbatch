#!/bin/bash
#SBATCH --job-name=download_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name.surname@polito.it
#SBATCH --partition=global
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=download_train_%j.log
#SBATCH --mem-per-cpu=1024M

/home/mla_group_11/GU03-bioinspired-basecaller/sbonito/download/download_wick_train.sh