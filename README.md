# GU03-bioinspired-basecaller
This repository contain the code and scripts to train and evaluate nanopore basecalling neural networks Bonito based on the Based on the code base: https://github.com/marcpaga/nanopore_benchmark used for the paper Comprehensive benchmark and architectural analysis of deep learning models for nanopore sequencing basecalling available at: https://www.biorxiv.org/content/10.1101/2022.05.17.492272v2. Our repository give the possibility to replace N levels SLSTM with N levels SNN in the encoder of the architecture bonito (with N ranging from 0 to 5). We also implemented other models like bonitospikeconv and bonitospikelin that attempt to build a fully spiking architecture.


## Installation
This code has been tested on python 3.9.16.
```

git clone https://github.com/srrmtt/GU03-bioinspired-basecaller.git
python3 -m venv gu3
source gu3/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```
## Getting started
### Data Download
To download the data check the bash scripts in ./download.
To download use this three scripts to download specific datasets: download_wick_train.sh and download_wick_test.sh

WARNING: This is a large data download, about 222GB in disk space.
### Data processing
There is two main step for processing the data
#### Annotate the raw data with the reference/true sequence
For this step we used Tombo, which models the expected average raw signal level and aligns the expected signal with the measured signal. See their [documentation ](https://nanoporetech.github.io/tombo/resquiggle.html) for detailed info on how to use it.
After installation of tombo (you have to install it in a different environment, as it is not compatible with the training environment) you should be able to run the following.
```
source .bashrc
conda activate gu3.7
cd GU03-bioinspired-basecaller/sbonito
python ./apply_tombo.py --dataset-dir ./wick_to_be_resquiggled --processes 1

```
#### Chunk the raw signal and save it numpy arrays
In this step, we take the raw signal and splice it into segments of a fixed length so that they can be fed into the neural network.

This can be done by running the following script:
```
source .bashrc
conda activate gu3
python GU03-bioinspired-basecaller/sbonito/scripts/data_prepare_numpy.py \
--fast5-dir  GU03-bioinspired-basecaller/sbonito/wick_to_be_resquiggled \
--output-dir  GU03-bioinspired-basecaller/sbonito/new_train_numpy_after_resquiggle \
--total-files  4 \
--window-size 2000 \
--window-slide 0 \
--n-cores 4 \
--verbose
```
## Model Training
In this step we fed all the data we prepared (in numpy arrays), and train the model.

We can train three different types of model: bonito(classic architecture of Bonito), bonitosnn(with nlstm that indicate how many layer lstm you want in decoder) or bonitospikeconv (Bonito with layer SNN in the feature extracture and in encoder, nlstm indicates how many LSTM levels to insert into the encoder) :
```
source .bashrc
conda activate gu3
python GU03-bioinspired-basecaller/sbonito/scripts/train_original.py \
--data-dir GU03-bioinspired-basecaller/sbonito/new_train_numpy_after_resquiggle \
--output-dir GU03-bioinspired-basecaller/sbonito/trained_bonito \
--model bonito \
--window-size 2000 \
--batch-size 64 \
--starting-lr 0.001 \
--nlstm 1

```
## Model Training with NNI
If you want to train the net using nni you must use the following script, model parameter can be bonitosnn to insert snn layer in the decoder, or bonitospikeconv  (Bonito with layer SNN in the feature extracture and in encoder) :
```
source .bashrc
conda activate gu3
python GU03-bioinspired-basecaller/sbonito/experimentnni.py \
--data-dir ./new_wick2_train_numpy \
--output-dir ./test_nni_2 \
--model bonitosnn \
--nlstm 0 \
--train-file /home/mla_group_11/GU03-bioinspired-basecaller/sbonito/scripts/train_originalnni.py \
--code-dir GU03-bioinspired-basecaller/sbonito \
--nni-dir GU03-bioinspired-basecaller/sbonito/nni-experiments \
--num-epochs 5
```
The tuning in bonitosnn works with the following hyperparameters:
```
 search_space = {
        'batch-size': {'_type': 'randint', '_value': [16, 128]},
        'starting-lr': {'_type': 'loguniform', '_value': [0.0001, 0.01]},
        'slstm_threshold':{'_type': 'uniform', '_value': [0.01, 0.2]},
        }
```
The tuning in bonitospikeconv works with the following hyperparameters:
```
 search_space = {
        'batch-size': {'_type': 'randint', '_value': [16, 128]},
        'starting-lr': {'_type': 'loguniform', '_value': [0.0001, 0.01]},
        'slstm_threshold':{'_type': 'uniform', '_value': [0.01, 0.2]},
        'conv_th':{'_type': 'uniform', '_value': [0.01, 0.2]},
        }
```
Bonitspikeconv tuning also performs tuning in leaky neurons present in feature extracture

## Basecalling
Once a model has been trained, it can be used for basecalling. Here's an example command with the demo data:
```
source .bashrc
conda activate gu3
cd GU03-bioinspired-basecaller/sbonito/
python ./scripts/basecall_original.py \
--model bonitosnn \
--fast5-list ./inter_task_test_reads.txt \
--checkpoint ./trained/papermodels/inter_2000/checkpoint.pt \
--output-file ./trained/papermodels/inter_basecall_snn.fastq
```

## Evaluation
For the evaluation of the various model lunch this script:
```
source .bashrc
conda activate gu3
cd GU03-bioinspired-basecaller/sbonito
python3 evaluate.py --basecalls-path trained_bonito/inter_basecall_snn.fastq \
--references-path wick_to_be_resquiggled/all_references.fasta \
--output-file evaluations/bonito/fasta_bonito.csv \
--model-name bonitosnn \
--nlstm 2
```
WARNING: Not use underscore in model name

## Report
To create a report of the evaluation of your model based on the reference and basecalls do the following:
```
source .bashrc
conda activate gu3
cd GU03-bioinspired-basecaller/sbonito
python3 report.py \
--evaluation-file ./evaluations/bonito/fasta_bonito.csv \
--output-dir ./evaluations/bonito/reports \
--model-name bonitosnn
```
This will generate a bunch of report csv files in the output dir:

* absoultecounts: this contains the counts across all reads for different metrics (matched bases, mismatches bases, homopolymer correct bases, etc.)
* auc: this contains the values necessary to plot the AUC for the model.
* fraction: top fraction of best reads according to phredq.
* match_rate: match rate of reads in that fraction.
* phredq_mean: average PhredQ score of reads in that fraction.
* event rates: this contains the boxplot statistics for the main alignment events: match, mismatch, insertion and deletion.
* homopolymerrates: this containts the boxplot statistics for the homopolymer error rates per base or all together.
* phredq: this contains the boxplot statistics for the PhredQ scores of correctly and incorrectly basecalled bases.
* readoutcomes: this contains the number of reads that are successfully evaluated or that had some sort of error.
* signatures: this contains the rates and counts of different types of errors for each base in a 3-mer context. The 3-mer contexts are based on the basecalls, not the reference.
* singlevalues: this contains single summary values across all metrics based on the absolute counts, the read outcomes and the PhredQ scores distributions.

## Plots
For the plot of the various models lunch the following script(depth is how deep it will search between the folders):
```
conda activate gu3
python3 plot.py \
--reports evaluations \
--output-dir evaluations/plots \
--depth 5
```
