import nni
from nni.experiment import Experiment
import os
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #args di train_original_nni
    parser.add_argument("--data-dir", type=str, help='Path where the data for the dataloaders is stored', 
    default="/root/Acinetobacter_baumannii_AYP-A2/train_numpy_resquiggled/nni")
    parser.add_argument("--output-dir", type=str, help='Path where the model is saved',default="./test_nni")
    parser.add_argument("--model", type=str, choices=[
        'bonito',
        'catcaller',
        'causalcall',
        'mincall',
        'sacall',
        'urnano',
        'halcyon',
        'bonitosnn',
        'bonitospikeconv',
        'bonitospikelin'
    ], help='Model',default="bonitosnn")
    parser.add_argument("--num-epochs", type=int, default = 1)
    parser.add_argument("--nlstm",type=int,choices=[0,1,2,3,4],help='number of lstm blocks must be between 0 and 4',default=0)
    #args dell'esperimento
    parser.add_argument("--train-file", type=str, help='Path of the train_nni file', 
        default="/mnt/c/Users/utente/Desktop/MLA/progetto/GU03-bioinspired-basecaller/sbonito/scripts/train_originalnni.py")
    parser.add_argument("--code-dir", type=str, help='Path of the code dir', default='./sbonito')
    parser.add_argument("--nni-dir", type=str, help='Path of the nni-experiments dir', default='./sbonito/nni-experiments')
    args = parser.parse_args()


    search_space = {
    'batch-size': {'_type': 'randint', '_value': [16, 128]},
    'starting-lr': {'_type': 'loguniform', '_value': [0.0001, 0.01]},
    #'warmup-steps': 5000,
    'slstm_threshold':{'_type': 'uniform', '_value': [0.01, 0.1]},
    }

    print("working dir: ",os.getcwd())

    experiment = Experiment('local')

    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'Anneal'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

    experiment.config.trial_command = 'python '+args.train_file+" --data-dir "+args.data_dir+" --output-dir "+args.output_dir+\
        " --model "+args.model
    experiment.config.trial_code_directory = args.code_dir #'./sbonito'
    #experiment.config.trial_gpu_number=1
    #experiment.config.use_active_gpu=True

    experiment.config.experiment_working_directory=args.nni_dir #'./sbonito/nni-experiments'

    experiment.config.max_trial_number = 10
    experiment.config.trial_concurrency = 1

    #experiment.config.max_experiment_duration = '5m'

    experiment.run(8080)