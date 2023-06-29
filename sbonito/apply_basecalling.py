from pathlib import Path
import os
import argparse
from subprocess import Popen,PIPE



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, help='wick structured dataset')
    parser.add_argument("--checkpoint", type=str, help='path of checkpoint file')
    parser.add_argument("--output-dir", type=str, help='directory of the resulting basecalls files')
    args = parser.parse_args()

    print("args: ",args)
    """
    dataset_dir
        specie1
            fast5
            reference.fasta
        specie2 
        ...
    """
    """
    file tree risultante:
    output_dir
        specie1.fastq
        specie2.fastq
        ...
    """
    process_handles=list()
    for specie_dir in os.listdir(args.dataset_dir)[0:5]:
        if specie_dir not in ["tmp", "genomes"]:
            #specie_dir=os.path.join(args.dataset_dir,specie_dir)

            """
            python ./scripts/basecall_original.py \
            --model bonito \
            --fast5-dir ./specie_dir/fast5 \
            --checkpoint checkpoint_path \
            --output-file ./output_dir/specie_dir.fastq
            """

            cmd_str="python ./scripts/basecall_original.py --model bonito --fast5-dir " \
            +os.path.join(os.path.join(args.dataset_dir,specie_dir),"fast5") \
            +" --checkpoint "+args.checkpoint \
            +" --output-file "+os.path.join(args.output_dir,specie_dir+".fastq")+" --batch-size 8"  #per farlo entrare nella memoria locale

            print("current command: ", cmd_str)
            
            process_handles.append(Popen([cmd_str], shell=True,
                stdin=None, stdout=PIPE, stderr=PIPE))
            
    for handle in process_handles:
        (stdout_data, stderr_data)=handle.communicate()
        print(stdout_data)
        