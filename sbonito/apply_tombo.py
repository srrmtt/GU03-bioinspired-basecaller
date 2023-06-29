from pathlib import Path
import os
import argparse
from subprocess import Popen,run
import time



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, help='dataset, structured as wick')
    args = parser.parse_args()
    """
    dataset_dir
        specie1
            fast5
            reference.fasta
        specie2 
        ...
    """
    process_handles=list()
    for i,specie_dir in enumerate(os.listdir(args.dataset_dir)):
        if specie_dir not in ["tmp", "genomes"]:

            specie_dir=os.path.join(args.dataset_dir,specie_dir)

            cmd_str="tombo resquiggle "+os.path.join(specie_dir,"fast5")+" "\
            +os.path.join(specie_dir,"read_references.fasta")\
            +" --processes 2 --dna --num-most-common-errors 5 --ignore-read-locks --overwrite"

            print("\n\ncurrent command:\n\n",i," ",cmd_str)
            
            """
            subprocess.run(["tombo", "resquiggle", os.path.join(specie_dir,"fast5"), 
                            os.path.join(specie_dir,"read_references.fasta"), 
                            "--processes", "2", "--dna",
                            "--num-most-common-errors","5", 
                            "--ignore-read-locks",
                            "--overwrite" ])
            """
            #run([cmd_str])
            process_handles.append(Popen([cmd_str], shell=True,stdin=None, stdout=None, stderr=None, close_fds=True))
            
    for handle in process_handles:
        (stdout_data, stderr_data)=handle.communicate()
        print(stdout_data)
         