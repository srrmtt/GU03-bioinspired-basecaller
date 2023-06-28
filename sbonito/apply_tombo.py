from pathlib import Path
import os
import argparse
from subprocess import Popen



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
    for specie_dir in os.listdir(args.dataset_dir):
        if specie_dir not in ["tmp", "genomes"]:

            specie_dir=os.path.join(args.dataset_dir,specie_dir)

            cmd_str="tombo resquiggle "+os.path.join(specie_dir,"fast5")+" "\
            +os.path.join(specie_dir,"read_references.fasta")\
            +" --processes 2 --dna --num-most-common-errors 5 --ignore-read-locks --overwrite"
            
            """
            subprocess.run(["tombo", "resquiggle", os.path.join(specie_dir,"fast5"), 
                            os.path.join(specie_dir,"read_references.fasta"), 
                            "--processes", "2", "--dna",
                            "--num-most-common-errors","5", 
                            "--ignore-read-locks",
                            "--overwrite" ])
            """
            Popen([cmd_str], shell=True,
                stdin=None, stdout=None, stderr=None, close_fds=True)
         