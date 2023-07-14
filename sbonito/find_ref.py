import os
import argparse


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--fast5-list", type=str, required = False)
    #args = parser.parse_args()

    fast5_list="./sbonito/inter_task_test_reads.txt"

    specie_list=set()
    with open(fast5_list, 'r') as f:
            for line in f:
                specie_list.add(line.rsplit('/',2)[0])
    
    cnt=0
    no_fasta=0
    for specie_dir in specie_list:
        cnt+=1
        if not os.path.exists(os.path.join(specie_dir,"read_references.fasta")):
            no_fasta+=1
            print(cnt," no fasta for: ", specie_dir.split("/"[-1]))
    
    print(no_fasta)
    print(cnt)