import os
import argparse


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--fast5-list", type=str, required = False)
    #args = parser.parse_args()

    fast5_list="./sbonito/inter_task_test_reads.txt"

    specie_list={}
    with open(fast5_list, 'r') as f:
            for line in f:
                specie=line.rsplit('/',3)[0]
                if specie in specie_list:
                    specie_list[specie].append(line)
                else:
                    specie_list[specie]=[line]

    with open("inter_test_cleaned.txt","w") as fout:
    
        cnt=0
        no_fasta=0
        for specie_dir in specie_list.keys():
            cnt+=1
            if not os.path.exists(os.path.join(specie_dir,"read_references.fasta")):
                no_fasta+=1
                print(cnt," no fasta for: ", specie_dir.split("/"[-1]))
            #else:
            #    for read in specie_list[specie_dir]:
            #        fout.write(read)
            if os.path.exists(specie_dir):
                for read in specie_list[specie_dir]:
                    fout.write(read)
             
    
    print(no_fasta)
    print(cnt)