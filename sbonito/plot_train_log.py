import argparse
import csv
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, help='Path of the log file')
    parser.add_argument("--output-img", type=str, help='Path of the output svg image')
    args = parser.parse_args()

    with open(args.log_file) as logfile:
        logfile=csv.reader(logfile,delimiter=',')
        linecnt=0
        steps=[]
        train_accs=[]
        val_accs=[]
        train_loss=[]
        val_loss=[]
        lr_schedule=[]
        for line in logfile:
            if linecnt==0:
                pass
            else:
                steps.append(float(line[1]))
                train_loss.append(float(line[3]))
                val_loss.append(float(line[4]))
                train_accs.append(float(line[7]))
                val_accs.append(float(line[8]))
                lr_schedule.append(float(line[9]))
            linecnt+=1

        fig, ax = plt.subplots(2, 1)

        ax[0].plot(steps,train_loss,label="training loss")
        ax[0].plot(steps,val_loss,label="validation loss")
        ax[0].set_title('model loss')
        ax[0].set_ylabel('loss')
        ax[0].set_xlabel('timestep')
        ax[0].legend()

        ax[1].plot(steps,train_accs,label="training accuracy")
        ax[1].plot(steps,val_accs,label="validation accuracy")
        ax[1].set_title('model accuracy')
        ax[1].set_ylabel('accuracy')
        ax[1].set_xlabel('timestep')
        ax[1].legend()
        
        #plt.show()
        plt.tight_layout()
        plt.savefig(args.output_img,format='svg')



if __name__ == '__main__':
    main()