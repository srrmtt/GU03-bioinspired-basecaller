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
    #plot multiple learning curves
    log_list=["../trainlogs/ourbonito.log",
              "../trainlogs/bonitosnn.log",
              "../trainlogs/bonito_1_snn.log",
              "../trainlogs/bonito_2_snn.log",
              "../trainlogs/bonito_3_snn.log",
              "../trainlogs/bonito_4_snn.log"]

    label_list=["ourbonito","bonitosnn","1 snn","2 snn","3 snn","4 snn"]

    fig, ax = plt.subplots(2, 1)

    for i,filepath in enumerate(log_list):
        with open(filepath) as logfile:
            logfile=csv.reader(logfile,delimiter=',')
            #logfile2=csv.reader(logfile2,delimiter=',')

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
            
            linecnt=0
            

            ax[0].plot(steps,train_loss,label=label_list[i]) #"training loss "+
            #ax[0].plot(steps,val_loss,label="validation loss "+label_list[i])

            #ax[0].plot(steps,train_loss,label="training loss lr:1e-4")
            #ax[0].plot(steps,val_loss,label="validation loss lr:1e-4")

            ax[0].set_title('model loss')
            ax[0].set_ylabel('loss')
            ax[0].set_xlabel('timestep')
            ax[0].legend()

            ax[1].plot(steps,train_accs,label=label_list[i]) #"training accuracy "+
            #ax[1].plot(steps,val_accs,label="validation accuracy "+label_list[i])

            #ax[1].plot(steps2,train_accs2,label="training accuracy lr:1e-4")
            #ax[1].plot(steps2,val_accs2,label="validation accuracy lr:1e-4")

            ax[1].set_title('model accuracy')
            ax[1].set_ylabel('accuracy')
            ax[1].set_xlabel('timestep')
            ax[1].legend()
            
    plt.show()
            #plt.tight_layout()
            #plt.savefig(args.output_img,format='svg')
    
    main()