import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np

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


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

if __name__ == '__main__':
    lrs=[6.66,64.3,16,9.01,1.6,13.5,44.8]
    slstm_ths=[2.98,9.36,9.87,8.67,5.54,2.51,2.62]
    batch_sizes=np.array([80,79,68,60,24,116,103])
    accuracies=[0.6567,0.8636,0.8613,0.8696,0.8532,0.8583,0.8148]
    
    #Bonitospikeconv
    #lrs=[25.2,10.5,9.56,2.68,17.5]
    #slstm_ths=[8.93,8.37,10,16.3,14.1]
    #leaky_ths=[18.8,11.8,17.6,7.13,16.9]
    #batch_sizes=np.array([114,95,49,118,110])
    #accuracies=[0.5085,0.5863,0.4492,0.5385,0.5355]

    #plt.style.use("seaborn")

    plt.scatter(lrs,slstm_ths,s=batch_sizes/116*200,
                c=accuracies,cmap="jet", label="Batch size")
    cbar=plt.colorbar()
    cbar.set_label("Accuracy")
    plt.title("BonitoSnn hyperparameter optimization trials")
    plt.xlabel("learning rate (10$^{-4}$)")
    plt.ylabel("slstm threshold (10$^{-2}$)")
    plt.legend()

    plt.show()
    """
    Script to quickly plot learning curves from train log files
    """
    #plot multiple learning curves 80 79 68 60 24
    """
              "../trainlogs/bonitosnn.log",
              "../trainlogs/bonito_1_snn.log",
              "../trainlogs/bonito_2_snn.log",
              "../trainlogs/bonito_3_snn.log",
              "../trainlogs/bonito_4_snn.log",
    """
    log_list=["../trainlogs/nni_spikeconv/trial_0/train.log",
              "../trainlogs/nni_spikeconv/trial_1/train.log",
              "../trainlogs/nni_spikeconv/trial_2/train.log",
              "../trainlogs/nni_spikeconv/trial_3/train.log",
              #"../trainlogs/nni/trial_4/train.log",
              #"../trainlogs/nni/trial_5/trial_6.log",
              "../trainlogs/nni/trial_6/trial_7.log"

              #"../trainlogs/bonito_1_snn.log",
              #"../trainlogs/bonito_2_snn.log",
              #"../trainlogs/bonito_3_snn.log",
              #"../trainlogs/bonito_4_snn.log"
            ]

    label_list=["trial 1","trial 2","trial 3","trial 4","trial 5","trial 6","trial 7"]
    #window_list=[1,80//79,80//68,80//60,80//24]
    npts=236
    #"bonitosnn","1 snn","2 snn","3 snn","4 snn"]

    fig, ax = plt.subplots(1, 1)

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

            steps=steps[:npts] #np.array(steps)[np.linspace(0, len(steps)-1, npts,dtype=np.int32)]
            #train_loss=np.array(train_loss)[np.linspace(0, len(train_loss)-1, npts,dtype=np.int32)]
            lr_schedule=np.array(lr_schedule)[np.linspace(0, len(lr_schedule)-1, npts,dtype=np.int32)]
            #train_accs=np.array(train_accs)[np.linspace(0, len(train_accs)-1, npts,dtype=np.int32)]
            #steps=moving_average(steps,window_list[i])
            #train_loss=moving_average(train_loss,window_list[i])
            #train_accs=moving_average(train_accs,window_list[i])
            

            #ax[0].plot(steps,train_loss,label="training loss "+label_list[i]) #"training loss "+
            #ax[0].plot(steps,val_loss,'--',label="validation loss "+label_list[i])
            
            ax.plot(steps,lr_schedule,label="lr schedule"+label_list[i])

            #ax[0].plot(steps,train_loss,label="training loss lr:1e-4")
            #ax[0].plot(steps,val_loss,label="validation loss lr:1e-4")

            ax.set_title('learning rate schedule')
            ax.set_ylabel('learning rate')
            ax.set_xlabel('timestep')
            ax.legend()

            #ax[1].plot(steps,train_accs,label="training accuracy "+label_list[i]) #"training accuracy "+
            #ax[1].plot(steps,val_accs,'--',label="validation accuracy "+label_list[i])

            #ax[1].plot(steps2,train_accs2,label="training accuracy lr:1e-4")
            #ax[1].plot(steps2,val_accs2,label="validation accuracy lr:1e-4")

            #ax[1].set_title('model accuracy')
            #ax[1].set_ylabel('accuracy')
            #ax[1].set_xlabel('timestep')
            #ax[1].legend()
            
    plt.show()
            #plt.tight_layout()
            #plt.savefig(args.output_img,format='svg')
    
    main()