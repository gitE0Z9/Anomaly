from os import makedirs
from os.path import dirname
import matplotlib.pyplot as plt

# generate output directory

def output_dir(hostname,data_path):
    try:
        text_path=dirname(__file__)+'/alert/text/{}/{}'.format(hostname,data_path[-12:-4])
        plot_path=dirname(__file__)+'/alert/plot/{}/{}'.format(hostname,data_path[-12:-4])
        makedirs(text_path)
        makedirs(plot_path)
    except FileExistsError:
        print('Output directory exists')
    except Exception:
        print('Output directory failed to be made.')
        print(Exception)

def drawing(hostname,data_path,time_field,data_field,mle,maxima,timestep,feature_num,sample_size,t_limit,time_interval,critical_value,hour):
    text_path=dirname(__file__)+'/alert/text/{}/{}'.format(hostname,data_path[-12:-4])
    plot_path=dirname(__file__)+'/alert/plot/{}/{}'.format(hostname,data_path[-12:-4])
    
    plt.figure(figsize=[20,16])
    plt.subplot(2,1,1)
    plt.plot(time_field[0:maxima],data_field[0:maxima])
    plt.xlim(min(time_field),max(time_field))
    plt.ylim(min(data_field),max(data_field))

    for interval in time_interval:
        plt.axvspan(interval[0],interval[1],color='red',alpha=1)
        with open('{}/{}.txt'.format(text_path,hour), 'a') as f :
            f.write(','.join(map(str,interval))+"\n")
    
    plt.title("%d minute filtered result" %(int(t_limit/60)), loc="left", fontsize=20)
    
    # draw error
    plt.subplot(2,1,2)
    plt.plot(mle)
    plt.xlim(0,sample_size)
    plt.ylim(-20,1)
    plt.axhline(critical_value,color='green')
    plt.title('error',loc="left",fontsize=20)
    
    # save result
    plt.savefig('{}/{}.png'.format(plot_path,hour)) 
    
    plt.close()