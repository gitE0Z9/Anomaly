import os
import matplotlib.pyplot as plt
from .output import *
from .preprocess import preprocess_lstm
from pandas import read_csv, to_datetime, Timedelta
from numpy import mean, std, cov, pad
from scipy.stats import norm, multivariate_normal
from keras.models import load_model

class detector():
    """
    Time series anomaly detector
        Please specify the type of model: cnn, lstm, custom.
    """
    def __init__(self,model_type,model_path=None):

        assert model_type != "", "Please specify the type of model: cnn, lstm, custom."
        self.model_type=model_type

        #choose model
        if self.model_type == "cnn":
            self.model_path=os.path.dirname(__file__)+"/model/cnn.h5"
        elif self.model_type == "lstm":
            self.model_path=os.path.dirname(__file__)+"/model/lstm.h5"
        elif self.model_type == "custom":
            assert model_path!=None , "Please provide the path to the custom model in the model_path argument."
            assert os.path.isfile(model_path) , "File doesn't exist."
            self.model_path=model_path
        
        # load model

        self.model=load_model(self.model_path)
        self.timestep = self.model.layers[0].input_shape[1]
        self.feature_num = self.model.layers[0].input_shape[2]

        self.model.summary()

    def detect(self,hostname,data_path,critical_value,fordstep=None,t_limit=3600,density=20):
        """
        Detect anomaly and output time interval in plot and text fashion.
        """

        # read data
        x = read_csv(data_path).loc[:,['Date', '% Processor Time']]
        x['Date'] = to_datetime(x['Date'],unit='s')+ Timedelta('08:00:00')
        sample_size = len(x)
        time_field = x['Date']
        data_field = x['% Processor Time']
        data = preprocess_lstm(x,'% Processor Time',self.timestep,mode="testing")
        del x

        #generate output dir
        output_dir(hostname,data_path)
        
        # predict error
        total_hour = max(time_field.dt.hour) + 1
        for j in range(total_hour):
            maxima=int((sample_size/total_hour)*(j+1))

            if fordstep:
                error = self.model.predict(data[0:max(maxima-2*self.timestep,0),:,:])[:,0:fordstep,:] - data[self.timestep:max(maxima-self.timestep,0),0:fordstep,:]
                error = error.reshape(error.shape[0],error.shape[1])
                mle = multivariate_normal(mean(error,axis=0),cov(error.transpose())).logpdf(error) # multiple variable multiple step
            else:
                error = self.model.predict(data[0:max(maxima-2*self.timestep,0),:,:]) - data[self.timestep:max(maxima-self.timestep,0),0,:]      
                if self.feature_num > 1:
                    mle = norm(mean(error[:,0]),std(error[:,0])).logpdf(error[:,0]) # multiple variable single step
                else:
                    mle = norm(mean(error),std(error)).logpdf(error) # single variable single step
            raw_alert = mle < critical_value
            raw_alert = pad(raw_alert,((self.timestep,self.timestep),(0,0)),'constant',constant_values=(False,False)).reshape(-1) #padding
            
            # alert
            filtered_alert=time_field[0:maxima][raw_alert].diff(density).dt.seconds
            time_interval = []
            for i in range(density+1,len(filtered_alert)):
                if filtered_alert.iloc[i] <= t_limit:
                    start = time_field[filtered_alert.index[i-density]]
                    end = time_field[filtered_alert.index[i]]
                    if len(time_interval)==0:
                        time_interval.append([start,end])
                    else:
                        if time_interval[-1][1] >= start >= time_interval[-1][0]:
                            time_interval[-1][1]==end
                        else:
                            time_interval.append([start,end])                    

            # draw data and write text
            if len(time_interval) > 0:
                print('%s got alerted at %d hour on %s' %(hostname, j, data_path[-12:-4]))
                drawing(hostname,data_path,time_field,data_field,mle,maxima,self.timestep,self.feature_num,sample_size,t_limit,time_interval,critical_value,hour=j)