import os
import matplotlib.pyplot as plt
from .output import *
import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal
from keras.models import load_model


class detector():
    """
    Time series anomaly detector
        Please specify the type of model: cnn, lstm, custom.
    """

    def __init__(self, model_type, model_path=None, verbose=True):

        assert model_type != "", "Please specify the type of model: cnn, lstm, custom."

        # validate model
        if model_type in ["cnn", "lstm"]:
            model_path = f"{os.path.dirname(__file__)}/model/{model_type}.h5"
            assert os.path.exists(model_path), "Model isn't found."
        elif model_type == "custom":
            assert model_path != None, "Please provide the path to the custom model in the model_path argument."
            assert os.path.isfile(model_path), "File doesn't exist."
        else:
            raise Exception('No such kind of model.')

        # load model
        self.model = load_model(model_path)

        if verbose:
            self.model.summary()

        # settings
        self.model_type, self.model_path = model_type, model_path
        _, self.timestep, self.feature_num = self.model.layers[0].input_shape

    def read_data(self, hostname, item, date, time_field, data_field, zone='Asia/Taipei'):
        """
        Read time series from a csv file.
        -----------
        item: cpu, diskIO, networkIO, memory
        date: int e.g. 20190115
        ------------
        """
        path = f'anomaly.experiment/data/{item}/{hostname}/{int(date)//100}/CPU_ALL-{date}.csv'

        assert os.path.exists(path), 'Please check query variables because no file matchs.'
        x = pd.read_csv(path, usecols=[time_field, data_field]) # 'Date', '% Processor Time'

        # time conversion
        x[time_field] = pd.to_datetime(x[time_field], unit='s').dt.tz_localize('UTC').dt.tz_convert(zone)
        
        # column rename to standard
        x.rename(columns={time_field:'Time',
                          data_field:'Value'},
                inplace=True)
        # x.set_index(time_field,inplace=True)

        # settings
        self.data = x
        self.hostname, self.data_path = hostname, path

    def _preprocess_lstm(self):

        tmp=[self.data['Value'].to_numpy()[i:i+self.timestep] for i in range(len(self.data)-self.timestep)]
        tmp=np.array(tmp).reshape(-1,self.timestep,self.feature_num)
        return tmp

    def _generate_error_dist(self,error,multi=None):
        """
        Normal distribution to fit error
        """
        if multi:
            return multivariate_normal(np.mean(error, axis=0), np.cov(error.transpose())).logpdf(error)
        else:
            return norm(np.mean(error), np.std(error)).logpdf(error)

    def detect(self, critical_value=-13, fordstep=None, t_limit=3600, density=20):
        """
        Detect anomaly and return time interval in plot and text fashion.

        params:
        -------------
        critical_value: mle should not over this value
        fordstep: predict more than single step
        time_limit: filter length
        density: the number of anomaly in time filter
        --------------
        """
        # generate output dir
        output_dir(self.hostname, self.data_path)

        # generate time series array
        ts = self._preprocess_lstm()

        # predict error
        sample_size = len(self.data)
        total_hour = np.max(self.data.Time.dt.round('H').dt.hour)

        for j in range(total_hour):
            maxima = (j+1)*sample_size//total_hour

            if fordstep:
                # multiple step
                pred = self.model.predict(ts[0:max(maxima-2*self.timestep, 0), :, :])[:, 0:fordstep, :]
                real = ts[self.timestep:max(maxima-self.timestep, 0), 0:fordstep, :]
                error = (pred - real).reshape(error.shape[0], error.shape[1])
                mle =  self._generate_error_dist(error,multi=True) # multiple variable 
            else:
                # single step
                pred = self.model.predict(ts[0:max(maxima-2*self.timestep, 0), :, :])
                real = ts[self.timestep:max(maxima-self.timestep, 0), 0, :]
                error = pred - real
                mle = self._generate_error_dist(error[:, 0] if self.feature_num > 1 else error)  # multiple /single variable

            # alert
            raw_alert = mle < critical_value # less likelihood
            raw_alert = np.pad(raw_alert, ((self.timestep, self.timestep), (0, 0)),
                            'constant', constant_values=(False, False)).reshape(-1)  # padding

            filtered_alert = self.data.iloc[:maxima]['Time'][raw_alert].diff(density).dt.seconds if len(raw_alert) != 0 else []
            time_interval = []
            for i in range(density+1, len(filtered_alert)):
                if filtered_alert.iloc[i] <= t_limit:
                    start = self.data.Time[filtered_alert.index[i-density]]
                    end = self.data.Time[filtered_alert.index[i]]
                    if len(time_interval) == 0:
                        # empty interval
                        time_interval.append([start, end])
                    else:
                        # extend interval
                        latest_interval = time_interval[-1]
                        if latest_interval[1] >= start >= latest_interval[0]:
                            latest_interval[1] = end
                        else:
                        # new interval
                            time_interval.append([start, end]) 

            # if alert, draw data and write text
            if len(time_interval) > 0:
                print(f'{self.hostname} got alerted at {j} hour on {self.data_path[-12:-4]}')
                drawing(self.hostname, self.data_path, self.data,'Time', 'Value', mle, maxima, self.timestep,
                        self.feature_num, sample_size, t_limit, time_interval, critical_value, hour=j)