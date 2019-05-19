from numpy import min, max, expand_dims, array,squeeze
from sklearn.preprocessing import MinMaxScaler

def preprocess_lstm(x,name,t,mode="training"):
    if mode=="training":
        norm=x[name].values
        norm=(norm-min(norm))/(max(norm)-min(norm))
        tmp=[norm[i:i+t] for i in range(len(x)-t)]
    else:
        tmp=[x[name].values[i:i+t] for i in range(len(x)-t)]
    tmp=array(tmp)
    tmp=expand_dims(tmp,axis=2)
    return tmp

def data_prepare(x,t,multi_step,test=True):
    tmp=preprocess_lstm(x,'% Processor Time',t)
    tmpf=0
    if test:
        if multi_step:
            tmpf=t
            y_test=tmp[(int(len(x)*0.8)+1+t):len(x),0:tmpf+1,:]
            y_train=tmp[t:int(len(x)*0.8)+t,0:tmpf+1,:]
        else:
            y_test=squeeze(tmp[(int(len(x)*0.8)+1+t):len(x),0:tmpf+1,:],axis=(2,))
            y_train=squeeze(tmp[t:int(len(x)*0.8)+t,0:tmpf+1,:],axis=(2,))
        x_test=tmp[(int(len(x)*0.8)+1):(len(x)-t*2),:,:]
        x_train=tmp[0:int(len(x)*0.8),:,:]
        return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test}
    else:
        x_train=tmp[0:len(x)-2*t,:,:]
        y_train=squeeze(tmp[t:len(x),0:tmpf+1,:],axis=(2,))
        return {'x_train':x_train,'y_train':y_train}