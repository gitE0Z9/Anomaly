# Anomaly
Time series anomaly detector with deep learning

> This project aims to provide an alerting tool for monitoring a cluster with hundreds of nodes.

_*Detect mode*_

```
  from Anomaly.detect import detector
  
   # model_tyep includes lstm, cnn and custom. If the custom is used, please provice th path to the model

  model=detector(model_type='lstm') 

   # critical_value is optimized manually due to no label.
   
  model.detect(hostname='PNT16',data_path='anomaly.experiment/data/cpu/PNT16/201901/CPU_ALL-20190115.csv',critical_value=-13.5)

```

_*Train mode*_

```
  from Anomaly.train import training

  new_model=training()

  # cnn and lstm are available.
  
  new_model.create_cnn()
  
  #training history is returned

  history = new_model.train(train,validate)

  #save model

  new_model.save('new_model.h5')

```
