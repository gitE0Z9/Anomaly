# Anomaly
Time series anomaly detector with deep learning

> This project aims to provide an alerting tool for monitoring a cluster with hundreds of nodes.

_*Detect mode*_

```python
  from Anomaly.detect import detector
  
   # model_tyep includes lstm, cnn and custom. If the custom is used, please provice th path to the model

  model=detector(model_type='lstm')

  # read data from specific path csv

  model.read_data(hostname='PNT16', item='cpu', date='20190115', time_field='Date', data_field='% Processor Time')

   # critical_value is optimized manually due to no label.
   
  model.detect(critical_value=-13.5)

```

_*Train mode*_

```python
  from Anomaly.train import training

  new_model=training()

  # cnn and lstm are available.
  
  new_model.create_cnn()
  
  #training history is returned

  history = new_model.train(train,validate)

  #save model

  new_model.save('new_model.h5')

```
