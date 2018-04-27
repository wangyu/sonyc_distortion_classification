
# coding: utf-8

import pickle
import h5py as h5
import numpy as np
import sklearn
import modAL
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
import collections
import csv

features = h5.File('/scratch/yw3004/sonyc/sonyc_distortion_classification/_old_features.h5', 'r')
d_features = list(features.values())[0]

alldata = {}
for i in range(len(d_features)):
    sensor_timestamp = d_features[i][1]
    
    if len(sensor_timestamp.decode("utf-8").split('_')) ==1:
        sensor_id = sensor_timestamp.decode("utf-8").split('-')[0]
        timestamp = sensor_timestamp.decode("utf-8").split('-')[1]
    else:     
        sensor_id=sensor_timestamp.decode("utf-8").split('_')[0]
        timestamp=sensor_timestamp.decode("utf-8").split('_')[1]
    
    features=d_features[i][2]
    
    if sensor_id not in alldata.keys():
        alldata[sensor_id] = {timestamp: features}
    else:
        alldata[sensor_id][timestamp]=features 

with open("/scratch/yw3004/sonyc/sonyc_distortion_classification/data/features.pickle", 'wb') as pfile:
    pickle.dump(alldata, pfile, protocol=pickle.HIGHEST_PROTOCOL)
