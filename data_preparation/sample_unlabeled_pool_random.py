
# coding: utf-8

from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import h5py
import pickle
from sklearn.ensemble import RandomForestClassifier
import random
import os.path
from datetime import datetime


features = h5py.File('/scratch/yw3004/sonyc/sonyc_distortion_classification/_old_features.h5', 'r')
d_features = list(features.values())[0]

frames = h5py.File('/scratch/yw3004/sonyc/sonyc_distortion_classification/clusters_frames.hdf5', 'r')
d_frames = list(frames.values())[0]

test_sensors = ['b827eb0d8af7', 'b827eb0fedda', 'b827eb122f0f', 'b827eb1685c7','b827eb2a1bce', 'b827eb429cd4', 'b827eb42bd4a', 'b827eb44506f', 'b827eb4e7821', 'b827eb5895e9', 'b827eb815321', 'b827eb86d458','b827eb8e2420', 'b827eb9bed23', 'b827ebad073b']

audio_path = '/beegfs/work/sonyc/audio/'

def timestamp_to_date(timestamp):
    date = datetime.fromtimestamp(timestamp)
    return str(date)[:10]

def build_random_pool(num_samples):
    
    pool = np.empty([num_samples, 132], dtype=object)
    idx = list(range(len(d_frames)))
    random.shuffle(idx)
    
    n_sample = 0
    
    for ind in idx:
        if n_sample == num_samples:
            break
        #check if in test sensor lists
        sensor_id = d_frames[ind]['sensor_id'].decode('UTF-8')
        if sensor_id not in test_sensors:
            timestamp = d_frames[ind]['timestamp']
            frame = d_frames[ind]['frame']
            identifier = (sensor_id+'_'+str(timestamp)).encode('UTF-8')
            where_in_dfeatures = np.where(d_features['identifier'] == identifier)
            #check if in d_features
            if where_in_dfeatures[0].shape[0] == 1:
                pool[n_sample][:128] = d_features[where_in_dfeatures[0][0]]['features_z'][frame]
                pool[n_sample][128] = sensor_id
                pool[n_sample][129] = timestamp
                pool[n_sample][130] = frame
                    
                feature_path = d_features[where_in_dfeatures[0][0]]['path'].decode('UTF-8')
                decrypt_path = os.path.join(audio_path, os.path.split(os.path.dirname(feature_path))[-1], 
                                               os.path.splitext(feature_path)[0]+'.tar')
                pool[n_sample][131] = decrypt_path
                n_sample+=1
            
    return pool


X_pool_100000_random = build_random_pool(100000)

with open('/scratch/yw3004/sonyc/sonyc_distortion_classification/data/X_pool_1000000_random.pickle', 'wb') as handle:
    pickle.dump(X_pool_100000_random, handle, protocol=pickle.HIGHEST_PROTOCOL)

