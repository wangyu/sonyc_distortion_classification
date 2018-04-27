
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

positive_clusters = [9,11,24,25,40,49,52,60,61,75,78,94,95,106,107,124,129]

test_sensors = ['b827eb0d8af7', 'b827eb0fedda', 'b827eb122f0f', 'b827eb1685c7','b827eb2a1bce', 'b827eb429cd4', 'b827eb42bd4a', 'b827eb44506f', 'b827eb4e7821', 'b827eb5895e9', 'b827eb815321', 'b827eb86d458','b827eb8e2420', 'b827eb9bed23','b827ebad073b']

def split_frames_based_on_cluster():
    frames_in_positive_clusters = []
    frames_in_negative_clusters = []
    
    for d in d_frames:
        if d['assignment'] in positive_clusters:
            frames_in_positive_clusters.append((d['sensor_id'].decode('UTF-8')+'_'+str(d['timestamp'])).encode('UTF-8'))
        else:
            frames_in_negative_clusters.append((d['sensor_id'].decode('UTF-8')+'_'+str(d['timestamp'])).encode('UTF-8'))
    
    pickle.dump(frames_in_positive_clusters, open("frames_in_positive_clusters.pickle", "wb" ))
    pickle.dump(frames_in_positive_clusters, open("frames_in_positive_clusters.pickle", "wb" ))

audio_path = '/beegfs/work/sonyc/audio/'

def timestamp_to_date(timestamp):
    date = datetime.fromtimestamp(timestamp)
    return str(date)[:10]

def build_pool(num_samples):
    #initiate arrays, first 128 dimension: VGG, followed bt sensor id, timestamp, frame
    positive_pool = np.empty([num_samples//2, 132], dtype=object)
    negative_pool = np.empty([num_samples//2, 132], dtype=object)
    
    positive_samples = 0
    negative_samples = 0
    
    #generate random index sequence
    idx = list(range(len(d_frames)))
    random.shuffle(idx)
    
    for ind in idx:
        #if got enough samples, stop the loop
        if negative_samples ==  positive_samples == num_samples//2:
            break
            
        #check not in test sensors list
        sensor_id = d_frames[ind]['sensor_id'].decode('UTF-8')
        if sensor_id not in test_sensors:
            
            #if assigned to negative clusters, add to negative samples
            if d_frames[ind]['assignment'] not in positive_clusters and negative_samples < num_samples//2:
                timestamp = d_frames[ind]['timestamp']
                frame = d_frames[ind]['frame']
                
                #check if it is in d_features
                identifier = (sensor_id+'_'+str(timestamp)).encode('UTF-8')
                where_in_dfeatures = np.where(d_features['identifier'] == identifier)
                if where_in_dfeatures[0].shape[0] == 1:
                    #update negative pool
                    negative_pool[negative_samples][:128] = d_features[where_in_dfeatures[0][0]]['features_z'][frame]
                    negative_pool[negative_samples][128] = sensor_id
                    negative_pool[negative_samples][129] = timestamp
                    negative_pool[negative_samples][130] = frame
                    
                    feature_path = d_features[where_in_dfeatures[0][0]]['path'].decode('UTF-8')
                    decrypt_path = os.path.join(audio_path, os.path.split(os.path.dirname(feature_path))[-1], 
                                               os.path.splitext(feature_path)[0]+'.tar')
                    negative_pool[negative_samples][131] = decrypt_path
                    negative_samples+=1
       
            #if assigned to positive clusters, add to positive samples
            elif d_frames[ind]['assignment'] in positive_clusters and positive_samples < num_samples//2:
                timestamp = d_frames[ind]['timestamp']
                frame = d_frames[ind]['frame']
            
                #check if it is in d_features
                identifier = (sensor_id+'_'+str(timestamp)).encode('UTF-8')
                where_in_dfeatures = np.where(d_features['identifier'] == identifier)
                if where_in_dfeatures[0].shape[0] == 1:
                    #update positive pool
                    positive_pool[positive_samples][:128] = d_features[where_in_dfeatures[0][0]]['features_z'][frame]
                    positive_pool[positive_samples][128] = sensor_id
                    positive_pool[positive_samples][129] = timestamp
                    positive_pool[positive_samples][130] = frame
                    
                    feature_path = d_features[where_in_dfeatures[0][0]]['path'].decode('UTF-8')
                    decrypt_path = os.path.join(audio_path, os.path.split(os.path.dirname(feature_path))[-1],
                                               os.path.splitext(feature_path)[0]+'.tar')
                    positive_pool[positive_samples][131] = decrypt_path
                    positive_samples+=1
                    
    X_pool = np.concatenate((positive_pool, negative_pool),axis=0)
            
    return X_pool

X_pool= build_pool(100000)

with open('/scratch/yw3004/sonyc/sonyc_distortion_classification/data/X_pool_100000_new.pickle', 'wb') as handle:
    pickle.dump(X_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
