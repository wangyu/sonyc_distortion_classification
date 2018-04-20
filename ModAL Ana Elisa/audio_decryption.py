
# coding: utf-8

# In[9]:


import utils
import tarfile
import librosa.display
import IPython.display
from datetime import datetime
import os


# In[1]:


get_ipython().system(u'which python')


# In[2]:


get_ipython().system(u'conda env list')


# In[10]:


import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
get_ipython().magic(u'matplotlib inline')


# In[11]:


def timestamp_to_date(timestamp):
    date = datetime.fromtimestamp(timestamp)
    return str(date)[:10]


# ## Decrypting with given sensor id and timestamp

# In[12]:


def decrypt_and_load_audio(audio_path, sensor_id, timestamp, sample_rate=44100, frame=None, **kwargs):
    #given sensor id and timestamp, decrypt and load audio for playback
    #audio_path: str
    #sensor_id: str
    #timestamp: float
    #**kwargs for decrypt
    
    date = timestamp_to_date(timestamp)
    day_tar_filepath = os.path.join(audio_path,'sonycnode-'+sensor_id+'.sonyc', date+'.tar')
    enc_tar_filename = os.path.join('.', sensor_id + '_' + str(timestamp) + '.tar.gz')
    
    y, sr, idt = utils.read_encrypted_tar_audio_file_from_day_tar(day_tar_filepath, 
                                                                enc_tar_filename, 
                                                                sample_rate, **kwargs)

    # if frame is specified, trim the audio to frame-1:frame+1

    if frame is not None:
        y = y[sr*frame:sr*(frame+1)]
        
    return IPython.display.Audio(data=y, rate=sr)


# In[14]:


audio_path = '/beegfs/work/sonyc/audio/'
sensor_id = 'b827eb8e2420'
timestamp = 1490060542.09
frame = 2
decrypt_and_load_audio(audio_path, sensor_id, timestamp, sample_rate=44100, frame = frame,
                       url='https://decrypt-sonyc.engineering.nyu.edu/decrypt', 
                       cacert='CA.pem', 
                       cert='ana_elisa_data.pem',
                       key='aemm_key.pem')


# ## Decrypting for a specific date

# In[6]:


day_tar_filepath = '/beegfs/work/sonyc/audio/sonycnode-b827eb9bed23.sonyc/2017-04-04.tar'
tar = tarfile.open(day_tar_filepath)


# In[9]:


tar.getmembers()[1]


# In[10]:


enc_tar_filename = './b827eb9bed23_1491319570.98.tar.gz'


# In[11]:


y, sr, idt = utils.read_encrypted_tar_audio_file_from_day_tar(day_tar_filepath, 
                                                                enc_tar_filename, 
                                                                sample_rate=44100, 
                                                                url='https://decrypt-sonyc.engineering.nyu.edu/decrypt', 
                                                                cacert='CA.pem', 
                                                                cert='ana_elisa_data.pem',
                                                                key='aemm_key.pem')


# In[12]:


IPython.display.Audio(data=y, rate=sr)


# ## Unfinished function

# In[ ]:


def decrypt_and_load_audio_from_day(audio_path, sensor_id, date, start, end, **kwargs):
    
    day_tar_filepath = os.path.join(audio_path,'sonycnode-'+sensor_id+'.sonyc', date+'.tar')
    tar = tarfile.open(day_tar_filepath)
    members = tar.getmembers()[start:end]
    for member in members:
        enc_tar_filepath = str(tar.getmembers()[member]).split()

