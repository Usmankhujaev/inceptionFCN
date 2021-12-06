import os
from os import listdir
from posixpath import split
import numpy as np
import pandas as pd
from pandas.core.arrays.integer import UInt32Dtype

data_path="./icnv2_TSC_1/TSC_itr_2/"
results_file = 'df_metrics.csv'
folders = [f for f in listdir(data_path) if (os.path.join(data_path, f))]
save_path = '/home/lidar/projects/InceptionTime/results/' + data_path
save_file_name= 'results_TSC_itr_2.csv'
colum = ['Dataset','Presicion','Accuracy','Recall', 'Duration']
rows=[]
for i in folders:
    file_path = data_path+i+'/'+results_file
    #print(file_path)
    try:
        df=pd.read_csv(file_path)
    except:FileNotFoundError
    df.insert(0, 'Dataset', i, True)
    df['precision']= df['precision'].round(decimals=3)
    df['accuracy'] = df['accuracy'].round(decimals=3)
    df['recall'] = df['recall'].round(decimals=3)
    df['duration']= df['duration'].round(decimals=3)

    
    df.to_csv(save_path+save_file_name,mode='a',header=False, index=False, encoding='utf-8')
    #print(df.astype(str).values.tolist())
    print(save_path+save_file_name)