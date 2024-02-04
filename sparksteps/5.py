import os
import sys
import json
import time
import datetime
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

os.environ["HADOOP_USER_NAME"] = "xxx"


K_job = 500
K_person = 1000  # number of person classes ### same as the K in 16.data_xxx.ipynb
numTopics = 1000  
test_days = 60
ddays = 7
LIST_CNT = 30
start_date = sys.argv[1]
end_date = sys.argv[2]
path = sys.argv[3]
file_path = 'file://' + path
city_short = sys.argv[4]

def generate_train_test_dates(start_date, end_date):
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list_train = [(start + datetime.timedelta(days=x)).strftime('%Y-%m-%d') 
                 for x in range(0, (end - start).days + 1 - test_days)]
    date_list_test = [(start + datetime.timedelta(days=x)).strftime('%Y-%m-%d') 
                 for x in range((end - start).days + 1 - test_days, (end - start).days + 1)]
    
    return date_list_train, date_list_test

dates_train, dates_test = generate_train_test_dates(start_date, end_date)
print('dates tain:', dates_train)
print('dates test:', dates_test)

spark = SparkSession\
    .builder\
    .appName("spark_data_query")\
    .config("spark.sql.shuffle.partitions",500)\
    .config("spark.driver.memory", "32g")\
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# For training data
folder_path_train = f'{path}/train/log_res'
parquet_files = [os.path.join(folder_path_train, f) for f in os.listdir(folder_path_train) if f.endswith('.parquet')]
df_train = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
# display(df_train)
print('count:', df_train.shape[0])

# For testing data
folder_path_test = f'{path}/test/log_res'
parquet_files = [os.path.join(folder_path_test, f) for f in os.listdir(folder_path_test) if f.endswith('.parquet')]
df_test = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
# display(df_test)
print('count:', df_test.shape[0])

# For training data
df_train = df_train[df_train['list_size'] > LIST_CNT]
# display(df_train)
print('count:', df_train.shape[0])

# For testing data
df_test = df_test[df_test['list_size'] > LIST_CNT]
# display(df_test)
print('count:', df_test.shape[0])

# For training data
if not os.path.exists(f'{path}/train/relation'):
    os.mkdir(f'{path}/train/relation')
cnt_dict_train = {i:0 for i in range(K_person)}
for index, row in df_train.iterrows():
#     print(index, row)
    pred_job_list = row[6]
    pred_person = row[5][0]
    job_id_list = row[1]
    geek_id = row[0]
    date_list = row[4]
    unixtime_list = row[3]
    cnt = cnt_dict_train[pred_person]
#     print(str(pred_person) + ':'+str(cnt))
    relation_file = open(f'{path}/train/relation/relation_'+str(pred_person)+'_'+str(cnt)+'.csv', 'w')
    relation_file.write('id,pred_job,unix_time,date,job_id,geek_id\n')
    relation_file.close()
    relation_file = open(f'{path}/train/relation/relation_'+str(pred_person)+'_'+str(cnt)+'.csv', 'a')
    for i in range(len(pred_job_list)):
        relation_file.write(str(i)+','+str(pred_job_list[i])+','+str(unixtime_list[i])+','+str(date_list[i])+','+str(job_id_list[i])+','+str(geek_id)+'\n')
    relation_file.close()
    cnt_dict_train[pred_person] = cnt + 1
    
print(len(cnt_dict_train))

# For testing data
if not os.path.exists(f'{path}/test/relation'):
    os.mkdir(f'{path}/test/relation')
cnt_dict_test = {i:0 for i in range(K_person)}
for index, row in df_test.iterrows():
#     print(index, row)
    pred_job_list = row[6]
    pred_person = row[5][0]
    job_id_list = row[1]
    geek_id = row[0]
    date_list = row[4]
    unixtime_list = row[3]
    cnt = cnt_dict_test[pred_person]
#     print(str(pred_person) + ':'+str(cnt))
    relation_file = open(f'{path}/test/relation/relation_'+str(pred_person)+'_'+str(cnt)+'.csv', 'w')
    relation_file.write('id,pred_job,unix_time,date,job_id,geek_id\n')
    relation_file.close()
    relation_file = open(f'{path}/test/relation/relation_'+str(pred_person)+'_'+str(cnt)+'.csv', 'a')
    for i in range(len(pred_job_list)):
        relation_file.write(str(i)+','+str(pred_job_list[i])+','+str(unixtime_list[i])+','+str(date_list[i])+','+str(job_id_list[i])+','+str(geek_id)+'\n')
    relation_file.close()
    cnt_dict_test[pred_person] = cnt + 1
    
print(len(cnt_dict_test))

# For training data
new_dict = {key: value for key, value in cnt_dict_train.items() if value != 0}
print(new_dict)
print('count:', len(new_dict))

# For testing data
new_dict = {key: value for key, value in cnt_dict_test.items() if value != 0}
print(new_dict)
print('count:', len(new_dict))

import pickle as pkl

# For training data
with open(f'{path}/train/relation/dict.pkl', 'wb') as f:
    pkl.dump(cnt_dict_train, f)

# For testing data
with open(f'{path}/test/relation/dict.pkl', 'wb') as f:
    pkl.dump(cnt_dict_test, f)