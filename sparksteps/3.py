import os
import sys
import json
import time
import datetime
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

os.environ["HADOOP_USER_NAME"] = "xxx"


K_job = 500
K_person = 1000  # number of person classes ### same as the K in 16.data_xxx.ipynb
numTopics = 1000  
test_days = 60
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

log = spark.read.parquet(f'{file_path}/sparksteps/0data/202code2_query_{city_short}/log')
log_train = log.filter(F.col('date').isin(dates_train)).cache()
log_test = log.filter(F.col('date').isin(dates_test)).cache()
log.show(20)
log.printSchema()
print('relation cnt:', log.count())
print('log_train cnt:', log_train.count())
print('log_test cnt:', log_test.count())

# For training data
person_train = spark.read.parquet(f'{file_path}/sparksteps/train/person')\
                        .select(F.col('geek_id').alias('geekID'),F.col('prediction').alias('pred_person'),F.col('date').alias('date_person'))
person_train.show(1)
print('person train cnt:', person_train.count())

# For testing data
person_test = spark.read.parquet(f'{file_path}/sparksteps/test/person')\
                        .select(F.col('geek_id').alias('geekID'),F.col('prediction').alias('pred_person'),F.col('date').alias('date_person'))
person_test.show(1)
print('person test cnt:', person_test.count())

# For training data
tmp_person_train = person_train.distinct()
print('tmp person train cnt:', tmp_person_train.count())

# For testing data
tmp_person_test = person_test.distinct()
print('tmp person test cnt:', tmp_person_test.count())

# For training data
occ_train = spark.read.parquet(f'{file_path}/sparksteps/train/occ')\
                    .select(F.col('job_id').alias('jobID'),F.col('prediction').alias('pred_job'),F.col('date').alias('date_job'))
occ_train.show(1)
print('occ train cnt:', occ_train.count())

# For testing data
occ_test = spark.read.parquet(f'{file_path}/sparksteps/test/occ')\
                    .select(F.col('job_id').alias('jobID'),F.col('prediction').alias('pred_job'),F.col('date').alias('date_job'))
occ_test.show(1)
print('occ test cnt:', occ_test.count())

# For training data
tmp_occ_train = occ_train.distinct()
print('tmp occ train cnt:', tmp_occ_train.count())

# For testing data
tmp_occ_test = occ_test.distinct()
print('tmp occ test cnt:', tmp_occ_test.count())

# For training data
log_train = log_train.join(person_train, (log_train.geek_id == person_train.geekID) & (log_train.ds == person_train.date_person), 'left')\
            .drop("geekID", 'date_person').filter(F.col('pred_person').isNotNull())
log_train = log_train.join(occ_train, (log_train.job_id == occ_train.jobID) & (log_train.ds == occ_train.date_job), 'left')\
            .drop('jobID', 'date_job').filter(F.col('pred_job').isNotNull())
log_train.show(1)
print('log train cnt:', log_train.count())

# For testing data
log_test = log_test.join(person_test, (log_test.geek_id == person_test.geekID) & (log_test.ds == person_test.date_person), 'left')\
            .drop("geekID", 'date_person').filter(F.col('pred_person').isNotNull())
log_test = log_test.join(occ_test, (log_test.job_id == occ_test.jobID) & (log_test.ds == occ_test.date_job), 'left')\
            .drop('jobID', 'date_job').filter(F.col('pred_job').isNotNull())
log_test.show(1)
print('log test cnt:', log_test.count())

# For training data
log_train = log_train.orderBy('pred_person', 'pred_job', 'geek_id', 'job_id', 'date')
log_train.show(1)

# For testing data
log_test = log_test.orderBy('pred_person', 'pred_job', 'geek_id', 'job_id', 'date')
log_test.show(1)

# log.orderBy('geek_id', 'deal_time', 'pred_person', 'pred_job', 'job_id').show(10000)

log_train.write.mode('overwrite').parquet(f'{file_path}/sparksteps/train/log')
log_test.write.mode('overwrite').parquet(f'{file_path}/sparksteps/test/log')
print('log train cnt:', log_train.count())
print('log test cnt:', log_test.count())

print('log train cnt:', log_train.distinct().count())
print('log test cnt:', log_test.distinct().count())