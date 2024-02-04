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
ddays = 7
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
log_train = spark.read.parquet(f'{file_path}/train/log').cache()
log_train.show(1)
# person.printSchema()
print('log train cnt:', log_train.count())

# For testing data
log_test = spark.read.parquet(f'{file_path}/test/log').cache()
log_test.show(1)
# person.printSchema()
print('log test cnt:', log_test.count())

# For training data



windowSpec = Window.partitionBy("geek_id").orderBy("deal_time")

# log = log.withColumn("next_deal_time", F.lead("deal_time", 1).over(windowSpec))
# log = log.withColumn("next_pred_person", F.lead("pred_person", 1).over(windowSpec))

# time_diff = (F.col("next_deal_time") - F.col("deal_time")) / 86400
# log = log.withColumn("time_diff", time_diff)
# log = log.filter((F.col("pred_person") == F.col("next_pred_person")) & (F.col("time_diff") <= ddays))

log_train = log_train.withColumn("prev_deal_time", F.lag("deal_time").over(windowSpec))
log_train = log_train.withColumn("time_diff", F.when(F.isnull(log_train.deal_time - log_train.prev_deal_time), 0)
                                  .otherwise(log_train.deal_time - log_train.prev_deal_time))

log_train = log_train.withColumn("new_record", F.when(log_train.time_diff > ddays * 86400, 1).otherwise(0))

log_train = log_train.withColumn("group_id", F.sum("new_record").over(windowSpec))

# from pyspark.ml.feature import StringIndexer
# inputs1 = ["job_id"]
# outputs1 = ["job_code"]
# stringIndexer1 = StringIndexer(inputCols=inputs1,outputCols=outputs1).setHandleInvalid('keep')
# model1 = stringIndexer1.fit(log_train)
# model1.write().overwrite().save(f'{file_path}/models/StringIndexer_prepare')
# log_train = model1.transform(log_train)

# min_job_code = log.select(F.min("job_code")).first()[0]
# max_job_code = log.select(F.max("job_code")).first()[0]

# print(f"min new job id：{min_job_code}")
# print(f"max new job id：{max_job_code}")

log_train = log_train.groupBy("geek_id", "group_id").agg(
    F.collect_list("job_id").alias("job_id_list"),
    F.collect_list("deal_type").alias("deal_type_list"),
    F.collect_list("deal_time").alias("deal_time_list"),
    F.collect_list("date").alias("date_list"),
    F.collect_list("pred_person").alias("pred_person_list"),
    F.collect_list("pred_job").alias("pred_job_list")
).drop("group_id").withColumn("list_size", F.size("job_id_list"))
log_train.show()

# For testing data



windowSpec = Window.partitionBy("geek_id").orderBy("deal_time")
log_test = log_test.withColumn("prev_deal_time", F.lag("deal_time").over(windowSpec))
log_test = log_test.withColumn("time_diff", F.when(F.isnull(log_test.deal_time - log_test.prev_deal_time), 0)
                                  .otherwise(log_test.deal_time - log_test.prev_deal_time))

log_test = log_test.withColumn("new_record", F.when(log_test.time_diff > ddays * 86400, 1).otherwise(0))

log_test = log_test.withColumn("group_id", F.sum("new_record").over(windowSpec))

# from pyspark.ml.feature import StringIndexerModel
# inputs1 = ["job_id"]
# outputs1 = ["job_code"]
# if os.path.exists(f'{file_path}/models/StringIndexer_prepare'):
#     model1 = StringIndexerModel.load(f'{file_path}/models/StringIndexer_prepare')
# log_test = model1.transform(log_test)

log_test = log_test.groupBy("geek_id", "group_id").agg(
    F.collect_list("job_id").alias("job_id_list"),
    F.collect_list("deal_type").alias("deal_type_list"),
    F.collect_list("deal_time").alias("deal_time_list"),
    F.collect_list("date").alias("date_list"),
    F.collect_list("pred_person").alias("pred_person_list"),
    F.collect_list("pred_job").alias("pred_job_list")
).drop("group_id").withColumn("list_size", F.size("job_id_list"))
log_test.show()


# For training data
log_train.write.mode('overwrite').parquet(f'file:///individual/hanxiao/train/log_res')
print('log train cnt:', log_train.count())

# For testing data
log_test.write.mode('overwrite').parquet(f'file:///individual/hanxiao/test/log_res')
print('log test cnt:', log_test.count())

