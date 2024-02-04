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
POS_CODE_LOW = '100000'
POS_CODE_HIGH = '110000'
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
    .config("spark.driver.memory", "16g")\
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

occ = spark.read.parquet(f'{file_path}/0data/202code2_query_{city_short}/occupation')
occ_train = occ.filter(F.col('date').isin(dates_train)).cache()
occ_test = occ.filter(F.col('date').isin(dates_test)).cache()
occ.show(1)
occ.printSchema()
print('occ_train cnt:', occ_train.count())
print('occ_test cnt:', occ_test.count())

# 查看job postion 数量
# job_pos = occ.select(F.col('position_lv3_code')).distinct()
job_pos = occ.filter((F.col('position_lv3_code') >= POS_CODE_LOW) & (F.col('position_lv3_code') < POS_CODE_HIGH))\
                .select(F.col('position_lv3_code')).distinct().collect()
job_pos_list = list(set([row['position_lv3_code'] for row in job_pos]))
print('job pos cnt:', len(job_pos_list))
print(job_pos_list)

print('job 数值化1')
# For training data
# occ_train = occ_train.filter(F.col('position_lv3_code').isin(job_pos_list)).select(\
occ_train = occ_train.select(\
                 F.col('job_id').cast(LongType()),\
                 F.col('city_code').alias('city_code_str'), \
                 F.col('city_name'),\
                 F.col('update_time').cast(LongType()),\
                 F.col('position_lv3_code').alias('position_code').cast(IntegerType()),\
                 F.col('low_salary').cast(IntegerType()),\
                 F.col('high_salary').cast(IntegerType()),\
                 F.col('degree_code').cast(IntegerType()),\
                 F.col('experience_code').cast(IntegerType()),\
                 F.col('title').alias('name'),\
                 F.col('days_per_week_u').cast(IntegerType()),\
                 F.col('area_city_code_u').cast(IntegerType()),\
                 F.to_date(F.col('date')).alias('date'),\
                )
print('\tNULL field process...')
occ_train = occ_train.fillna({'days_per_week_u': 5}).fillna(0).fillna("无")
occ_train.show(1)

# For testing data
# occ_test = occ_test.filter(F.col('position_lv3_code').isin(job_pos_list)).select(\
occ_test = occ_test.select(\
                 F.col('job_id').cast(LongType()),\
                 F.col('city_code').alias('city_code_str'), \
                 F.col('city_name'),\
                 F.col('update_time').cast(LongType()),\
                 F.col('position_lv3_code').alias('position_code').cast(IntegerType()),\
                 F.col('low_salary').cast(IntegerType()),\
                 F.col('high_salary').cast(IntegerType()),\
                 F.col('degree_code').cast(IntegerType()),\
                 F.col('experience_code').cast(IntegerType()),\
                 F.col('title').alias('name'),\
                 F.col('days_per_week_u').cast(IntegerType()),\
                 F.col('area_city_code_u').cast(IntegerType()),\
                 F.to_date(F.col('date')).alias('date'),\
                )
occ_test = occ_test.fillna({'days_per_week_u': 5}).fillna(0).fillna("无")
occ_test.show(1)

print('job 数值化2')
from pyspark.ml.feature import StringIndexer, StringIndexerModel

inputs1 = ["name", "city_code_str"]
outputs1 = ["name_code", "city_code"]

# For training data
stringIndexer1 = StringIndexer(inputCols=inputs1,outputCols=outputs1).setHandleInvalid('keep')
model1 = stringIndexer1.fit(occ_train)
model1.write().overwrite().save(f'{file_path}/models/StringIndexer_occ')

occ_train = model1.transform(occ_train)
# for name in inputs1:
#     occ_train = occ_train.drop(name)
occ_train = occ_train.withColumn('area_city_code_u', F.col('city_code')*10000 + F.col('area_city_code_u'))
occ_train.show(1)

# For testing data
if os.path.exists(f'{file_path}/models/StringIndexer_occ'):
    model1 = StringIndexerModel.load(f'{file_path}/models/StringIndexer_occ')

occ_test = model1.transform(occ_test)
# for name in inputs1:
#     occ_test = occ_test.drop(name)
occ_test = occ_test.withColumn('area_city_code_u', F.col('city_code')*10000 + F.col('area_city_code_u'))
occ_test.show(1)

print('job 数值化3')
def experience_change_to_year(num):
    if num <= 100:
        return 0
    elif num == 101:
        return 0
    elif num == 103:
        return 1
    elif num == 104:
        return 1.5
    elif num == 105:
        return 3
    elif num == 106:
        return 5.5
    elif num == 107:
        return 10
    else:
        return 0 # 应届生/在校生
change_udf1 = F.udf(experience_change_to_year, IntegerType())

def degree_change_to_level(num):
    if num ==209:
        return 1
    elif num == 208:
        return 2
    elif num == 206:
        return 3
    elif num == 202:
        return 4
    elif num == 203:
        return 5
    elif num == 204:
        return 6
    elif num == 205:
        return 7
    else:
        return 0
change_udf2 = F.udf(degree_change_to_level, IntegerType())

# For training data
occ_train = occ_train.withColumn("experience_code", change_udf1(occ_train.experience_code))\
            .withColumn("degree_code", change_udf2(occ_train.degree_code))\
            .drop('city_code')
occ_train.show(1)
occ_train.write.mode('overwrite').parquet(f'{file_path}/train/occ_tmp')
occ_train = occ_train.withColumn("position_code", occ_train.position_code-100000)
for name in inputs1:
    occ_train = occ_train.drop(name)

# For testing data
occ_test = occ_test.withColumn("experience_code", change_udf1(occ_test.experience_code))\
            .withColumn("degree_code", change_udf2(occ_test.degree_code))\
            .drop('city_code')
occ_test.show(1)
occ_test.write.mode('overwrite').parquet(f'{file_path}/test/occ_tmp')
occ_test = occ_test.withColumn("position_code", occ_test.position_code-100000)
for name in inputs1:
    occ_test = occ_test.drop(name)
    
occ_train.show(1)
occ_train = occ_train.dropna(how='any')
occ_test.show(1)
occ_test = occ_test.dropna(how='any')

from pyspark.ml.feature import VectorAssembler


vectorAssembler = VectorAssembler(inputCols=['position_code', 'low_salary', 'high_salary', 'degree_code', \
                                             'experience_code', 'days_per_week_u', 'area_city_code_u', 'name_code'], \
                                  outputCol="features")
# tansdata = vectorAssembler.transform(occ).select('job_id', 'features', 'date')
# tansdata.show(1)

# For training data
tansdata_train = vectorAssembler.transform(occ_train).select('job_id', 'features', 'date')
tansdata_train.show(1)

# For testing data
tansdata_test = vectorAssembler.transform(occ_test).select('job_id', 'features', 'date')
tansdata_test.show(1)



print('数据归一化')
from pyspark.ml.feature import StandardScaler, StandardScalerModel

standardScaler = StandardScaler(inputCol="features",outputCol="scaledFeatures")
print(1)
# For training data
model_one = standardScaler.fit(tansdata_train)
model_one.write().overwrite().save(f'{file_path}/models/StandardScaler_occ')

print(2)
stddata_train = model_one.transform(tansdata_train)
stddata_train.show(1)

# For testing data
if os.path.exists(f'{path}/hanxiao/models/StandardScaler_occ'):
    model_one = StandardScalerModel.load(f'{file_path}/models/StandardScaler_occ')

stddata_test = model_one.transform(tansdata_test)
stddata_test.show(1)



# import sklearn.cluster as cluster
from pyspark.ml.clustering import KMeans, KMeansModel


# # traindata,testdata = stddata.randomSplit([0.8,0.2])
# traindata = stddata.filter(F.col('date').isin(dates_train))
# testdata = stddata.filter(F.col('date').isin(dates_test))
# print(traindata.count(),testdata.count())

print('KMeans')
kmeans = KMeans(featuresCol="scaledFeatures",k=K_job,seed=1)

# For training data
model_two = kmeans.fit(stddata_train) # traindata<->stddata
model_two.write().overwrite().save(f'{file_path}/models/KMeans_occ')
predictions_train = model_two.transform(stddata_train) # testdata<->stddata

# For testing data
if os.path.exists(f'{file_path}/models/KMeans_occ'):
    model_two = KMeansModel.load(f'{file_path}/models/KMeans_occ')

predictions_test = model_two.transform(stddata_test) # testdata<->stddata

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator

if os.path.exists(f'{file_path}/models/KMeans_occ'):
    model_two = KMeansModel.load(f'{file_path}/models/KMeans_occ')
    
# # Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions_test)
print("Silhouette with squared euclidean distance = " + str(silhouette))

print('Cluster Center')
print(model_two.clusterCenters()[0])
clustercenters = model_two.clusterCenters()
print(len(clustercenters))
print(type(clustercenters))
# print(clustercenters)

import pickle as pkl
with open(f'{path}/models/Clustercenters_occ.pkl', 'wb') as f:
    pkl.dump(clustercenters, f)
    
def extract_feature(vec, index):
    try:
        return float(vec[index])
    except IndexError:
        return None
split_udf = F.udf(extract_feature, FloatType())

# For training data
occ_pred_train = predictions_train.select(F.col('job_id'),\
                                       F.col('date'),\
                                       F.col('prediction'),\
                                       split_udf(predictions_train.scaledFeatures, F.lit(0)).alias('position_code'),\
                                       split_udf(predictions_train.scaledFeatures, F.lit(1)).alias('low_salary'),\
                                       split_udf(predictions_train.scaledFeatures, F.lit(2)).alias('high_salary'),\
                                       split_udf(predictions_train.scaledFeatures, F.lit(3)).alias('degree_code'),\
                                       split_udf(predictions_train.scaledFeatures, F.lit(4)).alias('exp_year'),\
                                       split_udf(predictions_train.scaledFeatures, F.lit(5)).alias('workdays'),\
                                       split_udf(predictions_train.scaledFeatures, F.lit(6)).alias('area_code'),\
                                       split_udf(predictions_train.scaledFeatures, F.lit(7)).alias('name_code'))
occ_pred_train.show(20)

# For testing data
occ_pred_test = predictions_test.select(F.col('job_id'),\
                                       F.col('date'),\
                                       F.col('prediction'),\
                                       split_udf(predictions_test.scaledFeatures, F.lit(0)).alias('position_code'),\
                                       split_udf(predictions_test.scaledFeatures, F.lit(1)).alias('low_salary'),\
                                       split_udf(predictions_test.scaledFeatures, F.lit(2)).alias('high_salary'),\
                                       split_udf(predictions_test.scaledFeatures, F.lit(3)).alias('degree_code'),\
                                       split_udf(predictions_test.scaledFeatures, F.lit(4)).alias('exp_year'),\
                                       split_udf(predictions_test.scaledFeatures, F.lit(5)).alias('workdays'),\
                                       split_udf(predictions_test.scaledFeatures, F.lit(6)).alias('area_code'),\
                                       split_udf(predictions_test.scaledFeatures, F.lit(7)).alias('name_code'))
occ_pred_test.show(20)

# For training data
occ_pred_train.write.mode('overwrite').parquet(f'{file_path}/train/occ')

# For testing data
occ_pred_test.write.mode('overwrite').parquet(f'{file_path}/test/occ')