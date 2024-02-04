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

person = spark.read.parquet(f'{file_path}/0data/202code2_query_{city_short}/person')
person_train = person.filter(F.col('date').isin(dates_train)).cache()
person_test = person.filter(F.col('date').isin(dates_test)).cache()
person.show(1)
person.printSchema()
print('person cnt:', person.count())
print('person_train cnt:', person_train.count())
print('person_test cnt:', person_test.count())

print('job 数值化1')

# For training data
person_train = person_train.select(\
                 F.col('geek_id').cast(LongType()),\
                 F.col('age').cast(IntegerType()), \
                 F.col('gender').cast(IntegerType()), \
                 F.col('degree_code').cast(IntegerType()),\
                 F.col('work_years').cast(IntegerType()),\
                 F.col('fresh_graduate').cast(IntegerType()),\
                 F.col('apply_status').cast(IntegerType()),\
                 F.col('expect1_update_time').cast(LongType()),\
                 F.col('expect2_update_time').cast(LongType()),\
                 F.col('expect3_update_time').cast(LongType()),\
                 F.col('edu1_education_id').alias('edu1_id').cast(IntegerType()),\
                 F.col('edu1_degree_code').cast(IntegerType()),\
                 F.col('edu1_standard_major_name').alias('edu1_major'),\
                 F.col('edu1_start_date').cast(LongType()),\
                 F.col('edu1_end_date').cast(LongType()),\
                 F.col('edu2_education_id').alias('edu2_id').cast(IntegerType()),\
                 F.col('edu2_degree_code').cast(IntegerType()),\
                 F.col('edu2_standard_major_name').alias('edu2_major'),\
                 F.col('edu2_start_date').cast(LongType()),\
                 F.col('edu2_end_date').cast(LongType()),\
                 F.col('edu3_education_id').alias('edu3_id').cast(IntegerType()),\
                 F.col('edu3_degree_code').cast(IntegerType()),\
                 F.col('edu3_standard_major_name').alias('edu3_major'),\
                 F.col('edu3_start_date').cast(LongType()),\
                 F.col('edu3_end_date').cast(LongType()),\
                 F.col('work1_position_lv3_code').cast(IntegerType()),\
                 F.col('work1_industry_lv1_code').cast(IntegerType()),\
                 F.col('work1_skills'),\
                 F.col('work1_start_date').cast(LongType()),\
                 F.col('work1_end_date').cast(LongType()),\
                 F.col('work2_position_lv3_code').cast(IntegerType()),\
                 F.col('work2_industry_lv1_code').cast(IntegerType()),\
                 F.col('work2_skills'),\
                 F.col('work2_start_date').cast(LongType()),\
                 F.col('work2_end_date').cast(LongType()),\
                 F.col('work3_position_lv3_code').cast(IntegerType()),\
                 F.col('work3_industry_lv1_code').cast(IntegerType()),\
                 F.col('work3_skills'),\
                 F.col('work3_start_date').cast(LongType()),\
                 F.col('work3_end_date').cast(LongType()),\
                 F.to_date(F.col('date')).alias('date'),\
                )
print('\tNULL field process...')
person_train = person_train.fillna({'work1_industry_lv1_code': 80000,\
                       'work2_industry_lv1_code': 80000,\
                       'work3_industry_lv1_code': 80000,\
                       }).fillna(0).fillna("无")
person_train.show(1)



# For testing data
person_test = person_test.select(\
                 F.col('geek_id').cast(LongType()),\
                 F.col('age').cast(IntegerType()), \
                 F.col('gender').cast(IntegerType()), \
                 F.col('degree_code').cast(IntegerType()),\
                 F.col('work_years').cast(IntegerType()),\
                 F.col('fresh_graduate').cast(IntegerType()),\
                 F.col('apply_status').cast(IntegerType()),\
                 F.col('expect1_update_time').cast(LongType()),\
                 F.col('expect2_update_time').cast(LongType()),\
                 F.col('expect3_update_time').cast(LongType()),\
                 F.col('edu1_education_id').alias('edu1_id').cast(IntegerType()),\
                 F.col('edu1_degree_code').cast(IntegerType()),\
                 F.col('edu1_standard_major_name').alias('edu1_major'),\
                 F.col('edu1_start_date').cast(LongType()),\
                 F.col('edu1_end_date').cast(LongType()),\
                 F.col('edu2_education_id').alias('edu2_id').cast(IntegerType()),\
                 F.col('edu2_degree_code').cast(IntegerType()),\
                 F.col('edu2_standard_major_name').alias('edu2_major'),\
                 F.col('edu2_start_date').cast(LongType()),\
                 F.col('edu2_end_date').cast(LongType()),\
                 F.col('edu3_education_id').alias('edu3_id').cast(IntegerType()),\
                 F.col('edu3_degree_code').cast(IntegerType()),\
                 F.col('edu3_standard_major_name').alias('edu3_major'),\
                 F.col('edu3_start_date').cast(LongType()),\
                 F.col('edu3_end_date').cast(LongType()),\
                 F.col('work1_position_lv3_code').cast(IntegerType()),\
                 F.col('work1_industry_lv1_code').cast(IntegerType()),\
                 F.col('work1_skills'),\
                 F.col('work1_start_date').cast(LongType()),\
                 F.col('work1_end_date').cast(LongType()),\
                 F.col('work2_position_lv3_code').cast(IntegerType()),\
                 F.col('work2_industry_lv1_code').cast(IntegerType()),\
                 F.col('work2_skills'),\
                 F.col('work2_start_date').cast(LongType()),\
                 F.col('work2_end_date').cast(LongType()),\
                 F.col('work3_position_lv3_code').cast(IntegerType()),\
                 F.col('work3_industry_lv1_code').cast(IntegerType()),\
                 F.col('work3_skills'),\
                 F.col('work3_start_date').cast(LongType()),\
                 F.col('work3_end_date').cast(LongType()),\
                 F.to_date(F.col('date')).alias('date'),\
                )
print('\tNULL field process...')
person_test = person_test.fillna({'work1_industry_lv1_code': 80000,\
                       'work2_industry_lv1_code': 80000,\
                       'work3_industry_lv1_code': 80000,\
                       }).fillna(0).fillna("无")
person_test.show(1)

print('job 数值化2')
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
change_udf = F.udf(degree_change_to_level, IntegerType())

# For training data
person_train = person_train.withColumn("degree_code", change_udf(person_train.degree_code))\
            .withColumn("edu1_degree_code", change_udf(person_train.edu1_degree_code))\
            .withColumn("edu2_degree_code", change_udf(person_train.edu2_degree_code))\
            .withColumn("edu3_degree_code", change_udf(person_train.edu3_degree_code))\
            .withColumn("work1_position_lv3_code", person_train.work1_position_lv3_code-100000)\
            .withColumn("work2_position_lv3_code", person_train.work2_position_lv3_code-100000)\
            .withColumn("work3_position_lv3_code", person_train.work3_position_lv3_code-100000)\
            .withColumn("work2_industry_lv1_code", person_train.work2_industry_lv1_code-100000)
person_train.show(1)
person_train.write.mode('overwrite').parquet(f'{file_path}/train/person_tmp')

# For testing data
person_test = person_test.withColumn("degree_code", change_udf(person_test.degree_code))\
            .withColumn("edu1_degree_code", change_udf(person_test.edu1_degree_code))\
            .withColumn("edu2_degree_code", change_udf(person_test.edu2_degree_code))\
            .withColumn("edu3_degree_code", change_udf(person_test.edu3_degree_code))\
            .withColumn("work1_position_lv3_code", person_test.work1_position_lv3_code-100000)\
            .withColumn("work2_position_lv3_code", person_test.work2_position_lv3_code-100000)\
            .withColumn("work3_position_lv3_code", person_test.work3_position_lv3_code-100000)\
            .withColumn("work2_industry_lv1_code", person_test.work2_industry_lv1_code-100000)
person_test.show(1)
person_test.write.mode('overwrite').parquet(f'{file_path}/test/person_tmp')


def concat_arrays(array1, array2, array3, array4):
    return array1 + array2 + array3 + array4


concat_udf = F.udf(concat_arrays, ArrayType(StringType()))

# For training data
person_train = person_train.withColumn("text_combined", concat_udf(F.array("edu1_major", "edu2_major", "edu3_major"), \
                                          F.split("work1_skills", "、"), F.split("work2_skills", "、"), F.split("work3_skills", "、")))\
        .drop('edu1_major', 'edu2_major', 'edu3_major', 'work1_skills', 'work2_skills', 'work3_skills')
person_train.show(1)

# For testing data
person_test = person_test.withColumn("text_combined", concat_udf(F.array("edu1_major", "edu2_major", "edu3_major"), \
                                          F.split("work1_skills", "、"), F.split("work2_skills", "、"), F.split("work3_skills", "、")))\
        .drop('edu1_major', 'edu2_major', 'edu3_major', 'work1_skills', 'work2_skills', 'work3_skills')
person_test.show(1)

from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.clustering import LDA, LocalLDAModel

# For training data
data_selected_train = person_train.select("geek_id", "date", "text_combined").cache()
data_selected_train.show(1)

# For testing data
data_selected_test = person_test.select("geek_id", "date", "text_combined").cache()
data_selected_test.show(1)


print('\tCalculating Word Freq')
vectorizer = CountVectorizer(inputCol="text_combined",
                             outputCol="features")

# For training data
model = vectorizer.fit(data_selected_train)
model.write().overwrite().save(f'{file_path}/models/CountVectorizer_person')

featuresData_train = model.transform(data_selected_train)
# featuresData.show(3)

# For testing data
if os.path.exists(f'{path}/models/CountVectorizer_person'):
    model = CountVectorizerModel.load(f'{file_path}/models/CountVectorizer_person')
    
featuresData_test = model.transform(data_selected_test)
# featuresData.show(3)

print('\tLda traininig...')
# For training data
lda = LDA(k=numTopics, maxIter=100)
ldaModel = lda.fit(featuresData_train)
ldaModel.write().overwrite().save(f'{file_path}/models/LDA_person')

# For testing data
if os.path.exists(f'{path}/models/LDA_person'):
    ldaModel = LocalLDAModel.load(f'{file_path}/models/LDA_person')

print('\tGet Topic')
# For training data
transformedData_train = ldaModel.transform(featuresData_train)
# transformedData.show()

# For testing data
transformedData_test = ldaModel.transform(featuresData_test)

print('\tUnderstand Topic')
def getTopTopics(args):
    geekId, date, topicDistribution = args
    topTopics = topicDistribution.toArray().argsort()[-1:][::-1][0]
    return (geekId, date, topTopics.tolist())

print('\tGet topic per person')
# For training data
documentTopics_train = transformedData_train.rdd.map(lambda row: (row["geek_id"], row["date"], row["topicDistribution"])) \
    .map(getTopTopics) \
    .toDF(["geek_id1", "date1", "skills"])
documentTopics_train.show(1, truncate=False)

# For testing data
documentTopics_test = transformedData_test.rdd.map(lambda row: (row["geek_id"], row["date"], row["topicDistribution"])) \
    .map(getTopTopics) \
    .toDF(["geek_id1", "date1", "skills"])
documentTopics_test.show(1, truncate=False)

print('\tjoin operator')
# For training data
person_train = person_train.join(documentTopics_train, (person_train.geek_id == documentTopics_train.geek_id1) & (person_train.date == documentTopics_train.date1), \
                     how="left").drop("text_combined", "geek_id1", "date1").cache()
person_train.show(1)
print('person cnt:', person_train.count())


# For testing data
person_test = person_test.join(documentTopics_test, (person_test.geek_id == documentTopics_test.geek_id1) & (person_test.date == documentTopics_test.date1), \
                     how="left").drop("text_combined", "geek_id1", "date1").cache()
person_test.show(1)
print('person cnt:', person_train.count())

from pyspark.ml.feature import VectorAssembler


vectorAssembler = VectorAssembler(inputCols=['age','gender','degree_code','work_years','fresh_graduate','apply_status',\
                                             'expect1_update_time','expect2_update_time','expect3_update_time','skills',\
                                             'edu1_id','edu1_degree_code','edu1_start_date','edu1_end_date',\
                                             'edu2_id','edu2_degree_code','edu2_start_date','edu2_end_date',\
                                             'edu3_id','edu3_degree_code','edu3_start_date','edu3_end_date',\
                                             'work1_position_lv3_code','work1_industry_lv1_code','work1_start_date','work1_end_date',\
                                             'work2_position_lv3_code','work2_industry_lv1_code','work2_start_date','work2_end_date',\
                                             'work3_position_lv3_code','work3_industry_lv1_code','work3_start_date','work3_end_date'],\
                                  outputCol="features")
# For training data
tansdata_train = vectorAssembler.transform(person_train).select('geek_id', 'features', 'date')
tansdata_train.show(1)

# For testing data
tansdata_test = vectorAssembler.transform(person_test).select('geek_id', 'features', 'date')
tansdata_test.show(1)

print('数据归一化')
from pyspark.ml.feature import StandardScaler, StandardScalerModel

# For training data
standardScaler = StandardScaler(inputCol="features",outputCol="scaledFeatures")
    
model_one = standardScaler.fit(tansdata_train)
model_one.write().overwrite().save(f'{file_path}/models/StandardScalerModel_person')

stddata_train = model_one.transform(tansdata_train)
stddata_train.show(1)

# For testing data
if os.path.exists(f'{path}/models/StandardScalerModel_person'):
    model_one = StandardScalerModel.load(f'{file_path}/models/StandardScalerModel_person')

stddata_test = model_one.transform(tansdata_test)
stddata_test.show(1)

from pyspark.ml.clustering import KMeans, KMeansModel


# # traindata,testdata = stddata.randomSplit([0.8,0.2])
# traindata = stddata.filter(F.col('date').isin(dates_train))
# testdata = stddata.filter(F.col('date').isin(dates_test))
# print(traindata.count(),testdata.count())

print('KMeans')
# For training data
kmeans = KMeans(featuresCol="scaledFeatures",k=K_person,seed=1)

model_two = kmeans.fit(stddata_train) # traindata<->stddata
model_two.write().overwrite().save(f'{file_path}/models/KMeans_person')

predictions_train = model_two.transform(stddata_train) # testdata<->stddata

# For testing data
if os.path.exists(f'{path}/models/KMeans_person'):
    model_two = KMeansModel.load(f'{file_path}/models/KMeans_person')
    
predictions_test = model_two.transform(stddata_test) # testdata<->stddata

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator

if os.path.exists(f'{path}/models/KMeans_person'):
    model_two = KMeansModel.load(f'{file_path}/models/KMeans_person')

# # Evaluate clustering by computing Silhouette score
# evaluator = ClusteringEvaluator()

# silhouette = evaluator.evaluate(predictions_test)
# print("Silhouette with squared euclidean distance = " + str(silhouette))

print('Cluster Center')
# print(model_two.clusterCenters())
clustercenters = model_two.clusterCenters()
print(len(clustercenters))
print(type(clustercenters))

import pickle as pkl
with open(f'{path}/models/Clustercenters_person.pkl', 'wb') as f:
    pkl.dump(clustercenters, f)
    
def extract_feature(vec, index):
    try:
        return float(vec[index])
    except IndexError:
        return None
split_udf = F.udf(extract_feature, FloatType())

# For training data
person_pred_train = predictions_train.select(F.col('geek_id'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(0)).alias('age'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(1)).alias('gender'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(2)).alias('degree_code'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(3)).alias('work_years'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(4)).alias('fresh_graduate'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(5)).alias('apply_status'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(6)).alias('expect1_update_time'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(7)).alias('expect2_update_time'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(8)).alias('expect3_update_time'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(9)).alias('skills'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(10)).alias('edu1_id'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(11)).alias('edu1_degree_code'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(12)).alias('edu1_start_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(13)).alias('edu1_end_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(14)).alias('edu2_id'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(15)).alias('edu2_degree_code'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(16)).alias('edu2_start_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(17)).alias('edu2_end_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(18)).alias('edu3_id'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(19)).alias('edu3_degree_code'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(20)).alias('edu3_start_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(21)).alias('edu3_end_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(22)).alias('work1_position_lv3_code'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(23)).alias('work1_industry_lv1_code'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(24)).alias('work1_start_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(25)).alias('work1_end_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(26)).alias('work2_position_lv3_code'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(27)).alias('work2_industry_lv1_code'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(28)).alias('work2_start_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(29)).alias('work2_end_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(30)).alias('work3_position_lv3_code'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(31)).alias('work3_industry_lv1_code'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(32)).alias('work3_start_date'),\
                                          split_udf(predictions_train.scaledFeatures, F.lit(33)).alias('work3_end_date'),\
                                          F.col('date'),
                                          F.col('prediction'))
person_pred_train.show(1)

# For testing data
person_pred_test = predictions_test.select(F.col('geek_id'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(0)).alias('age'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(1)).alias('gender'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(2)).alias('degree_code'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(3)).alias('work_years'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(4)).alias('fresh_graduate'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(5)).alias('apply_status'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(6)).alias('expect1_update_time'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(7)).alias('expect2_update_time'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(8)).alias('expect3_update_time'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(9)).alias('skills'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(10)).alias('edu1_id'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(11)).alias('edu1_degree_code'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(12)).alias('edu1_start_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(13)).alias('edu1_end_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(14)).alias('edu2_id'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(15)).alias('edu2_degree_code'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(16)).alias('edu2_start_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(17)).alias('edu2_end_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(18)).alias('edu3_id'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(19)).alias('edu3_degree_code'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(20)).alias('edu3_start_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(21)).alias('edu3_end_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(22)).alias('work1_position_lv3_code'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(23)).alias('work1_industry_lv1_code'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(24)).alias('work1_start_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(25)).alias('work1_end_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(26)).alias('work2_position_lv3_code'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(27)).alias('work2_industry_lv1_code'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(28)).alias('work2_start_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(29)).alias('work2_end_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(30)).alias('work3_position_lv3_code'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(31)).alias('work3_industry_lv1_code'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(32)).alias('work3_start_date'),\
                                          split_udf(predictions_test.scaledFeatures, F.lit(33)).alias('work3_end_date'),\
                                          F.col('date'),
                                          F.col('prediction'))
person_pred_test.show(1)

person_pred_train.write.partitionBy("prediction").mode('overwrite').parquet(f'{file_path}/train/person')
person_pred_test.write.partitionBy("prediction").mode('overwrite').parquet(f'{file_path}/test/person')

print('person train cnt:', person_pred_train.count())
print('person test cnt:', person_pred_test.count())