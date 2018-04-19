# Databricks notebook source

import pandas as pd
from os import listdir
from os.path import join, basename
import struct
import pickle
import json
import os
from scipy import misc
import datetime as dt
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# import matplotlib.pyplot as plt
# %matplotlib inline

# COMMAND ----------

# %pylab inline
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from bigdl.dataset import mnist
from bigdl.transform.vision.image import *
from zoo.pipeline.nnframes.nn_image_reader import *
from zoo.pipeline.nnframes.nn_image_transformer import *
from zoo.pipeline.nnframes.nn_classifier import *
from zoo.common.nncontext import *
import urllib


# COMMAND ----------


def scala_T(input_T):
    """
    Helper function for building Inception layers. Transforms a list of numbers to a dictionary with ascending keys 
    and 0 appended to the front. Ignores dictionary inputs. 
    
    :param input_T: either list or dict
    :return: dictionary with ascending keys and 0 appended to front {0: 0, 1: realdata_1, 2: realdata_2, ...}
    """    
    if type(input_T) is list:
        # insert 0 into first index spot, such that the real data starts from index 1
        temp = [0]
        temp.extend(input_T)
        return dict(enumerate(temp))
    # if dictionary, return it back
    return input_T

# COMMAND ----------

def Inception_Layer_v1(input_size, config, name_prefix=""):
    """
    Builds the inception-v1 submodule, a local network, that is stacked in the entire architecture when building
    the full model.  
    
    :param input_size: dimensions of input coming into the local network
    :param config: ?
    :param name_prefix: string naming the layers of the particular local network
    :return: concat container object with all of the Sequential layers' ouput concatenated depthwise
    """        
    
    '''
    Concat is a container who concatenates the output of it's submodules along the provided dimension: all submodules 
    take the same inputs, and their output is concatenated.
    '''
    concat = Concat(2)
    
    """
    In the above code, we first create a container Sequential. Then add the layers into the container one by one. The 
    order of the layers in the model is same with the insertion order. 
    
    """
    conv1 = Sequential()
    
    #Adding layes to the conv1 model we jus created
    
    #SpatialConvolution is a module that applies a 2D convolution over an input image.
    conv1.add(SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1).set_name(name_prefix + "1x1"))
    conv1.add(ReLU(True).set_name(name_prefix + "relu_1x1"))
    concat.add(conv1)
    
    conv3 = Sequential()
    conv3.add(SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1).set_name(name_prefix + "3x3_reduce"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1).set_name(name_prefix + "3x3"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3"))
    concat.add(conv3)
    
    
    conv5 = Sequential()
    conv5.add(SpatialConvolution(input_size,config[3][1], 1, 1, 1, 1).set_name(name_prefix + "5x5_reduce"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2).set_name(name_prefix + "5x5"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5"))
    concat.add(conv5)
    
    
    pool = Sequential()
    pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1, to_ceil=True).set_name(name_prefix + "pool"))
    pool.add(SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1).set_name(name_prefix + "pool_proj"))
    pool.add(ReLU(True).set_name(name_prefix + "relu_pool_proj"))
    concat.add(pool).set_name(name_prefix + "output")
    return concat

# COMMAND ----------

def Inception_v1(class_num):
    model = Sequential()
    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False).set_name("conv1/7x7_s2"))
    model.add(ReLU(True).set_name("conv1/relu_7x7"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool1/3x3_s2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("pool1/norm1"))
    model.add(SpatialConvolution(64, 64, 1, 1, 1, 1).set_name("conv2/3x3_reduce"))
    model.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).set_name("conv2/3x3"))
    model.add(ReLU(True).set_name("conv2/relu_3x3"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("conv2/norm2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
    model.add(Inception_Layer_v1(192, scala_T([scala_T([64]), scala_T(
         [96, 128]), scala_T([16, 32]), scala_T([32])]), "inception_3a/"))
    model.add(Inception_Layer_v1(256, scala_T([scala_T([128]), scala_T(
         [128, 192]), scala_T([32, 96]), scala_T([64])]), "inception_3b/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(Inception_Layer_v1(480, scala_T([scala_T([192]), scala_T(
         [96, 208]), scala_T([16, 48]), scala_T([64])]), "inception_4a/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([160]), scala_T(
         [112, 224]), scala_T([24, 64]), scala_T([64])]), "inception_4b/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([128]), scala_T(
         [128, 256]), scala_T([24, 64]), scala_T([64])]), "inception_4c/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([112]), scala_T(
         [144, 288]), scala_T([32, 64]), scala_T([64])]), "inception_4d/"))
    model.add(Inception_Layer_v1(528, scala_T([scala_T([256]), scala_T(
         [160, 320]), scala_T([32, 128]), scala_T([128])]), "inception_4e/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(Inception_Layer_v1(832, scala_T([scala_T([256]), scala_T(
         [160, 320]), scala_T([32, 128]), scala_T([128])]), "inception_5a/"))
    model.add(Inception_Layer_v1(832, scala_T([scala_T([384]), scala_T(
         [192, 384]), scala_T([48, 128]), scala_T([128])]), "inception_5b/"))
    model.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
    model.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
    model.add(View([1024], num_input_dims=3))
    model.add(Linear(1024, class_num).set_name("loss3/classifier"))
    model.add(LogSoftMax().set_name("loss3/loss3"))
    model.reset()
    return model

# COMMAND ----------

# MAGIC %md ## Download the images from Amazon s3
# MAGIC 
# MAGIC Make sure you have AWS command line interface to recursively download all images in s3 folder. You can set up aws cli from this link: http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html

# COMMAND ----------

import urllib
from os import path
MODEL_ROOT = "/mnt/nobigdl/few-inceptionv1"
# dbutils.fs.mkdirs(MODEL_ROOT)
#local_folder = DATA_ROOT + '/vegnonveg-samples'
checkpoint_path = path.join(MODEL_ROOT, "checkpoints")

# if not path.isdir(local_folder):
#   os.system('aws s3 cp --recursive s3://vegnonveg/vegnonveg-fewsamples %s' % local_folder)

# COMMAND ----------

# MAGIC %md ## Save images and load to Spark as BigDL ImageFrame
# MAGIC 
# MAGIC save data to parquet files and load to spark. Add label to each image.

# COMMAND ----------

DATA_ROOT = "/data/worldbank/"
sample_path = DATA_ROOT + 'samples/'
# sample_path = DATA_ROOT + 'imagenet_samples/'
# sample_path = '/mnt/nobigdl/vegnonveg-samples100/'
label_path = DATA_ROOT + 'vegnonveg-samples_labels.csv'
parquet_path = DATA_ROOT + 'sample_parquet/'
# dbutils.fs.rm(parquet_path, True)



# COMMAND ----------
sparkConf = create_spark_conf().setMaster("local[2]").setAppName("test_validation")
sc = get_spark_context(sparkConf)
sqlContext = SQLContext(sc)
#intializa bigdl
init_engine()
redire_spark_logs()

# This only runs at the first time to generate parquet files
image_frame = NNImageReader.readImages(sample_path, sc, minParitions=32)
# save dataframe to parquet files
# image_frame.write.parquet(parquet_path)
# ImageFrame.write_parquet(sample_path, parquet_path, sc, partition_num=32)

# COMMAND ----------

# load parquet file into spark cluster
import time
start = time.time()
image_raw_DF = sqlContext.read.parquet(parquet_path)
end = time.time()
print("Load data time is: " + str(end-start) + " seconds")

# COMMAND ----------

# create dict from item_name to label
labels_csv = pd.read_csv(label_path)
unique_labels = labels_csv['item_name'].unique().tolist()
label_dict = dict(zip(unique_labels, range(1,len(unique_labels)+1)))
class_num = len(label_dict)

# COMMAND ----------

# create label dataframe
label_raw_DF = sqlContext.read.format("com.databricks.spark.csv")\
    .option("header", "true")\
    .option("mode", "DROPMALFORMED")\
    .load(label_path)
get_label = udf(lambda item_name: float(label_dict[item_name]), FloatType())
change_name = udf(lambda uid: uid+".jpg", StringType())
labelDF = label_raw_DF.withColumn("label", get_label("item_name")).withColumn("image_name", change_name("obs_uid"))
labelDF.show(truncate=False)

# COMMAND ----------

get_name = udf(lambda row: row[0].split("/")[-1], StringType())
imageDF = image_raw_DF.withColumn("image_name", get_name("image"))
imageDF.show(truncate=False)
dataDF = imageDF.join(labelDF, "image_name", "inner").select("image", "image_name", "label")
dataDF.show(truncate=False)

# COMMAND ----------

# MAGIC %md ## Do Train/Test Split and preprocessing
# MAGIC Split Train/Test split with some ratio and preprocess images.

# COMMAND ----------

data = dataDF.randomSplit([0.8, 0.2], seed=10)
train_image = data[0]
val_image = data[1]
type(train_image)


# COMMAND ----------

IMAGE_SIZE = 224

train_transformer = NNImageTransformer(
    Pipeline([Resize(256, 256), RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
              ChannelNormalize(123.0, 117.0, 104.0, 1.0, 1.0, 1.0),
              MatToTensor()])
).setInputCol("image").setOutputCol("features")

train_data = train_transformer.transform(train_image)


# COMMAND ----------

train_size = train_image.count()

# COMMAND ----------

print(train_size)


# COMMAND ----------

val_transformer = NNImageTransformer(
    Pipeline([Resize(256,256),
              CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
              ChannelNormalize(123.0, 117.0, 104.0, 1.0, 1.0, 1.0),
              MatToTensor(to_rgb=True)]
            )
).setInputCol("image").setOutputCol("features")

# COMMAND ----------

test_data = val_transformer.transform(val_image)

# COMMAND ----------

# MAGIC %md ## Define Model

# COMMAND ----------

# Network Parameters
n_classes = len(label_dict)# item_name categories
model = Inception_v1(n_classes)

# COMMAND ----------

# Parameters
learning_rate = 0.2
# parameters for 
batch_size = 2 #depends on dataset
no_epochs = 1 #stop when validation accuracy doesn't improve anymore

# COMMAND ----------

criterion = ClassNLLCriterion()
classifier = NNClassifier(model, criterion, [3,IMAGE_SIZE,IMAGE_SIZE])\
    .setBatchSize(batch_size)\
    .setMaxEpoch(no_epochs)\
    .setLearningRate(learning_rate)
start = time.time()
trained_model = classifier.fit(train_data)
end = time.time()
print("Optimization Done.")
print("Training time is: %s seconds" % str(end-start))
# + dt.datetime.now().strftime("%Y%m%d-%H%M%S")

# COMMAND ----------

throughput = train_size * no_epochs / (end - start)
print("Average throughput is: %s" % str(throughput))

# COMMAND ----------

#predict
predict_model = trained_model.setBatchSize(batch_size)
predictionDF = predict_model.transform(test_data)
predictionDF.show()

# COMMAND ----------

num_preds = 1
preds = predictionDF.select("label", "prediction").take(num_preds)
for idx in range(num_preds):
#    true_label = str(map_to_label(map_groundtruth_label(truth[idx].label)))
    true_label = preds[idx][0]
    pred_label = preds[idx][1]
    print(idx + 1, ')', 'Ground Truth label: ', true_label)
    print(idx + 1, ')', 'Predicted label: ', pred_label)
    print("correct" if true_label == pred_label else "wrong")

# COMMAND ----------

'''
Measure Test Accuracy w/Test Set
'''
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictionDF)
# expected error should be less than 10%
print("Accuracy = %g " % accuracy)
