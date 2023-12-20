import os
# memory = '2g'
# pyspark_submit_args = ' --driver-memory ' + memory + '--executor-memory 8g' \
#                       + ' pyspark-shell'
# os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

import webbrowser
from pyspark.sql import SparkSession

from pyspark.ml.classification import LogisticRegression
import jieba
from pyspark.sql import functions
from pyspark.sql.types import ArrayType,StringType, IntegerType
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StringIndexer

os.environ['JAVA_HOME'] = r"C:\Program Files\Java\jdk1.8.0_101"
os.environ['PYSPARK_PYTHON'] = r"C:\Users\asus\AppData\Local\Programs\Python\Python37\python.exe"
def agg(line):
    # print(line)
    s = ""
    for text in list(line):
        # print(text)
        s += text
    # print(s)
    return s
# conf = SparkConf().setMaster("spark://192.168.10.100:8080").setAppName("test").set("spark.driver.host", "192.168.10.100")
spark = SparkSession.\
    builder\
    .config("spark.executor.memory", "10g")\
    .config("spark.driver.memory", "4g")\
    .config("spark.sql.shuffle.partition", "8")\
    .config("spark.executor.cores", "8")\
    .getOrCreate()
spark.sparkContext.setCheckpointDir("./myCheckPointDir")
# config("spark.sql.shuffle.partitions", "10")\
# spark.sparkContext.setLogLevel("DEBUG")
spark.sparkContext.setLogLevel("ERROR")
webbrowser.open("http://localhost:4040")
# conf = SparkConf().setMaster("spark
# ://192.168.10.100:8080").setAppName("test").set("spark.driver.host", "192.168.10.100")
for i in spark.sparkContext._conf.getAll():
    print(i)
# input_data = spark.read.option("delimiter", "\t").csv(r"./my_data.csv", header=True)
input_data = spark.read.csv(r"./my_data.csv", header=True)

# 删除缺失值
input_data = input_data.dropna()
# input_data.show()
# input_data.selectExpr
# 当使用`中文`时会替换失败
input_data = input_data.withColumnRenamed("岗位名", "position")\
    .withColumnRenamed("经验与学历", "educationBackground") \
    .withColumnRenamed("工作描述", "describe") \
# .withColumnRenamed("经验要求", "experience")\

# input_data.show()
# 替换掉一部分异常字符
split_str_udf = functions.udf(lambda x: x[1:].replace("\"", "").replace("\'", ""), StringType())
input_data = input_data.withColumn("describe", split_str_udf("describe"))
stringIndexer = StringIndexer(inputCol="position", outputCol="id")
# id 这一列没能成功替换变成了 _c0
si_model = stringIndexer.fit(input_data)
td = si_model.transform(input_data)
# 将异常列名替换掉
input_data = td.select(["id", "describe", "educationBackground", "position"])
# input_data = td.withColumnRenamed("_c0", "id")
td.show()

input_data.persist()

# input_data.show()
# temp_data.show()
# group_dataFrame = input_data.groupby("职位").count()

# test_rdd = input_data.rdd.map(lambda x: [str(x[0]), str(x[1])])
# print(test_rdd.collect())
# 对describe信息进行处理，将相同的describe拼接到一起
result = input_data.select("id", "describe").rdd.groupByKey()\
    .mapValues(agg)
#
# print(result.collect())

def seg_sentence(content): # 使用jieba分词，并清洗掉特殊字符
    lit = [i for i in jieba.cut(content, use_paddle=False)]
    special_characters = "“”（；-：、，-。！!@#$%^&*()_+{}[]|\:;'<>?,./\"【】1234567890）"
    lit = [string for string in lit if not any(char in special_characters for char in string)]
    # special_characters = "（；，-：、！!@#$%^&*()_+{}[]|\:;'<>?,./\"【】）"
    # lit = [string for string  lit if not any(char in special_characters for char in string)]
    return list(filter(lambda x: x != "" and x != " " and x != "xa0", lit))

seg_sentence = functions.udf(seg_sentence, ArrayType(StringType()))# 定义一个udf

# 使用udf 函数清洗掉describe中的特殊字符
split_df = result.map(lambda x: (x[0], str(x[1]))).toDF(["id", "describe"]).select("id", seg_sentence("describe").alias("words"))
split_df.cache()
# 复制position为一个新的列 position_name
# split_df.withColumn("position_name", split_df["position"])

# 过滤掉 words小于0的数据
split_df = split_df.filter(functions.size(split_df["words"]) > 0)
print("输出split_df")
split_df = split_df.select(["id", "words"])
split_df.show()

split_df.select(["id"]).groupby("id").count().alias("t1").where("t1.count > 1").show()
# from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
# 因为HashingTF无法获取词索引关系，所以将tf替换为countVectorizer
# hashingTF = HashingTF().setInputCol(value="words").setOutputCol(value="rawFeatures")

split_df.show(truncate=False)
split_df.checkpoint()
split_df.persist()
cv = CountVectorizer(inputCol="words", outputCol="rawFeatures"
                     , vocabSize=400, minDF=10
)
cv = cv.fit(split_df)
featurizeData = cv.transform(split_df)

# 计算词频
featurizeData.show()
# 存储词索引信息
dic = {}
c = 0
for k in cv.vocabulary:
    dic[str(c)] = k
    c += 1
# print("词索引信息:", dic)
def _prc_row(row):
    f = row.features
    indices = f.indices.tolist()
    values = f.values.tolist()
    kvs = {}
    c = 0
    for i in indices:
        kvs[dic.get(str(i))] = values[c]
        c += 1
    return row.id, kvs

# featurizeData = hashingTF.transform(split_df)
idf = IDF().setInputCol(value="rawFeatures").setOutputCol(value="features")
idfModel = idf.fit(featurizeData)


# 使用tf-idf 得到新的特征矩阵
rescaledData = idfModel.transform(featurizeData)
# rescaledData.select(["position", "features"]).show()
# stringIndexer = StringIndexer(inputCol="position", outputCol="id")
# si_model = stringIndexer.fit(rescaledData)
# td = si_model.transform(rescaledData)
spark.sql("drop table if exists user_tags")
rescaledData.rdd.map(_prc_row).toDF(["id", "kvs"]).createTempView("user_tags")

spark.sql("select * from user_tags").show()

spark.sql("select id,k,v from user_tags lateral view explode(kvs)  as k,v").createTempView("temp")

# 计算出每个文章的前20个重要词
spark.sql("select * from temp limit 10").show()

result_df = spark.sql("""
select * from 
( 
select 
id, k, v as score , 
row_number() over(partition by id order by v desc ) as ranking  
from temp)
 where ranking <= 20
""")


# 将模型切分一下，为每个职位名称训练一个模型

# result_df.selectExpr(["k", "max(v)"]).show()
# result_df.show()

# 切分测试集和训练集
# 训练
result_df.show()
# temp1 = result_df.groupby("id").pivot("ranking").count()
# temp1 = temp1.withColumn("flag", functions.lit(True))
temp = result_df.groupby("id").agg(functions.collect_list('k')).alias("feature_words")
"""
将数据聚合为 feature_words[出现频率1,出现频率2,出现频率3,出现频率4  *]
"""
# temp.withColumn("words", "collect_list(k)")
temp.show()
temp.persist()
input_data.show()
# join_df = input_data.select(["id", "describe"]).rdd.mapValues(agg).toDF("id", "words")
# 使用jieba分词，并清洗特殊字符
join_df = input_data.select(["id", "position",seg_sentence("describe").alias("words")])
join_df.persist()
temp = temp.join(join_df, on='id', how='inner')
temp.show()
"""
数据格式
|     id|               collect_list(k)|                         position|                         words|
+-------+------------------------------+---------------------------------+------------------------------+
|13533.0|[客服, 团队, 客户服务, 考核...|     电商客服主管（环境通风透气）|  [职位, 描述, 统筹, 客服, ...|
"""
# id, collect_list(k)[特征词]

def genera_vector(list1, list2):
    """
    判断list2中的词语是否在list1中出现，如果出现，则返回列表对应位置为次数，否则为0
    :param list1:
    :param list2:
    :return:
    """
    result = [0] * 20
    dic = {}
    for i in range(0, len(list2)):
        dic[list2[i]] = i
    key = dic.keys()
    for word in list1:
        if word in key:
            index = dic[word]
            result[index] += 1
    return result

def concatenate_columns(col1, col2):
    """
    生成一个列表
    如果col1中有col2的元素，返回列表对应位置为出现次数，否则为0
    :param col1: 第一列
    :param col2: 第二列
    :return: 返回一个列表
    """
    return genera_vector(col1, col2)
# 重命名一个列
temp = temp.withColumnRenamed("collect_list(k)", "vital_words")
temp.show()
concatenate_columns = functions.udf(concatenate_columns, ArrayType(IntegerType()))
temp = temp.withColumn("features", concatenate_columns("words", "vital_words"))

feature_df = temp.select(["id", "features"])
feature_df.show()
"""
id feature
这里的feature是一个列表，列表中记录了重要词的出现次数
"""
# ids = feature_df.select("id").distinct()
# ids.show()
# ids = ids.rdd.map(lambda x: x.id).collect()
# ids.rdd.map(lambda )
# ids = [item(0) for item in ids]
# print(ids)
# 使用广播变量 广播所有的职业id，然后根据职业id查询出
# broadcastVar = spark.sparkContext.broadcast(ids)
from pyspark.ml.linalg import Vectors, VectorUDT
# number = 1
# for id in broadcastVar.value:
#     if(number == 3):
#         break
#     print(id)
#     number += 1

feature_df.createTempView("feature_df")
# 查询职业出现次数
count_df = spark.sql("""
select id, count(id) as number from feature_df group by id having number > 100 
""")
count_df.show()
count_df.cache()
ids = count_df.select("id").rdd.map(lambda x: x.id).collect()
broadcastVar = spark.sparkContext.broadcast(ids)
# count_df = feature_df.selectExpr(["id", "count(id) as number"])
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
for id in broadcastVar.value:
    # 只有超过10条数据的职业，才能被使用
    # number = feature_df.where("id = {}".format(id)).selectExpr(["count(id) as number"]).rdd.collect()
    # print(number[0]["number"])
    # if(number[0]["number"] < 10):
        # print(id)
        # continue
    # count_df.select("")
    print(id)
    true_df = feature_df.selectExpr(["id", "features"]).where("id = {}".format(id))
    true_df.show(truncate=False)
    true_df = true_df.withColumn("flag", functions.lit(1))
    import random
    # lineNumber = (int)(random.gauss(1, number[0]["number"]))
    # print(lineNumber)
    # lineBroadcast = spark.sparkContext.broadcast(lineNumber)
    lineNumber = count_df.select("number").where("id = {}".format(id)).rdd.map(lambda x: x.number).collect()
    print(lineNumber)
    lineNumber = int((random.random() * lineNumber[0] + 1))
    false_df = feature_df.selectExpr(["id", "features"]).where(" id != {}".format(id)).limit(lineNumber)
    false_df = false_df.withColumn("flag", functions.lit(0))

    """
    根据列名将两个DataFrame拼接在一起
    使用unionByName而不是union方法，因为union方法会根据DataFrame的位置进行拼接
    只要DataFrame对应位置数据类型相同即可完成拼接，
    而unionByName会根据列名和类型名进行拼接
    
    注意: unionByName 在spark2.3时才有
    """
    orgin_df = true_df.unionByName(false_df)
    array_to_vector = functions.udf(lambda arr: Vectors.dense(arr), VectorUDT())
    orgin_df = orgin_df.withColumn("vector_col", array_to_vector("features"))
    orgin_df.show()

    splits = orgin_df.randomSplit([0.7, 0.3])
    print(splits)
    train_df = splits[0]
    test_df = splits[1]
    train_df.show()
    test_df.show()
    log_reg_model = LogisticRegression(featuresCol="vector_col", labelCol="flag", maxIter=10, regParam=0.3, elasticNetParam=0.8)\
        .fit(train_df.select(["vector_col", "flag"]))
    print("objective history")
    for objecttive in log_reg_model.summary.objectiveHistory:
        print(objecttive)
    # 评估模型
    # print param coefficient and interceptVector（系数和截距）
    print("coefficients："+str(log_reg_model.coefficients))
    print("intercept"+str(log_reg_model.intercept))
    predict = log_reg_model.transform(test_df.selectExpr("vector_col"))

    # test_df.show()
    # print(result)
    predict.show()

    print(predict.where("prediction != flag").count())
    print(predict.count())
    # MulticlassClassificationEvaluator(labelCol=)
    # break
    # false_df =

# def gener_vector():
#     """
#     将一个DataFrame的多个特征合并为vector
#     :return:
#     """
# temp.show()
# def split(content):
#     content = str(content)
#     arr = content[1: -1]
#     arr = arr.split(",")
#     return arr
# temp = temp.rdd.mapValues(split)
# print(temp.collect())

class job:
    def __init__(self,
                 position,
                 experience,
                 education_background,
                 describe
                 ):
        self.position = position
        self.experience = experience
        self.education_background = education_background
        self.describe = describe