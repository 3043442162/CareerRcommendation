package com.csjk.controller.util

import java.nio.charset.StandardCharsets

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
//import org.apache.spark.ml.linalg.Vector
//import org.apache.spark.mllib.linalg+
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD

class SparkUtil() {
  def predict(arr:Array[Double]): Double ={
    val spark = SparkSession.builder()
      .master("local")
      .getOrCreate()


    val model = LogisticRegressionModel.load("E:\\pythonProject\\MachineLearning\\Recommend\\model\\logical_model");


    val array = new Array[Array[Double]](1)
    array(0) = arr;
    val rdd = spark.sparkContext.parallelize(array)
//    spark.sparkContext.setLogLevel("ERROR")
    import  spark.implicits._

//    val value: RDD[linalg.Vector] = rdd.map(line => {
//      Vectors.dense(line)
//    })

    spark.udf.register("addName",(x:Array[Double])=> Vectors.dense(x))
//    val array_to_vector = functions.udf(lambda arr: Vectors.dense(arr), VectorUDT())
//    orgin_df = orgin_df.withColumn("vector_col", array_to_vector("features"))
//    value.toDF("vector_col")
    val dataFrame = rdd.toDF("vector_col")
    val df: DataFrame = dataFrame.selectExpr("addName(vector_col) as vector_col")
    val result = model.transform(df)

    val return_number: Array[Row] = result.select("prediction").rdd.collect()
    result.show()
    return_number(0).getDouble(0);
  }
}
