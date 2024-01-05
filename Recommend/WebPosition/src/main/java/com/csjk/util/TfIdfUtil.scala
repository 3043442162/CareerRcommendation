package com.csjk.util

import java.nio.file.Paths
import java.util

import scala.collection.JavaConverters._
import com.huaban.analysis.jieba.{JiebaSegmenter, WordDictionary}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

class TfIdfUtil {
//  def main(args: Array[String]): Unit = {
  def handleTokens(tokens:String):Array[String]={
//    val tokens = "能配合产品主管进行市场调研，了解客户需求及市场情况，并根据市场及用户反馈，分析需求，构思并策划产品方案，如进行产品定义、功能设计；\n能充分协调沟通内外资源，完成产品实现，如撰写产品需求文档，设计产品原型；\n曾负责产品生命周期管理，进行项目管理及协调，跟踪日常进度，把控关键节点以确保开发按计划要求开发出产品，不断优化产品的用户体验；\n对行业、竞品进行过调研分析，提出产品调研报告，制定产品优化方向；\n能协同业务方共同探索业务支撑需求。\n能跟踪产品开发进度，接收各部门意见，并进行反馈沟通；\n能跟进运营需求，在产品负责人的带领下设计产品迭代需求。\n有3年以上产品策划、分析、设计、实施工作经验，计算机相关专业，本科学历\n具备园林博物馆系统项目经验，有过一定的用户运营经验；有从事大数据分析经验；\n掌握需求分析方法，熟悉需求管理和研发过程管理；\n熟练使用Axure、Xmind、Visio等；\n具有较强的沟通能力，逻辑思维能力、分析能力和文档编写能力；\n较强的责任心及团队合作精神，能够承担工作压力；\n对互联网新兴概念有一定认知和浓厚的兴趣，对行业的业务模式、未来趋势有深入的理解和思考，具备良好的业务敏感度和视野，能后敏锐地捕捉产品机会和数据价值；"
    val spark = SparkSession.builder().master("local").getOrCreate()
    val strs = new Array[Array[String]](1)
    strs(0) = genertor_tokens(tokens)
    val rdd = spark.sparkContext.parallelize(strs)

    import spark.implicits._

    val chars = rdd.collect()

    val frame = rdd.toDF("token")
    val jobs = Array("移动开发","硬件开发","运维","企业软件","测试","前端开发","后端开发","项目管理","dba")
    val models =new  Array[PipelineModel](10)
    for (i <- jobs.indices){
      val str = jobs(i)
      val pipeline: PipelineModel = PipelineModel.load(s"./model/'$str'")
      models(i) = pipeline
    }
//    chars.foreach(println)

    val result = new ArrayBuffer[String]()
    for (i <- jobs.indices){
      val dataFrame = models(i).transform(frame)
      dataFrame.show()
      if (dataFrame.where("prediction = 1").count() == 1){
//        println(i)
//        println("您适合这个职业"+jobs(i))
        result.append(jobs(i))
      }
    }
    result.toArray
  }

  def genertor_tokens(tokens:String): Array[String]={
    WordDictionary.getInstance().init(Paths.get("E:\\pythonProject\\MachineLearning\\Recommend\\WebPosition\\src\\main\\resources\\conf"))
    val segmenter = new JiebaSegmenter()
    val strings: util.List[String] = segmenter.sentenceProcess(tokens)
    strings.asScala.toArray
  }
}
