package com.csjk.controller;

import com.csjk.controller.util.SparkUtil;
import com.huaban.analysis.jieba.JiebaSegmenter;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/position")
public class PositionController {

    @CrossOrigin
    @GetMapping("/result")
    public String get(String code){

        String s  = "'产品', '用户', '需求', '设计', '文档', '规划', '运营', '市场', '分析', '研发', '互联网', '改进', '协调', '持续', '优化', '项目', '对', '能力', '技术', '跟踪'";
        s = s.replaceAll("'", "");
        s = s.replaceAll(" ", "");
        String[] arr = s.split(",");
        Map<String, Integer> map = new HashMap<>();
        for (String str :
                arr) {
            map.put(str, 0);
        }
        JiebaSegmenter jieba = new JiebaSegmenter();
        List<String> list = jieba.sentenceProcess(code);
        for (String seg :
                list) {
            Integer number = map.get(seg);
            // 如果能查出来这个词就让这个词对应的特征向量为1
            if(number != null){
                map.put(seg, 1);
            }
        }
        double[] vector = new double[20];
        SparkUtil uti = new SparkUtil();
        double predict = uti.predict(vector);
//        for (int i = 0; i < 20; i++) {
//            vector[i] = map.get(arr[i]);
//        }
//
//        SparkSession spark = SparkSession.builder()
//                .master("local")
//                .getOrCreate();
//        Vector dense = Vectors.dense(vector);
//
//
//        LogisticRegressionModel model = LogisticRegressionModel.load("E:\\pythonProject\\MachineLearning\\Recommend\\model\\logical_model");
//        SparkContext sc = spark.sparkContext();
//        SparkUtil sparkUtil = new SparkUtil();
//        sparkUtil.predict(vector);
//        sc.parallelize( vector, 1);
//        spark.createDataFrame(vector, Double.class)

////        List<Double> collect = Arrays.stream(vector).boxed().collect(Collectors.toList());
//        List<Double> list1 = new ArrayList<>();
//        for (int i = 0; i < vector.length; i++) {
//            list1.add(i, vector[i]);
//        }
//        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5);
//        JavaRDD<Integer> distData = sc.parallelize(data);
//        JavaRDD<Double> parallelize = sc.parallelize(list1);
//        List<StructField> reportFields = new ArrayList<StructField>();
//        reportFields.add(DataTypes.createStructField("vector_col", DataTypes.DoubleType, true));
//
//        StructType details = new StructType(new StructField[]{
//                new StructField("vector_col", DataTypes.DoubleType, false, Metadata.empty()),
//        });
//        spark.createDataFrame(collect, details);
//        System.out.println(Arrays.toString(vector));
//        System.out.println(code);
        if (predict == 1.0){
            return "您适合成为一名产品经理";
        }else {
            return "您不适合成为一名产品经理";
        }
    }
}
