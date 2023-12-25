import com.csjk.controller.util.SparkUtil;

public class Test {
    @org.junit.jupiter.api.Test
    void test(){
        double[] vector = new double[20];
        for (int i = 0; i < 20; i++) {
            vector[i] = 1;
        }
        SparkUtil sparkUtil = new SparkUtil();
        sparkUtil.predict(vector);
    }
}
