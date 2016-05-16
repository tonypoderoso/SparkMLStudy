package algorithms

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by tonypark on 2016. 5. 12..
  */
object LinearExampleDatasetTest {

  def main(args:Array[String]): Unit = {


    val sc = new SparkContext(new SparkConf()
      //.setMaster("local[*]")
      .setAppName("Distributed Dataset Genration")
      .set("spark.driver.maxResultSize", "4g")
      .set("spark.akka.frameSize", "1024"))
    //val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("Test"))

    var num_features:Int =100
    var num_samples:Int =100000
    var num_bins:Int = 200
    var num_new_partitions:Int = 5*16

    if (!args.isEmpty){

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt

    }


    val dataset = new LinearExampleDataset(num_samples,num_features-1,0.1)

    val lds: RDD[LabeledPoint] = sc.parallelize(dataset.labeledPoints,num_new_partitions)

    val mi = new MutualInformation

    val ffd = mi.featureFromDataset(lds,1)

    ffd.saveAsTextFile("feature_data.txt" )

    sc.stop()



  }

}
