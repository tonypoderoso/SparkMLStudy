package algorithms

import breeze.linalg.{DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.IndexedSeq
/**
  * Created by tonypark on 2016. 5. 12..
  */
object LinearExampleDatasetDistributed {
  def main(args:Array[String]): Unit = {


    val sc = new SparkContext(new SparkConf()
      //.setMaster("local[*]")
      .setAppName("Dataset Genration")
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

    val recordsPerPartition: Int = num_samples/num_new_partitions


    val noise = 0.1
    val gauss = new Gaussian(0.0,1.0)
    val weights: Array[Double] = gauss.sample(num_features).toArray
    val w = BDV(weights)

    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](),num_new_partitions)
      .mapPartitions { _ => {
        val gauss=new Gaussian(0.0,1.0)
        (1 to recordsPerPartition).map { _ =>
          val x = BDV(gauss.sample(num_features).toArray)
          val l = x.dot(w) + gauss.sample() * noise
          new LabeledPoint(l, Vectors.dense(x.toArray))
        }
      }.toIterator
      }

    val mi = new MutualInformation

    val ffd = mi.featureFromDataset(lds,1)

    ffd.saveAsTextFile("feature_data.txt")

    sc.stop()



  }
}
