package algorithms

import preprocessing._
import breeze.stats.distributions.Gaussian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by tonypark on 2016. 6. 7..
  */
object CovarianceTestMain {

  def main(args: Array[String]): Unit = {


    //Logger.getLogger("org").setLevel(Level.OFF)

    val sc = new SparkContext(new SparkConf()
      //.setMaster("local[*]").setAppName("PCAExampleTest")
      .set("spark.driver.maxResultSize", "90g")
      .set("spark.akka.timeout","200000")
      .set("spark.worker.timeout","500000")
      .set("spark.storage.blockManagerSlaveTimeoutMs","5000000")
      .set("spark.akka.frameSize", "1024")
      .set("spark.network.timeout","600")
      .set("spark.rpc.askTimeout","600")
      .set("spark.rpc.lookupTimeout","600"))

    //val sc = new SparkContext(new SparkConf().setMaster("local[*]"))

    var num_features:Int =1000
    var num_samples:Int =100000
    var num_bins:Int = 200
    var num_new_partitions:Int = 50

    if (!args.isEmpty){

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt

    }


    val recordsPerPartition: Int = num_samples / num_new_partitions

    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitions { _ => {
        val gauss = new Gaussian(0.0, 1.0)
        val r = scala.util.Random
        (1 to recordsPerPartition).map { _ =>
          new LabeledPoint(0, Vectors.dense(gauss.sample(num_features).toArray))
        }
      }.toIterator
      }

    lds.cache()
    //val start = System.currentTimeMillis
    val covrdd = lds.computeCovarianceRDD(num_samples, num_features)

    val res=covrdd.toRowMatrix()



    res.rows.take(100).foreach{x => x.toArray.foreach(println)}



    //println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start) / 1000.0))

    sc.stop()

  }

}
