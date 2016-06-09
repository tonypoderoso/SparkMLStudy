package algorithms

import algorithms.common.PrintWrapper
import breeze.linalg.{DenseMatrix => BDM, cov => brzCov}
import preprocessing._
import breeze.stats.distributions.Gaussian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
/**
  * Created by tonypark on 2016. 6. 7..
  */
object CovarianceTestMain {

def toBreeze(matin : CoordinateMatrix): BDM[Double] = {
    val m = matin.numRows().toInt
    val n = matin.numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    matin.entries.collect().foreach { case MatrixEntry(i, j, value) =>
      mat(i.toInt, j.toInt) = value
    }
    mat
  }

  def main(args: Array[String]): Unit = {


    //Logger.getLogger("org").setLevel(Level.OFF)

    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]").setAppName("PCAExampleTest")
      .set("spark.driver.maxResultSize", "90g")
      .set("spark.akka.timeout","200000")
      .set("spark.worker.timeout","500000")
      .set("spark.storage.blockManagerSlaveTimeoutMs","5000000")
      .set("spark.akka.frameSize", "1024")
      .set("spark.network.timeout","600")
      .set("spark.rpc.askTimeout","600")
      .set("spark.rpc.lookupTimeout","600"))

    //val sc = new SparkContext(new SparkConf().setMaster("local[*]"))

/*
    val csv: RDD[String] = sc.textFile("/Users/tonypark/ideaProjects/SparkMLStudy/src/test/resources/pcatest.csv")

    val dataRDD: RDD[Vector] = csv.map(line => Vectors.dense( line.split(",").map(elem => elem.trim.toDouble)))
    val mat = new RowMatrix(dataRDD)
    val num_samples=mat.numRows().toInt
    val num_features=mat.numCols().toInt
    val num_new_partitions=dataRDD.getNumPartitions

    val lds: RDD[LabeledPoint] = dataRDD.map{ x=>
      new LabeledPoint(0, x)
    }

*/


    var num_features:Int =10000
    var num_samples:Int =10000
    var num_bins:Int = 20
    var num_new_partitions:Int = 5*16

    if (!args.isEmpty){

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt

    }

    val recordsPerPartition: Int = num_samples / num_new_partitions
    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitionsWithIndex { (idx,iter) => {
        val gauss = new Gaussian(0.0, 1.0)
        (1 to recordsPerPartition).map { i =>
          new LabeledPoint(idx, Vectors.dense(gauss.sample(num_features).toArray))
        }
      }.toIterator
      }


    lds.cache()

    val start = System.currentTimeMillis
    val aa: (Int, Vector) =lds.computeCovarianceRDD(num_samples, num_features).first()

    //val mat=new RowMatrix(lds.map(x => Vectors.dense(x.features.toArray)))

    //val rowcov: Matrix = mat.computeCovariance()
    //val res: Iterator[String] =Iterator.tabulate(10){ i=>i.toString + ","}

    println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start)/1000.0))
    //println(rowcov)

    //println(distcov)
    //PrintWrapper.RowMatrixPrint(distcov,"New Covariance computation",20,20)

    sc.stop()

  }

}
