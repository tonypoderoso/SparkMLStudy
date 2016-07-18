package algorithms

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import breeze.stats.distributions.Gaussian
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

//import breeze.linalg.DenseVector
/**
  * Created by tonypark on 2016. 6. 20..
  */
class MatlabTest{
  def testBreeze(arr: Array[Double]): BDM[Double] ={
    val a: BDV[Double] =BDV(arr)
    a * a.t
  }

  def SayHi(message: String): Unit ={ System.out.println( message )}
  def computeAA = {
    val a = Array(1.0, 2, 4 , 5 )
    a.foreach(println)
  }
  def sparktest: Array[Double] = {

    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]")
      .setAppName("MatlabTest"))

      //.set("spark.driver.maxResultSize", "90g")
      //.set("spark.akka.timeout","200000")
      //.set("spark.worker.timeout","500000")
      //.set("spark.storage.blockManagerSlaveTimeoutMs","5000000")
      //.set("spark.akka.frameSize", "1024"))
    //val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("Test"))

    var num_features:Int =1000
    var num_samples:Int =100000
    var num_bins:Int = 200
    var num_new_partitions:Int = 5*16



    val recordsPerPartition: Int = num_samples/num_new_partitions

    val noise = 0.1
    val gauss = new Gaussian(0.0,1.0)
    val weights: Array[Double] = gauss.sample(num_features).toArray
    //val w = BDV(weights)
    weights(3)=0.0

    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](),num_new_partitions)
      .mapPartitions { _ => {
        val gauss=new Gaussian(0.0,1.0)
        val r= scala.util.Random
        (1 to recordsPerPartition).map { _ =>
          val x =gauss.sample(num_features).toArray
          x(3)=Double.NaN


          val l = x.zip(weights).map{case(i,j)=>i*j}.reduce(_+_) + gauss.sample() * noise
          (1 to r.nextInt(num_features)).map(i=> x(r.nextInt(num_features))=Double.NaN)
          new LabeledPoint(l, Vectors.dense(x))
        }
      }.toIterator
      }

    val res = lds.first.features.toArray
    sc.stop()
    res
  }

}
