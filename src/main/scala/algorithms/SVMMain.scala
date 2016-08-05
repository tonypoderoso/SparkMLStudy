package algorithms

/**
  * Created by tonypark on 2016. 7. 4..
  */

import breeze.linalg.{DenseVector => BDV}
import breeze.numerics.abs
import breeze.stats.distributions.Gaussian
import org.apache.spark.mllib.linalg._
import org.apache.spark._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.rdd.RDD

import scala.util.Random

object SVMMain {



  // Generate noisy input of the form Y = signum(x.dot(weights) + intercept + noise)
  def generateSVMInput(
                        intercept: Double,
                        weights: Array[Double],
                        nPoints: Int,
                        seed: Int): Seq[LabeledPoint] = {
    val rnd = new Random(seed)
    val weightsMat = new BDV(weights)
    val x = Array.fill[Array[Double]](nPoints)(
      Array.fill[Double](weights.length)(rnd.nextDouble() * 2.0 - 1.0))
    val y = x.map { xi =>
      val yD = new BDV(xi).dot(weightsMat) + intercept + 0.01 * rnd.nextGaussian()
      if (yD < 0) 0.0 else 1.0
    }
    y.zip(x).map(p => LabeledPoint(p._1, Vectors.dense(p._2)))
  }


def main(args:Array[String]):Unit ={

  val sc = new SparkContext(new SparkConf()
    .setMaster("local[*]").setAppName("PCAExampleTest")
    .set("spark.scheduler.mode", "FAIR")
    .set("spark.driver.maxResultSize", "90g")
    .set("spark.akka.timeout", "2000000")
    .set("spark.worker.timeout", "5000000")
    .set("spark.storage.blockManagerSlaveTimeoutMs", "5000000")
    .set("spark.akka.frameSize", "2047")
    .set("spark.akka.threads", "12")
    .set("spark.network.timeout", "7200")
    .set("spark.rpc.askTimeout", "7200")
    .set("spark.rpc.lookupTimeout", "7200")
    .set("spark.network.timeout", "10000000")
    .set("spark.executor.heartbeatInterval", "10000000"))


  var num_features: Int = 10
  var num_samples: Int = 200
  var num_bins: Int = 20
  var num_new_partitions: Int = 100
  var num_partsum:Int = 10

  if (!args.isEmpty) {

    num_features = args(0).toString.toInt
    num_samples = args(1).toString.toInt
    num_bins = args(2).toString.toInt
    num_new_partitions = args(3).toString.toInt
    num_partsum=args(4).toString.toInt

  }

  val recordsPerPartition: Int = num_samples / num_new_partitions
  val gauss: Gaussian = new Gaussian(0.0, 1.0)
  val weight=gauss.sample(num_features).toArray
  val broweight=sc.broadcast(weight)
  val intercept=gauss.sample()
  val brointercept = sc.broadcast(intercept)
  val points: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
    .mapPartitionsWithIndex { (idx, iter) => {
      (1 to recordsPerPartition).flatMap { i =>
        val rnd = new Random
        generateSVMInput(brointercept.value, broweight.value, recordsPerPartition, rnd.nextInt)
      }
    }.toIterator
    }.cache()

  points.collect.foreach {x=>
    print("new KVModel(\"" + x.label.toString +"\", Arrays.asList(")
    x.features.toArray.foreach(y=>print(y+","))
    print(")),")
    println()
  }

  // If we serialize data directly in the task closure, the size of the serialized task would be
  // greater than 1MB and hence Spark would throw an error.
  val model = SVMWithSGD.train(points, 2)
  val predictions: Array[Double] = model.predict(points.map(_.features)).collect
  val inputclass: Array[Double] = points.map(_.label).collect


  val accu = (0 until num_samples).map{i => abs(predictions(i)-inputclass(i))
  }.reduce(_+_)
  println("Accuracy is : " +(1 - accu/num_samples))


}
}