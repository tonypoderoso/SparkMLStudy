package algorithms

import breeze.linalg.{DenseMatrix, Transpose, min, DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
//import org.scalatest.FunSuite

import scala.collection.immutable.Range.Inclusive

/**
  * Created by tonypark on 2016. 4. 19..
  */
object IncrementalPCAMain{
  def featureFromDataset(inputs: RDD[LabeledPoint], normConst: Double): RDD[Vector] ={
    inputs.map { v =>
      val x: BDV[Double] = new BDV(v.features.toArray)
      val y: BDV[Double] = new BDV(Array(v.label))
      Vectors.dense(BDV.vertcat(x, y).toArray)
    }
  }


  def main(args:Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf()
      //.setMaster("local[*]")
      .setAppName("Random Forest")
      .set("spark.driver.maxResultSize", "40g")
      .set("spark.akka.timeout","20000")
      .set("spark.worker.timeout","50000")
      .set("spark.storage.blockManagerSlaveTimeoutMs","500000")
      .set("spark.akka.frameSize", "1024"))


    var num_features:Int =10000
    var num_samples:Int =100000
    var num_bins:Int = 200
    var num_new_partitions:Int = 5*16

    if (!args.isEmpty){

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt

    }
    // Distributed Data generation
    //************************************************
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




    val datat: RDD[Vector] = featureFromDataset(lds,1)
    //val datat: Array[Vector] =

    val NUM_EVECS=16
    val BLOCK_SIZE=1000
    val FF = 1
    val NUM_DATA=num_samples

    //val homedir ="/Users/tonypark/ideaProjects/"
    //val homedir ="/home/tony/IdeaProjects/"
    //val csv: RDD[String] = sc.textFile(homedir + "SparkMLStudy/src/test/resources/dataall.csv")
    //val datatmp: RDD[Array[String]] = csv.map(line => line.split(','))
    //val data: RDD[Vector] = datatmp.map(x => Vectors.dense(x.map(i=>i.toDouble)))

    val ipca = new IncrementalPCA
    //val datat: Array[Vector] = ipca.rddTranspose(data).collect

    println ("The number of rows of data transposed is " + datat.count)

    //val datain: RDD[Vector] =ipca.rddTranspose( sc.parallelize((0 until BLOCK_SIZE).map(i => datat(i))))
    var (u: RDD[Vector],s: Array[Double],mu: Array[Double],n)=ipca.fit(datain)

    (BLOCK_SIZE until 605 by BLOCK_SIZE).map {ii=>
      val index = ii to min(ii+BLOCK_SIZE-1,NUM_DATA)
      val datain = ipca.rddTranspose(sc.parallelize(index.map( i=>datat(i))))
      val out = ipca.fit(datain,u,s,mu,Array(n),FF,Array(NUM_EVECS))
      u=out._1
      s=out._2
      mu=out._3
      n=out._4

    }

    ipca.RowMatrixPrint(new RowMatrix(u)," The EigenVectors (PCs)")
    println("EigenValues : "+ s.map(i=>i.toString + ", ").reduce(_+_))
    println("mu : "+ mu.map(i=>i.toString + ", ").reduce(_+_))

    sc.stop()
  }
}

