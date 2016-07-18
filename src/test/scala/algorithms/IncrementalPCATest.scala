package algorithms

import breeze.linalg.{DenseMatrix, Transpose, min, DenseVector => BDV}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite

import scala.collection.immutable.Range.Inclusive

/**
  * Created by tonypark on 2016. 4. 19..
  */
class IncrementalPCATest extends FunSuite {
  test("Simple run of least-squares regression") {
    val sconf = new SparkConf()
      .setMaster("local[4]")
      .setAppName("MutualInformationTest")
      .set("spark.driver.memory", "8G")

    val sc = new SparkContext(sconf)

    //*******************************************
    // Test one iteration
    //*********************************************

    /*
    //val homedir= "/home/tony/IdeaProjects/mltest01/"
    val homedir ="/Users/tonypark/ideaProjects/"
    val csv: RDD[String] = sc.textFile(homedir + "SparkMLStudy/src/test/resources/data.csv")
    val datatmp: RDD[Array[String]] = csv.map(line => line.split(','))
    val data: RDD[Vector] = datatmp.map(x => Vectors.dense(x.map(i=>i.toDouble)))

    val csv1: RDD[String] = sc.textFile(homedir+"SparkMLStudy/src/test/resources/U.csv")
    val utmp: RDD[Array[String]] = csv1.map(line => line.split(','))
    val u: RDD[Vector] = utmp.map(x => Vectors.dense(x.map(i=>i.toDouble)))

    val csv2: RDD[String] = sc.textFile(homedir + "SparkMLStudy/src/test/resources/S.csv")
    val stmp: RDD[Array[String]] = csv2.map(line => line.split(','))
    val s: Array[Double] = stmp.map(x => x.map(i=>i.toDouble)).collect.flatten

    val csv3: RDD[String] = sc.textFile(homedir+ "SparkMLStudy/src/test/resources/mu0.csv")
    val mu0tmp: RDD[Array[String]] = csv3.map(line => line.split(','))
    val mu0: Array[Double] = mu0tmp.map(x => x.map(i=>i.toDouble)).collect.flatten

    val mi = new IncrementalPCA
    mi.fit(data,u,s,mu0,Array(190.0),1.0)

    */

    //************************************************
    // Test Iterations
    //************************************************

    val NUM_EVECS=16
    val BLOCK_SIZE=5
    val FF = 1
    val NUM_DATA=605

    //val homedir ="/Users/tonypark/ideaProjects/"
    val homedir ="/home/tony/IdeaProjects/"
    val csv: RDD[String] = sc.textFile(homedir + "SparkMLStudy/src/test/resources/dataall.csv")
    val datatmp: RDD[Array[String]] = csv.map(line => line.split(','))
    val data: RDD[Vector] = datatmp.map(x => Vectors.dense(x.map(i=>i.toDouble)))

    val ipca = new IncrementalPCA("aaa")
    val datat: Array[Vector] = ipca.rddTranspose(data).collect

    println ("The number of rows of data transposed is " + datat.length)

    val datain: RDD[Vector] =ipca.rddTranspose( sc.parallelize((0 until BLOCK_SIZE).map(i=>datat(i))))
    var (u: RDD[Vector],s: Array[Double],mu: Array[Double],n)=ipca.fit(datain)

    /*(BLOCK_SIZE until 605 by BLOCK_SIZE).map {ii=>
      val index = ii to min(ii+BLOCK_SIZE-1,NUM_DATA)
      val datain = ipca.rddTranspose(sc.parallelize(index.map( i=>datat(i))))
      val out = ipca.fit(datain,u,s,mu,Array(n),FF,Array(NUM_EVECS))
      u=out._1
      s=out._2
      mu=out._3
      n=out._4

    } */

    ipca.RowMatrixPrint(new RowMatrix(u)," The EigenVectors (PCs)")
    println("EigenValues : "+ s.map(i=>i.toString + ", ").reduce(_+_))
    println("mu : "+ mu.map(i=>i.toString + ", ").reduce(_+_))

    sc.stop()
  }
}
