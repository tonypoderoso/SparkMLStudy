package algorithms

import algorithms.preprocessing._
import breeze.stats.distributions.Gaussian
import breeze.linalg.{cov=>brzCov,DenseMatrix=>BDM}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, RowMatrix}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSuite

import scala.collection.immutable.IndexedSeq

class preprocessingSuite extends FunSuite {

  test("Operator Test") {

    val a: Array[Double] = Array(1.1, 2, 5)
    val b: Array[Double] = Array(5.1, 3, 7.1)
    (a + b).foreach(x => println(x))

    val aa: Array[(Double, Double)] = Array((1.1, 0.0), (2.0, 0.1), (5.0, 1.2))
    val bb: Array[(Double, Double)] = Array((5.1, 2.1), (3.2, 1.0), (7.1, 2.0))
    (aa + bb).foreach(println)

  }
/*
  test("PreProcessing Test") {

    Logger.getLogger("org").setLevel(Level.OFF)

    val conf = new SparkConf()
      .setAppName("FirstExample")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    val num_features: Int = 1000
    val num_samples: Int = 100000
    val num_new_partitions: Int = 5 * 16

    val recordsPerPartition: Int = num_samples / num_new_partitions

    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitions { _ => {
        val gauss = new Gaussian(0.0, 1.0)
        val r = scala.util.Random
        (1 to recordsPerPartition).map { _ =>
          val x: Array[Double] = gauss.sample(num_features).toArray
          x(3) = Double.NaN
          (1 to r.nextInt(num_features)).map(i => x(r.nextInt(num_features)) = Double.NaN)
          new LabeledPoint(0, Vectors.dense(x))
        }
      }.toIterator
      }

    val start = System.currentTimeMillis
    val idx1: ArrayBuffer[Int] = ArrayBuffer()
    val idx2: ArrayBuffer[Int] = ArrayBuffer()
    val idx3: ArrayBuffer[Int] = ArrayBuffer()
    val ffd: RDD[LabeledPoint] = lds
      .removeCols( (num_samples*0.5).toInt, idx1 )
      .removeRows( (num_features*0.25).toInt, idx2 )
      .removeCols( (num_samples*0.25).toInt, idx3 )
      .replaceNaNWithAverage
    println("%.3f seconds".format((System.currentTimeMillis - start)/1000.0))
    println(ffd.count() + " rows")
    println(idx1.length)
    println(idx2.length)
    println(idx3.length)

    sc.stop()

  }
*/
  test("Covariance Functionality Test") {

    Logger.getLogger("org").setLevel(Level.OFF)

    val conf = new SparkConf()
      .setAppName("FirstExample")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    val num_features: Int = 10
    val num_samples: Int = 10000
    val num_new_partitions: Int = 5 * 16

    val recordsPerPartition: Int = num_samples / num_new_partitions

    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitions { _ => {
        val gauss = new Gaussian(0.0, 1.0)
        (1 to recordsPerPartition).map { _ =>
          new LabeledPoint(0, Vectors.dense(gauss.sample(num_features).toArray * 5.0))
        }
      }.toIterator
      }

    lds.cache()
    val mat: RowMatrix = new RowMatrix( lds.map( x => Vectors.dense(x.features.toArray) ) )
    val cov: CoordinateMatrix = mat.columnSimilarities()
    val cov2: Matrix = mat.computeCovariance()
    val cov3: RowMatrix = lds.computeCovarianceRDD1(num_samples, num_features,num_new_partitions).toRowMatrix

    println(num_features + " dim")

    println("built-in cosine (RDD)")
    cov.toRowMatrix().rows.collect().foreach(println)

    println("built-in covariance (local)")
    println(cov2.toString())

    println("proposed covariance (RDD)")
    cov3.rows.collect().foreach(println)

    sc.stop()

  }

  test("Covariance Runtime Test") {


    Logger.getLogger("org").setLevel(Level.OFF)
    val conf = new SparkConf()
      .setAppName("FirstExample")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    val num_features: Int = 100
    val num_samples: Int = 1000
    val num_new_partitions: Int = 5 * 16

    val recordsPerPartition: Int = num_samples / num_new_partitions
    val gauss = new Gaussian(0.0, 1.0)
    val dat: IndexedSeq[LabeledPoint] =(1 to num_samples).map { _ =>
      new LabeledPoint(0, Vectors.dense(gauss.sample(num_features).toArray))
    }

    val local = dat.map{x=>x.features.toArray}

    val lds: RDD[LabeledPoint] = sc.parallelize(dat,num_new_partitions)


    lds.cache()
    val start = System.currentTimeMillis
    val distcomp: RowMatrix =lds.computeCovarianceRDD1(num_samples, num_features,num_new_partitions).toRowMatrix


    println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start)/1000.0))
    //val localmat: Matrix =lds.DenseFeatureMatrix
    //println(localmat)
    //val localcov: BDM[Double] =brzCov(new BDM(localmat.numRows,localmat.numCols,localmat.toArray))

    //println(localcov)
    //val firstelem: Array[Double] = distcomp.rows.map{ x=>x.toArray(0)}.collect


    //(0 until 100).foreach{i=>
   // println(firstelem(i) + " vs " + localcov(i,0) )}

    sc.stop()

  }

}