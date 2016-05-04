package algorithms

import breeze.linalg.{DenseMatrix, Transpose, DenseVector => BDV}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite

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

    val csv: RDD[String] = sc.textFile("/Users/tonypark/ideaProjects/SparkMLStudy/src/test/resources/data.csv")
    val datatmp: RDD[Array[String]] = csv.map(line => line.split(','))
    val data: RDD[Vector] = datatmp.map(x => Vectors.dense(x.map(i=>i.toDouble)))

    val csv1: RDD[String] = sc.textFile("/Users/tonypark/ideaProjects/SparkMLStudy/src/test/resources/U.csv")
    val utmp: RDD[Array[String]] = csv1.map(line => line.split(','))
    val u: RDD[Vector] = utmp.map(x => Vectors.dense(x.map(i=>i.toDouble)))

    val csv2: RDD[String] = sc.textFile("/Users/tonypark/ideaProjects/SparkMLStudy/src/test/resources/S.csv")
    val stmp: RDD[Array[String]] = csv2.map(line => line.split(','))
    val s: Array[Double] = stmp.map(x => x.map(i=>i.toDouble)).collect.flatten

    val mi = new IncrementalPCA
    mi.fit(data,u,s)
    sc.stop()
  }
}
