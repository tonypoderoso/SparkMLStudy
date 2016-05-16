package algorithms

import breeze.linalg.{DenseMatrix, Transpose, DenseVector => BDV}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd._
import org.scalatest.FunSuite

/**
  * Created by tonypark on 2016. 4. 19..
  */
class MutualInformationTest extends FunSuite{
  test("Simple run of least-squares regression"){
    val sconf=new SparkConf()
      .setMaster("local[*]")
      .setAppName("MutualInformationTest")
      .set("spark.driver.memory","15G")

    val sc = new SparkContext(sconf)
    val num_features = 2
    val num_samples = 100000
    val num_bins = 200
    val dataset = new LinearExampleDataset(num_samples,num_features-1,0.1)


    //println("//////////////////////////////")
    //dataset.labeledPoints.map(_.label).take(10).foreach(println)
    //dataset.labeledPoints.map(_.features).take(10).foreach(println)
   // println("//////////////////////////////")

    val lds: RDD[LabeledPoint] = sc.parallelize(dataset.labeledPoints)

    val mi = new MutualInformation

    val unitdata: RDD[DenseVector]= mi.normalizeToUnit(lds,1)


    val trans: RDD[Vector] = mi.rddTranspose1(unitdata)

    val dvec: RDD[Array[Int]] = mi.discretizeVector1(trans,num_bins)

    dvec.take(5).map{x=>
      x.foreach(print)
      println(" \\\\\\")
    }

   // val mut =mi.computeMutualInformation(dvec.first(),dvec.first,5,5)



    val MIRDD: RDD[MatrixEntry] =mi.computeMIMatrixRDD(dvec,num_features,num_bins,num_bins)

    println("The number of elementes :" + MIRDD.count())
    MIRDD.foreach {x => println("r : " + x.i + " c : "+x.j + " value : "+ x.value)}



    /*val MIMAT: DenseMatrix[Double] =mi.computeMIMatrix(dvec,num_features,num_bins,num_bins)


    MIMAT.foreachPair{ (x,y)=>println(x._1 + ", " + x._2 + " --> " + y) }
    println(MIMAT.rows)
    println(MIMAT.cols)
    println("///////////////////////////////////////////")


    val mrMR=new minRedundancyMaxRelevanceFeatureSelection
    val H: DenseMatrix[Double] =MIMAT(0 until MIMAT.rows-1,0 until MIMAT.cols-1)
    val f: BDV[Double] = MIMAT(MIMAT.rows-1,0 until MIMAT.cols-1).t
    println("The H is a maxrix of size " +H.rows +" rows and  " + H.cols +" columns")
    println("The f vector is of length"+ f.length)

    val mrMRFS = mrMR.evaluate( H,f, num_features-1)
    println("the result is: ")
    mrMRFS.foreach(println)*/

  }
}
