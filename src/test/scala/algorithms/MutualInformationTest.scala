package algorithms

import breeze.linalg.DenseMatrix
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite

/**
  * Created by tonypark on 2016. 4. 19..
  */
class MutualInformationTest extends FunSuite{
  test("Simple run of least-squares regression"){
    val sc = new SparkContext("local","LeastSquaresRegressionTest")

    val dataset = new LinearExampleDataset(100,99,0.1)


    //println("//////////////////////////////")
    //dataset.labeledPoints.map(_.label).take(10).foreach(println)
    //dataset.labeledPoints.map(_.features).take(10).foreach(println)
   // println("//////////////////////////////")

    val lds = sc.parallelize(dataset.labeledPoints)

    val mi = new MutualInformation

    val unitdata: RDD[DenseVector]= mi.normalizeToUnit(lds,1)

    val trans: RDD[DenseVector] = mi.rddTranspose(unitdata)

    val dvec: RDD[Array[Int]] = mi.discretizeVector(trans,10)

    dvec.take(5).map{x=>
      x.foreach(print)
      println(" \\\\\\")
    }

   // val mut =mi.computeMutualInformation(dvec.first(),dvec.first,5,5)

    val MIMAT: DenseMatrix[Double] =mi.computeMIMatrix(dvec,100,100)

    MIMAT.foreachPair{ (x,y)=>println(x._1 + ", " + x._2 + " --> " + y) }

    println("///////////////////////////////////////////")
    val temp = new Array[Array[Double]](5)
    MIMAT.mapPairs{
    }
  }
    for (i<-0 until 5 ; j<-0 until 5){
      temp.apply(i).apply(j)=
    sc.stop()

  }




}
