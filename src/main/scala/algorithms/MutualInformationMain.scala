package algorithms

import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by tonypark on 2016. 4. 19..
  */
object MutualInformationMain {

  def main(args:Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("Test"))
    val num_features = 500
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

    val trans: RDD[DenseVector] = mi.rddTranspose(unitdata)

    val dvec: RDD[Array[Int]] = mi.discretizeVector(trans,num_bins)

    dvec.take(5).map{x=>
      x.foreach(print)
      println(" \\\\\\")
    }

    // val mut =mi.computeMutualInformation(dvec.first(),dvec.first,5,5)

    val MIMAT: DenseMatrix[Double] =mi.computeMIMatrix(dvec,num_features,num_bins,num_bins)

    MIMAT.foreachPair{ (x,y)=>println(x._1 + ", " + x._2 + " --> " + y) }
    println(MIMAT.rows)
    println(MIMAT.cols)
    println("///////////////////////////////////////////")

    val mrMR=new MaxRelevanceOverMinRedundancyFeatureSelection
    val H: DenseMatrix[Double] =MIMAT(0 until MIMAT.rows-1,0 until MIMAT.cols-1)
    val f: BDV[Double] = MIMAT(MIMAT.rows-1,0 until MIMAT.cols-1).t
    println("The H is a maxrix of size " +H.rows +" rows and  " + H.cols +" columns")
    println("The f vector is of length"+ f.length)

    val mrMRFS = mrMR.evaluate( H,f, num_features-1)
    println("the result is: ")
    mrMRFS.foreach(println)

  }
}
