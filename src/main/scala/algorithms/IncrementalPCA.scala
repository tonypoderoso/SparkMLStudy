package algorithms

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.{ DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
/**
  * Created by tonyp on 2016-04-27.
  */
class IncrementalPCA {
  def fit(data : RDD[Vector], U0 :RDD[Vector] , D0 : BDV[Vector]): Unit ={
    /*val n=data.rows.first().toArray.length
    val n0=data.numCols()
    val ff=1.0
    val mu1: RDD[Double] = data.rows.map(x=>sum(x.toArray)/n)
    val ndata: RDD[Array[Double]] = data.rows.map{ x =>
        val y: Double = sum(x.toArray)/n
        x.toArray.map(z=>z/y)
    }

     */
    val D = diag(D0)
    val data_proj= data.zipWithIndex.map{case (x,y)=>
        data

    }
    //val res = data - U0.t*data_proj.t


  }

}
