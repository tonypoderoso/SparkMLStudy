package algorithms

import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => BDM}
import breeze.linalg.{DenseVector => BDV}

/**
  * Created by tony on 16. 4. 17.
  */
class LeastSquaresRegression {
  def fit(dataset:RDD[LabeledPoint]):DenseVector= {
    val features: RDD[Vector] = dataset.map(_.features)
    println("the first line of features: " + features.take(1).foreach(println))
    val covarianceMatrix: BDM[Double] = features.map { v =>
      val x = BDM(v.toArray)
      x.t * x
    }.reduce(_ + _)
    val featureTimesLabels: BDV[Double] = dataset.map{xy=>
      BDV(xy.features.toArray)*xy.label
    }.reduce(_+_)
    val weight: BDV[Double] = covarianceMatrix \ featureTimesLabels
    new DenseVector(weight.data)
  }



}
