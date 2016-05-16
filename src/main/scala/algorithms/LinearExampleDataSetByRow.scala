package algorithms

import breeze.stats.distributions.Gaussian
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.SparkContext

/**
  * Created by tony on 16. 4. 17.
  */
class LinearExampleDatasetByRow(n: Int, d: Int, noise: Double, sc:SparkContext) {
  private val g = new Gaussian(0.0,1.0)
  val  weights: Array[Double] = g.sample(d).toArray

  val labeledPoints= {
    val xs = (1 to n).map(i => BDV(g.sample(d).toArray))
    val w = BDV(weights)

    xs.map { x =>
      val l = x.dot(w) + g.sample() * noise
      new LabeledPoint(l, Vectors.dense(x.toArray))
    }
  }

  val labeledPointsSc= sc.parallelize(labeledPoints)
}
