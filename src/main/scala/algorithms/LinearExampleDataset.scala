package algorithms

import breeze.stats.distributions.Gaussian
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{DenseVector=>BDV}

/**
  * Created by tony on 16. 4. 17.
  */
class LinearExampleDataset(n: Int, d: Int, noise: Double) {
  private val g = new Gaussian(0.0,1.0)

  val  weights= g.sample(d).toArray

  val labeledPoints: Seq[LabeledPoint]= {
    val xs = (1 to n).map(i => BDV(g.sample(d).toArray))

    println(xs(1))

    val w = BDV(weights)

    xs.map { x =>
      val l = x.dot(w) + g.sample() * noise
      new LabeledPoint(l, new DenseVector(x.toArray))
    }
  }
}
