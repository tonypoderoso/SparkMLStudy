package algorithms

import org.apache.spark.SparkContext
import org.scalatest._

/**
  * Created by tony on 16. 4. 17.
  */
class LeastSquareRegressionTest extends FunSuite{
  test("Simple run of least-squares regression"){
    val sc = new SparkContext("local","LeastSquaresRegressionTest")

    val dataset = new LinearExampleDataset(100,3,0.1)

    println("//////////////////////////////")
    dataset.labeledPoints.map(_.label).take(10).foreach(println)
    dataset.labeledPoints.map(_.features).take(10).foreach(println)

    println("//////////////////////////////")

    val lds = sc.parallelize(dataset.labeledPoints)

    val lsr = new LeastSquaresRegression
    val weights=lsr.fit(lds)

    println("Real weight= " + dataset.weights.toSeq)
    println("Fitted weights = " + weights)


    sc.stop()

  }


}
