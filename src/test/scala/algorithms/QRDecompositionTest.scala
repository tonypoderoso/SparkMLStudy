package algorithms

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import breeze.stats.distributions.Gaussian
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, QRDecomposition, Vector, Vectors}
import org.scalatest.FunSuite

import scala.collection.immutable.IndexedSeq

/**
  * Created by tonyp on 2016-04-26.
  */
class QRDecompositionTest extends FunSuite{

  def toBreeze(in:RowMatrix): BDM[Double] ={
    val m = in.numRows().toInt
  val n = in.numCols().toInt
  val mat = BDM.zeros[Double](m, n)
  var i = 0
  in.rows.collect().foreach { vector =>
    vector.foreachActive { case (j, v) =>
      mat(i, j) = v
    }
    i += 1
  }
  mat
}

  def closeToZero(G: BDM[Double]): Boolean = {
    G.valuesIterator.map(math.abs).sum < 1e-6
  }

  test("QR Decomposition") {

    val g = new Gaussian(0.0,1.0)

    val d =10000
    val nn = 10000
    val xs: IndexedSeq[Vector] =(1 to nn).map(i => Vectors.dense( g.sample(d).toArray))

    val m = 4
    val n = 3
    val arr: Array[Double] = Array(0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 0.0, 2.0, 5.0, 8.0, 1.0)
    val denseData: Seq[Vector] = Seq(
      Vectors.dense(0.0, 1.0, 2.0),
      Vectors.dense(3.0, 4.0, 5.0),
      Vectors.dense(6.0, 7.0, 8.0),
      Vectors.dense(9.0, 0.0, 1.0)
    )
    val sparseData = Seq(
      Vectors.sparse(3, Seq((1, 1.0), (2, 2.0))),
      Vectors.sparse(3, Seq((0, 3.0), (1, 4.0), (2, 5.0))),
      Vectors.sparse(3, Seq((0, 6.0), (1, 7.0), (2, 8.0))),
      Vectors.sparse(3, Seq((0, 9.0), (2, 1.0)))
    )

    val sc = new SparkContext("local","LeastSquaresRegressionTest")

      //val denseMat = new RowMatrix(sc.parallelize(denseData, 2))
      val sparseMat = new RowMatrix(sc.parallelize(sparseData, 2))
      val denseMat =  new RowMatrix(sc.parallelize(xs,4))




    for (mat <- Seq(denseMat, sparseMat)) {
      val result: QRDecomposition[RowMatrix, Matrix] = mat.tallSkinnyQR(computeQ =true)
      val expected = breeze.linalg.qr.reduced(toBreeze(mat))
      val calcQ: RowMatrix = result.Q
      val calcR: Matrix = result.R
      assert(closeToZero(abs(expected.q) - abs(toBreeze(calcQ))))
      //assert(closeToZero(abs(expected.r) - abs(BDM(calcR.toArray))))
      assert(closeToZero(toBreeze(calcQ.multiply(calcR)) - toBreeze(mat)))

    }
  }
}
