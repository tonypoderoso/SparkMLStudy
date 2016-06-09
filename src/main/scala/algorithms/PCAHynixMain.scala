package algorithms

import java.util.Arrays

import algorithms.common.{BLAS, BreezeConversion}
import breeze.linalg.{Transpose, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, svd => brzSvd}
import breeze.stats.distributions.Gaussian
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by tonypark on 2016. 6. 8..
  */
object PCAHynixMain {
  def asBreeze(mat:Matrix): BM[Double] = {
    if (!mat.isTransposed) {
      new BDM[Double](mat.numRows, mat.numCols, mat.toArray)
    } else {
      val breezeMatrix = new BDM[Double](mat.numCols, mat.numRows, mat.toArray)
      breezeMatrix.t
    }
  }



  def computePrincipalComponentsAndExplainedVariance(mat:RowMatrix,k: Int): (Matrix, Vector) = {
    val n = mat.numCols().toInt
    require(k > 0 && k <= n, s"k = $k out of range (0, n = $n]")

    val Cov = asBreeze(mat.computeCovariance()).asInstanceOf[BDM[Double]]

    val brzSvd.SVD(u: BDM[Double], s: BDV[Double], _) = brzSvd(Cov)

    val eigenSum = s.data.sum
    val explainedVariance = s.data.map(_ / eigenSum)

    if (k == n) {
      (Matrices.dense(n, k, u.data), Vectors.dense(explainedVariance))
    } else {
      (Matrices.dense(n, k, Arrays.copyOfRange(u.data, 0, n * k)),
        Vectors.dense(Arrays.copyOfRange(explainedVariance, 0, k)))
    }
  }

  def triuToFull( n: Int, U: Array[Double]): Matrix = {
    val G = new BDM[Double](n, n)

    var row = 0
    var col = 0
    var idx = 0
    var value = 0.0
    while (col < n) {
      row = 0
      while (row < col) {
        value = U(idx)
        G(row, col) = value
        G(col, row) = value
        idx += 1
        row += 1
      }
      G(col, col) = U(idx)
      idx += 1
      col +=1
    }

    Matrices.dense(n, n, G.data)
  }


  def computeGramianMatrix(mat:RowMatrix): Matrix = {
    val n = mat.numCols().toInt
    //mat.checkNumColumns(n)
    // Computes n*(n+1)/2, avoiding overflow in the multiplication.
    // This succeeds when n <= 65535, which is checked above
    val nt = if (n % 2 == 0) ((n / 2) * (n + 1)) else (n * ((n + 1) / 2))

    // Compute the upper triangular part of the gram matrix.
    val GU: BDV[Double] = mat.rows.treeAggregate(new BDV[Double](nt))(
      seqOp = (U, v) => {
        BLAS.spr(1.0, v, U.data)
        U
      }, combOp = (U1, U2) => U1 += U2)

    triuToFull(n, GU.data)
  }



  def main(args: Array[String]): Unit = {


    /// preprocessing /////////
    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]").setAppName("PCAHynixTest"))
    //val sc = new SparkContext(new SparkConf().setMaster("local[*]"))



    //val pca = new PCA(k).fit(dataRDD)

    val csv: RDD[String] = sc.textFile("/Users/tonypark/ideaProjects/SparkMLStudy/src/test/resources/pcatest.csv")

    val dataRDD: RDD[Vector] = csv.map(line => Vectors.dense( line.split(",").map(elem => elem.trim.toDouble)))
    val mat = new RowMatrix(dataRDD)
    val (pc: Matrix, explainedVariance: Vector) = computePrincipalComponentsAndExplainedVariance(mat,mat.numCols.toInt)


    val mat_multiply: RowMatrix = mat.multiply(pc)

    val sampleMeanVector: BDV[Double] =new  BDV(mat.computeColumnSummaryStatistics().mean.toArray)

    val diff: BDV[Double] = (sampleMeanVector.t * asBreeze(pc).asInstanceOf[BDM[Double]]).t

    val score: RDD[BDV[Double]] = mat_multiply.rows.map{ x=>
      BDV(x.toArray) :-diff
    }

    println("The pc matrix ")

    println(pc)


    println( " The score Matrix ")
    score.take(10).foreach{x=>
       x.toArray.map{elem =>
         print(elem.toString +"," )
         }
      println("")
    }


    println (" Explained Variance")
    println(explainedVariance)


  }
}
