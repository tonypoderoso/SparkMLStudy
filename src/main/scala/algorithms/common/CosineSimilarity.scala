package algorithms.common

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
//import org.apache.spark.util.random.XORShiftRandom

import scala.collection.mutable.ListBuffer

/**
  * Created by tonypark on 2016. 6. 7..
  */
class CosineSimilarity {


  /**
    * Find all similar columns using the DIMSUM sampling algorithm, described in two papers
    *
    * http://arxiv.org/abs/1206.2082
    * http://arxiv.org/abs/1304.1467
    *
    * @param colMags A vector of column magnitudes
    * @param gamma The oversampling parameter. For provable results, set to 10 * log(n) / s,
    *              where s is the smallest similarity score to be estimated,
    *              and n is the number of columns
    * @return An n x n sparse upper-triangular matrix of cosine similarities
    *         between columns of this matrix.
    */
def columnSimilaritiesDIMSUM( mat:RowMatrix,
                                               colMags: Array[Double],
                                               gamma: Double): CoordinateMatrix = {
    require(gamma > 1.0, s"Oversampling should be greater than 1: $gamma")
    require(colMags.size == mat.numCols(), "Number of magnitudes didn't match column dimension")
    val sg = math.sqrt(gamma) // sqrt(gamma) used many times

    // Don't divide by zero for those columns with zero magnitude
    val colMagsCorrected = colMags.map(x => if (x == 0) 1.0 else x)

    val sc = mat.rows.context
    val ttt: Array[Double] =colMagsCorrected.map(c => sg / c)
    val pBV = sc.broadcast(colMagsCorrected.map(c => sg / c))
    val qBV = sc.broadcast(colMagsCorrected.map(c => math.min(sg, c)))

    val sims = mat.rows.mapPartitionsWithIndex { (indx, iter) =>
      val p = pBV.value
      val q = qBV.value
      val ttt: Iterator[Vector] = iter
      val rand = new XORShiftRandom(indx)
      val scaled = new Array[Double](p.size)
      iter.flatMap { row =>
        row match {
          case SparseVector(size, indices, values) =>
            val nnz = indices.size
            var k = 0
            while (k < nnz) {
              scaled(k) = values(k) / q(indices(k))
              k += 1
            }

            Iterator.tabulate (nnz) { k =>
              val buf = new ListBuffer[((Int, Int), Double)]()
              val i = indices(k)
              val iVal = scaled(k)
              if (iVal != 0 && rand.nextDouble() < p(i)) {
                var l = k + 1
                while (l < nnz) {
                  val j = indices(l)
                  val jVal = scaled(l)
                  if (jVal != 0 && rand.nextDouble() < p(j)) {
                    buf += (((i, j), iVal * jVal))
                  }
                  l += 1
                }
              }
              buf
            }.flatten
          case DenseVector(values) =>
            val n = values.size
            var i = 0
            while (i < n) {
              scaled(i) = values(i) / q(i)
              i += 1
            }
            Iterator.tabulate (n) { i =>
              val buf: ListBuffer[((Int, Int), Double)] = new ListBuffer[((Int, Int), Double)]()
              val iVal = scaled(i)
              if (iVal != 0 && rand.nextDouble() < p(i)) {
                var j = i + 1
                while (j < n) {
                  val jVal = scaled(j)
                  if (jVal != 0 && rand.nextDouble() < p(j)) {
                    buf. += (((i, j), iVal * jVal))
                  }
                  j += 1
                }
              }
              buf
            }.flatten
        }
      }
    }.reduceByKey(_ + _).map { case ((i, j), sim) =>
      MatrixEntry(i.toLong, j.toLong, sim)
    }
    new CoordinateMatrix(sims, mat.numCols(), mat.numCols())
  }


def columnSimilarities(mat:RowMatrix,
                       threshold: Double): CoordinateMatrix = {
  require(threshold >= 0, s"Threshold cannot be negative: $threshold")

  if (threshold > 1) {
  println(s"[warn] Threshold is greater than 1: $threshold " +
  "Computation will be more efficient with promoted sparsity, " +
  " however there is no correctness guarantee.")
}

  val gamma = if (threshold < 1e-6) {
  Double.PositiveInfinity
} else {
  10 * math.log(mat.numCols()) / threshold
}

  columnSimilaritiesDIMSUM(mat,mat.computeColumnSummaryStatistics().normL2.toArray, gamma)
}
}