package algorithms

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.{ DenseMatrix => BDM, DenseVector => BDV}
import algorithms.common.XORShiftRandom
import breeze.numerics.sqrt

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object preprocessing {

  class ArrayImprovements(val arr: Array[Double]) {

    //val res: Array[Double] = operand.clone()
    //blas.daxpy(arr.length, 1.0, arr, 1, res, 1)
    //res
    def +(operand: Array[Double]) = (arr, operand).zipped.map( _ + _ )

    def +(operand: Double) = arr.map( _ + operand )

    def +(operand: Int) = arr.map( _ + operand )

    def -(operand: Array[Double]) = (arr, operand).zipped.map( _ - _ )

    def *(operand: Array[Double]) = (arr, operand).zipped.map( _ * _ )

    def *(operand: Double) = arr.map( _ * operand )

    def /(operand: Array[Double]) = (arr, operand).zipped.map( _ / _ )

    def /(operand: Double) = arr.map( _ / operand )

  }

  implicit def ArrayToArray(arr: Array[Double]) = new ArrayImprovements(arr)

  class ArrayImprovements2(val arr: Array[Int]) {

    def +(operand: Array[Int]) = (arr, operand).zipped.map( _ + _ )

    def /(operand: Array[Int]) = (arr, operand).zipped.map( _ / _ )

  }

  implicit def ArrayToArray2(arr: Array[Int]) = new ArrayImprovements2(arr)

  class TupleArrayImprovements(val arr: Array[(Double, Double)]) {

    def +(operand: Array[(Double, Double)]) = (arr, operand).zipped.map( (x, y) => (x._1 + y._1, x._2 + y._2) )

    def /(operand: Array[(Double, Double)]) = (arr, operand).zipped.map( (x, y) => (x._1 / y._1, x._2 / y._2) )

  }

  implicit def TupleArrayToTupleArray(arr: Array[(Double, Double)]) = new TupleArrayImprovements(arr)

  class TupleArrayImprovements2(val arr: Array[(Double, Int)]) {

    def +(operand: Array[(Double, Int)]) = (arr, operand).zipped.map( (x, y) => (x._1 + y._1, x._2 + y._2) )

    def /(operand: Array[(Double, Int)]) = (arr, operand).zipped.map( (x, y) => (x._1 / y._1, x._2 / y._2) )

  }

  implicit def TupleArrayToTupleArray2(arr: Array[(Double, Int)]) = new TupleArrayImprovements2(arr)

  class RowMatrixImprovements(val rowmat: RowMatrix) {

    // This is faster than built-in RowMatrix member function.
    // You can easily compare performances of both methods.
    //
    // RowMatrix.computeColL2Norm
    // versus
    // RowMatrix.computeColumnSummaryStatistics().normL2.toArray
    //
    // 10 dim : 0.042 vs. 0.062 seconds
    // 10000 dim : 1.188 vs. 8.180 seconds
    def computeColL2Norm: Array[Double] = rowmat.rows
      .map( x => x.toArray.map( y => y*y ) )
      .reduce( _ + _ )
      .map( x => math.sqrt(x) )

    def computeCovarianceRDD(N: Int) = {
      rowmat.rows.mapPartitionsWithIndex { (indx, iter) =>
        iter.flatMap { row =>
          row match {
            case DenseVector(values) =>
              val n = values.size
              Iterator.tabulate (n) { i =>
                val buf = new ListBuffer[((Int, Int), Double)]()
                val iVal = values(i)
                if (iVal != 0) {
                  var j = 0
                  while (j < n) {
                    val jVal = values(j)
                    if (jVal != 0) {
                      buf += (((i, j), iVal * jVal))
                    }
                    j += 1
                  }
                }
                buf
              }.flatten
          }
        }
      }.reduceByKey(_ + _)
        .map{ case ((i, j), sim) => (i, (j, sim)) }
        .groupByKey()
        .sortByKey() // sort by i-th index
        .values
        .map( x => Vectors.dense(x.toArray.sortBy( _._1 ).map( _._2 / (N-1) )) ) // sort by j-th index
    }

  }

  implicit def RowMatrixToRowMatrix(rowmat: RowMatrix) = new RowMatrixImprovements(rowmat)

  class RDDImprovements(val r: RDD[LabeledPoint]) {


    def DenseFeatureMatrix: Matrix = {
      val a1: RDD[Array[Double]] = r.map { x =>
        x.features.toArray
      }
      Matrices.dense(r.count().toInt, r.first.features.toArray.length, a1.toArray.flatten)
    }

    def removeCols(T: Int, L: ArrayBuffer[Int]) = {

      println("Columns with " + T + " NaNs will be removed.")

      val colCountNaN: Array[Int] = r
        .map(x => x.features.toArray.map(y => if (y.equals(Double.NaN)) 1 else 0))
        .reduce((x, y) => x + y)

      colCountNaN.zipWithIndex.filterNot(_._1 <= T).map(_._2)
        .foreach(x => L += x)

      r.map(x => LabeledPoint(x.label, Vectors.dense(
        (x.features.toArray, colCountNaN).zipped.filter((a, b) => b <= T)._1
      )))

    }

    def removeRows(T: Int, L: ArrayBuffer[Int]) = {

      println("Rows with " + T + " NaNs will be removed.")

      val rowCountNaN: RDD[Int] = r
        .map(x => x.features.toArray.map(y => if (y.equals(Double.NaN)) 1 else 0).sum)

      rowCountNaN.collect().zipWithIndex.filterNot(_._1 <= T).map(_._2)
        .foreach(x => L += x)

      r.filter(x => x.features.toArray.map(y => if (y.equals(Double.NaN)) 1 else 0).sum <= T)
      //r.zip(rowCountNaN).filter( _._2 < T ).map( _._1 )

    }

    def replaceNaNWithAverage = {

      println("Every NaN will be replaced.")

      val colMeanWithoutNaN: Array[Double] = r
        .map(x => x.features.toArray.map(y => if (y.equals(Double.NaN)) (0.0, 0) else (y, 1)))
        .reduce(_ + _)
        .map(x => x._1 / x._2)

      r.map(x => LabeledPoint(x.label, Vectors.dense(
        (x.features.toArray, colMeanWithoutNaN).zipped.map((a, b) => if (a.equals(Double.NaN)) b else a)
      )))

    }

    def computeCovarianceRDD(N: Int, P: Int): List[(Int, RowMatrix)] = {

      val colMean: Array[Double] = r
        .map( x => x.features.toArray )
        .reduce( _ + _ )
        .map( x => x / N )

      val MeanShiftedMat: Array[Array[Double]] = r
        .map( x => ( x.features.toArray - colMean ) / math.sqrt(N-1) )
        .collect()

      val transposedMat: Seq[Vector] = (new BDM(P, N, MeanShiftedMat.flatten))
        .t
        .toArray
        .sliding(N,N)
        .toArray
        .map( x => Vectors.dense(x) )
        .toSeq

      val transposedMatRDD: RowMatrix = new RowMatrix( r.context.parallelize(transposedMat) )

      // The number of partitions
      val K: Int = ( if (P < 10000) 2 else if (P < 40000) 2 else 4 )

      val buf: ListBuffer[(Int, RowMatrix)] = new ListBuffer
      for ( i <- 1 to K ) {
        buf += ((i, transposedMatRDD.multiply( Matrices.dense((P / K), N, MeanShiftedMat
          .map( x => x.slice((i - 1) * (P / K), i * (P / K)) ).flatten).transpose ) ))
      }
      buf.toList
      //buf(0)._2.rows.union(buf(1)._2.rows)
    }




    def computeCovarianceRDD1(N: Int, P: Int, part: Int): CoordinateMatrix = {

      val colMean: Array[Double] = r
        .map(x => x.features.toArray)
        .reduce(_ + _)
        .map(x => x / N)

      val gamma =10 * math.log(P) / 1e-5
      val sg = math.sqrt(gamma)

      val colMagsCorrected = colMean.map(x => if (x == 0) 1.0 else x)

      //val sc = r.context

      //val pBV = sc.broadcast(colMagsCorrected.map(c => sg / c))


      val sims: RDD[MatrixEntry] = r.map(x => Vectors.dense(x.features.toArray - colMean))
      .mapPartitionsWithIndex { (indx, iter) =>
       // val p =pBV.value
        val rand = new XORShiftRandom(indx)
        iter.flatMap { row =>
          row match {
            case DenseVector(values) =>
              val n = values.size
              Iterator.tabulate(n) { i =>
                val buf = new ListBuffer[((Int, Int), Double)]()
                val iVal= values(i)
                if (iVal != 0 ){//&& rand.nextDouble() < p(i)) {
                  var j = i
                  while (j < n) {
                    val jVal = values(j)
                    if (jVal != 0 ){//&& rand.nextDouble() < p(j)) {
                      buf += (((i, j), iVal * jVal))
                    }
                    j += 1
                  }
                }
                buf
              }.flatten
          }
        }
      }.reduceByKey(_ + _).map { case ((i, j), sim) =>
        MatrixEntry(i.toLong, j.toLong, sim/N)
      }
      new CoordinateMatrix(sims, P, P)
    }

  def computeCovarianceRDD2(N: Int, P: Int, part: Int): CoordinateMatrix = {

    val colMean: Array[Double] = r
      .map(x => x.features.toArray)
      .reduce(_ + _)
      .map(x => x / N)
    val colL2 = r.map(x=>BDV(x.features.toArray) :* BDV(x.features.toArray))
      .reduce(_+_)
      .map(x=>sqrt(x))

    val gamma =10 * math.log(P) / 1e-6
    val sg = math.sqrt(gamma)

    val colMagsCorrected = colL2.map(x => if (x == 0) 1.0 else x)

    val sc = r.context

    val pBV = sc.broadcast(colMagsCorrected.map(c => sg / c))


    val input = r.map(x => Vectors.dense(x.features.toArray - colMean)).cache()
      val sims = input.mapPartitionsWithIndex { (indx, iter) =>
        val p =pBV.value
        val rand = new XORShiftRandom(indx)
        iter.flatMap { row =>
          row match {
            case DenseVector(values) =>
              val n = values.size
              Iterator.tabulate(n) { i =>
                val buf = new ListBuffer[((Int, Int), Double)]()
                val iVal= values(i)
                if (iVal != 0 && rand.nextDouble() < p(i)) {
                var j = i
                  while (j < n) {
                    val jVal = values(j)
                    if (jVal != 0 && rand.nextDouble() < p(j)) {
                      buf += (((i, j), iVal * jVal))
                    }
                    j += 1
                  }
                }
                buf
              }.flatten
          }
        }
      }.reduceByKey(_ + _).map { case ((i, j), sim) =>
      MatrixEntry(i.toLong, j.toLong, sim/N)
    }
    new CoordinateMatrix(sims, P, P)
  }
}

  implicit def RDDToRDD(r: RDD[LabeledPoint]) = new RDDImprovements(r)

}