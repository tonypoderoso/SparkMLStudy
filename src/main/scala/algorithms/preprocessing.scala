package algorithms

import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import com.github.fommil.netlib.BLAS.{getInstance => blas}

import scala.collection.mutable.ArrayBuffer

object preprocessing {

  class ArrayImprovements(val arr: Array[Double]) {

    //val res: Array[Double] = operand.clone()
    //blas.daxpy(arr.length, 1.0, arr, 1, res, 1)
    //res
    def +(operand: Array[Double]) = (arr, operand).zipped.map( _ + _ )

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

  class RDDImprovements(val r: RDD[LabeledPoint]) {

    def removeCols(T: Int, L: ArrayBuffer[Int]) = {

      println("Columns with " + T + " NaNs will be removed.")

      val colCountNaN: Array[Int] = r
        .map( x => x.features.toArray.map( y => if (y.equals(Double.NaN)) 1 else 0 ) )
        .reduce( (x, y) => x + y )

      colCountNaN.zipWithIndex.filterNot( _._1 < T ).map( _._2 )
        .foreach( x => L += x )

      r.map( x => LabeledPoint(x.label, Vectors.dense(
        (x.features.toArray, colCountNaN).zipped.filter( (a, b) => b < T )._1
      ) ) )

    }

    def removeRows(T: Int, L: ArrayBuffer[Int]) = {

      println("Rows with " + T + " NaNs will be removed.")

      val rowCountNaN: RDD[Int] = r
        .map( x => x.features.toArray.map( y => if (y.equals(Double.NaN)) 1 else 0 ).sum )

      rowCountNaN.toArray.zipWithIndex.filterNot( _._1 < T ).map( _._2 )
        .foreach( x => L += x )

      r.filter( x => x.features.toArray.map( y => if (y.equals(Double.NaN)) 1 else 0 ).sum < T )
      //r.zip(rowCountNaN).filter( _._2 < T ).map( _._1 )

    }

    def replaceNaNWithAverage = {

      println("Every NaN will be replaced.")

      val colMeanWithoutNaN: Array[Double] = r
        .map( x => x.features.toArray.map( y => if (y.equals(Double.NaN)) (0.0, 0) else (y, 1) ) )
        .reduce( _ + _ )
        .map( x => x._1 / x._2 )

      r.map( x => LabeledPoint(x.label, Vectors.dense(
        (x.features.toArray, colMeanWithoutNaN).zipped.map( (a, b) => if (a.equals(Double.NaN)) b else a )
      ) ) )

    }

    def computeCovarianceRDD(N: Int, P: Int) = {

      val colMean: Array[Double] = r
        .map( x => x.features.toArray )
        .reduce( _ + _ )
        .map( x => x / N )

      val MeanShiftedMat: RowMatrix = new RowMatrix( r
        .map( x => Vectors.dense( x.features.toArray - colMean ) ) )

      val colL2Norm: Array[Double] = MeanShiftedMat.rows
        .map( x => x.toArray.map( y => y*y ) )
        .reduce( _ + _ )
        .map( x => math.sqrt(x) )

      //new CoordinateMatrix(
      //  MeanShiftedMat.columnSimilarities().toRowMatrix().rows
      //    .zipWithIndex()
      //    .map( x => Vectors.dense( x._1.toArray * colL2Norm * colL2Norm(x._2.toInt) / (N-1) ) )
      //)

      new CoordinateMatrix(
        MeanShiftedMat.columnSimilarities().entries
          .map( x => MatrixEntry(x.i, x.j, colL2Norm(x.i.toInt) * colL2Norm(x.j.toInt) * x.value / (N-1)) )
      )

    }

  }

  implicit def RDDToRDD(r: RDD[LabeledPoint]) = new RDDImprovements(r)

}