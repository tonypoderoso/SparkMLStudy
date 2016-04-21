package algorithms


//import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.{max, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.abs
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

//import org.apache.spark.sql._



/**
  * Created by tonypark on 2016. 4. 19..
  */
class MutualInformation {

  def normalizeToUnit(inputs: RDD[LabeledPoint], normConst: Double): RDD[DenseVector] = {

    //val features: RDD[Vector] = inputs.map(_.features)
    val features: RDD[Vector] = inputs.map { v =>
      val x: BDV[Double] = new BDV(v.features.toArray)
      val y: BDV[Double] = new BDV(Array(v.label))
      new DenseVector(BDV.vertcat(x, y).toArray)
    }
    val featuresRow: RowMatrix = new RowMatrix(features)
    val Max: BDV[Double] = new BDV(featuresRow.computeColumnSummaryStatistics().max.toArray)
    val Min: BDV[Double] = new BDV(featuresRow.computeColumnSummaryStatistics().min.toArray)
    val FinalMax: BDV[Double] = max(abs(Max), abs(Min))

    val result: RDD[DenseVector] = features.map((v: Vector) => new DenseVector((BDV(v.toArray) :/ FinalMax).toArray))

    return result

  }


  def rddTranspose(rdd: RDD[DenseVector]): RDD[DenseVector] = {
    // Split the matrix into one number per line.
    val byColumnAndRow: RDD[(Int, (Long, Double))] = rdd.zipWithIndex.flatMap{
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number, columnIndex) => columnIndex ->(rowIndex, number)
      }
    }
    // Build up the transposed matrix. Group and sort by column index first.
    val byColumn: RDD[Iterable[(Long, Double)]] = byColumnAndRow.groupByKey.sortByKey().values
    // Then sort by row index.
    val transposed: RDD[DenseVector] = byColumn.map {
      indexedRow => new DenseVector(indexedRow.toArray.sortBy(_._1).map(_._2))
    }
    return transposed
  }
}
