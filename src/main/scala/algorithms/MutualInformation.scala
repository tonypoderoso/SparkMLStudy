package algorithms


//import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import breeze.numerics.abs
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import scala.collection.parallel.mutable.ParArray

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
    val byColumnAndRow: RDD[(Int, (Long, Double))] = rdd.zipWithIndex.flatMap {
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

  def discretizeVector(input: RDD[DenseVector], level: Int): RDD[Array[Int]] = {

    input.map { vec =>

      ///val bvec=new BDV[Double](vec)
      val sorted_vec: BDV[Double] = BDV(vec.toArray.sortWith(_ < _))

      val bin_level: BDV[Int] = BDV((1 to level).toArray) * (sorted_vec.length / level)

      val pos: BDV[Double] = bin_level.map { x => sorted_vec(x - 1) }

      vec.toArray.map { x =>
        sum((BDV.fill(level)(x)
          :> pos).toArray.map(y => if (y == true) 1 else 0))
      }
    }
  }

  def computeMutualInformation(vec1: ParArray[Int], vec2: ParArray[Int], num_state1: Int, num_state2: Int): Double = {
    val output = BDM.zeros[Int](num_state1, num_state2)
    val rsum: BDV[Int] = BDV.zeros[Int](num_state1)
    val csum: BDV[Int] = BDV.zeros[Int](num_state2)
    val msum: Int = vec1.length

    vec1.zip(vec2).map { x =>
      output(x._1, x._2) = output(x._1, x._2) + 1
      rsum(x._1) = rsum(x._1) + 1
      csum(x._2) = csum(x._2) + 1
    }

    val MI = output.mapPairs { (coo, x) =>
      if (x > 0) {
        val tmp = msum.toDouble / (rsum(coo._1) * csum(coo._2))
        //println("the tmp :" +tmp +" the x : "+ +x + "  i-th " +rsum(coo._1)+"  j-th "+csum(coo._2)+" msun: " +msum)
        x * math.log(x * tmp) / math.log(2)
      } else
        0
    }.toArray.reduce(_ + _)

    /*
    println(" rsum : ")
    rsum.foreach(print)
    println("\ncsum : ")
    csum.foreach(print)
    println("\nVector1 : ")
    vec1.foreach(print)
    println("\nVector2 : ")
    vec2.foreach(print)
    println("\nMI value : "+ MI/msum)
*/
    MI / msum


  }

  def computeMIMatrix(input: RDD[Array[Int]],num_feature:Int, num_state1: Int, num_state2: Int): BDM[Double] = {
    val output = BDM.zeros[Double](num_feature, num_feature)

    val indexKey: RDD[(Long, Array[Int])] = input.zipWithIndex().map { x => (x._2, x._1) }

    indexKey.cache()

    output.mapPairs { (coor, x) =>
      if (coor._1 >= coor._2) {

        val a: ParArray[Int] = indexKey.lookup(coor._1).flatten.toParArray
        val b: ParArray[Int] = indexKey.lookup(coor._2).flatten.toParArray
        //a.foreach(print)
        //println("\\")
        //b.foreach(print)
        //println("\\")
        output(coor._1, coor._2) = computeMutualInformation(a, b, num_state1, num_state2)
      }

    }
    println(";laksjfkdsll........................................")
    output.mapPairs { (coor, x) =>
      if (coor._1 < coor._2) {
        output(coor._1, coor._2) = output(coor._2, coor._1)
      }
    }
    output
  }
}

