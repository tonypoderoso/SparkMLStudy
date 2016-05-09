package algorithms



import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import breeze.numerics.abs
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRowMatrix, MatrixEntry, RowMatrix}

import scala.collection.immutable.IndexedSeq

/**
  * Created by tonypark on 2016. 5. 4..
  */
class MutualInformation {

  def toBreeze(rm: RowMatrix): BDM[Double] = {
    val m = rm.numRows().toInt
    val n = rm.numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    var i = 0
    rm.rows.collect().foreach { vector =>
      vector.foreachActive { case (j, v) =>
        mat(i, j) = v
      }
      i += 1
    }
    mat
  }

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

    features.map((v: Vector) => new DenseVector((BDV(v.toArray) :/ FinalMax).toArray))


  }


  def rddTranspose1(rdd: RDD[DenseVector]): RDD[Vector] = {
    // Split the matrix into one number per line.
    val sc = rdd.sparkContext
    val mentry: RDD[MatrixEntry] = rdd.zipWithIndex.flatMap{
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number: Double, columnIndex: Int) => MatrixEntry(columnIndex ,rowIndex, number)
      }
    }
    new CoordinateMatrix(mentry).toRowMatrix().rows
  }

  def rddTranspose(rdd: RDD[DenseVector]): RDD[DenseVector] = {
    // Split the matrix into one number per line.
    val byColumnAndRow: RDD[(Int, (Long, Double))] = rdd.zipWithIndex.flatMap{
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number: Double, columnIndex: Int) => {columnIndex ->(rowIndex, number)}
      }
    }
    // Build up the transposed matrix. Group and sort by column index first.
    val byColumn: RDD[Iterable[(Long, Double)]] = byColumnAndRow.groupByKey.sortByKey().values
    // Then sort by row index.
    byColumn.map {
      indexedRow => new DenseVector(indexedRow.toArray.sortBy(_._1).map(_._2))
    }
  }

  def discretizeVector(input:RDD[DenseVector],level: Int): RDD[Array[Int]] ={

    input.map { vec =>

      ///val bvec=new BDV[Double](vec)
      val sorted_vec: BDV[Double] = BDV(vec.toArray.sortWith(_ < _))

      val bin_level: BDV[Int] = BDV((1 to level).toArray) * (sorted_vec.length / level)

      val pos: BDV[Double] = bin_level.map { x => sorted_vec(x - 1) }

      vec.toArray.map { x =>
        sum((BDV.fill(level) (x)
           :> pos).toArray.map(y => if (y == true) 1 else 0))
      }
    }
  }

  def discretizeVector1(input:RDD[Vector],level: Int): RDD[Array[Int]] ={

    input.map { vec =>

      ///val bvec=new BDV[Double](vec)
      val sorted_vec: BDV[Double] = BDV(vec.toArray.sortWith(_ < _))

      val bin_level: BDV[Int] = BDV((1 to level).toArray) * (sorted_vec.length / level)

      val pos: BDV[Double] = bin_level.map { x => sorted_vec(x - 1) }

      vec.toArray.map { x =>
        sum((BDV.fill(level) (x)
          :> pos).toArray.map(y => if (y == true) 1 else 0))
      }
    }
  }

  def computeMutualInformation(vec1:Array[Int] ,num_state1:Int, vec2:Array[Int],num_state2:Int): Double ={
    val output = BDM.zeros[Int](num_state1,num_state2)
    val rsum: BDV[Int] = BDV.zeros[Int](num_state1)
    val csum: BDV[Int] = BDV.zeros[Int](num_state2)
    val msum: Int = vec1.length

    vec1.zip(vec2).map { x =>
      output(x._1,x._2) = output(x._1,x._2) + 1
      rsum(x._1) = rsum(x._1) + 1
      csum(x._2) = csum(x._2) + 1
    }

    val MI = output.mapPairs{ (coo, x) =>
      if (x>0) {
        val tmp = msum.toDouble / (rsum(coo._1) * csum(coo._2))
        //println("the tmp :" +tmp +" the x : "+ +x + "  i-th " +rsum(coo._1)+"  j-th "+csum(coo._2)+" msun: " +msum)
        x * math.log(x*tmp) / math.log(2)
      }else
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
    MI/msum



  }


  def computeMIMatrixRDD1(in:RDD[Array[Int]],num_features:Int,num_state1:Int,num_state2:Int): RDD[MatrixEntry] = {

    val sc = in.sparkContext

    val computeMutualInformation1= ( input: ((Array[Int], Long),(Array[Int],Long))) =>{

      val row = input._1._2
      val col = input._2._2
      var res :MatrixEntry = MatrixEntry(-1,-1,-1)
      if (row >= col) {

        //val num_state2=input._2._1.length
        val output = BDM.zeros[Int](num_state1, num_state2)
        val rsum: BDV[Int] = BDV.zeros[Int](num_state1)
        val csum: BDV[Int] = BDV.zeros[Int](num_state2)
        val msum: Int = input._1._1.length

        input._1._1.zip(input._2._1).map { x =>
          output(x._1, x._2) = output(x._1, x._2) + 1
          rsum(x._1) = rsum(x._1) + 1
          csum(x._2) = csum(x._2) + 1
        }

        val MI = output.mapPairs { (coo, x) =>
          if (x > 0) {
            val tmp = msum.toDouble / (rsum(coo._1) * csum(coo._2))
            x * math.log(x * tmp) / math.log(2)
          } else
            0
        }.toArray.reduce(_ + _)

        val tmp= MI / msum
        //if (row == col)
        res=MatrixEntry(row, col, tmp)
        //println("r : " + res.i + " c : "+res.j + " value : "+ res.value)


      //else
        // res=Seq(MatrixEntry(row, col, tmp), MatrixEntry(col, row, tmp))

      }
      res
    }

    val a=in.zipWithIndex()
    val b=in.zipWithIndex()
    a.cartesian(b).map(computeMutualInformation1)

  }

  def computeMIMatrixRDD(input:RDD[Array[Int]],num_features:Int,num_state1:Int,num_state2:Int)
  : RDD[MatrixEntry] = {

    val sc = input.sparkContext

    val in: Array[Array[Int]] =input.collect()

    sc.parallelize(
      (0 until num_features).flatMap{ row =>
        (row until num_features).flatMap { col =>
          //println("Computhing " +row.toString+ "-th row  and "+ col.toString + "-th column")
          val a: Array[Int] = in(row) //indexKey.lookup(row).flatten.toArray
          val b: Array[Int] = in(col) //indexKey.lookup(col).flatten.toArray
          val tmp: Double =  computeMutualInformation(a, num_state1,  b,num_state2)
          if (row == col)
            Seq(MatrixEntry(row, col, tmp))
          else
            Seq(MatrixEntry(row, col, tmp), MatrixEntry(col, row, tmp))

        }
      })
  }

  /*def computeMIMatrixRDD(input:RDD[Array[Int]],num_features:Int,num_state1:Int,num_state2:Int)
  : RDD[MatrixEntry] = {

    val sc = input.sparkContext

    val indexKey: RDD[(Long, Array[Int])] = input.zipWithIndex().map { x => (x._2, x._1) }

    indexKey.cache()

    //val entries: RDD[MatrixEntry] =
    /*sc.parallelize(
      (0 until num_features).map { row =>
        (row until num_features).map { col =>
          val a: Array[Int] = indexKey.lookup(row).flatten.toArray
          val b: Array[Int] = indexKey.lookup(col).flatten.toArray
          val tmp = computeMutualInformation(a, b, num_state1, num_state2)
          if (row == col)
            Seq(MatrixEntry(row, col, tmp))
          else
            Seq(MatrixEntry(row, col, tmp), MatrixEntry(col, row, tmp))
        }.flatten
      }.flatten)*/
    sc.parallelize(
      (0 until num_features).flatMap{ row =>
        (row until num_features).flatMap { col =>
          //println("Computhing " +row.toString+ "-th row  and "+ col.toString + "-th column")
          val a: Array[Int] = indexKey.lookup(row).flatten.toArray
          val b: Array[Int] = indexKey.lookup(col).flatten.toArray
          val tmp: Double =  computeMutualInformation(a, b, num_state1, num_state2)
          //print(" First Array : ")
          //a.map(i=>i.toString + " , ").foreach(print)
          //println("")
          //print(" Second Array : ")
          //b.map(i=>i.toString + " , ").foreach(print)
          //println("")
          //println(" The Mutual Information value :" + tmp)
          if (row == col)
            Seq(MatrixEntry(row, col, tmp))
          else
            Seq(MatrixEntry(row, col, tmp), MatrixEntry(col, row, tmp))

        }
      })
    //val output = new CoordinateMatrix(entries)
  }*/

  def computeMIMatrix(input:RDD[Array[Int]],num_features:Int,num_state1:Int,num_state2:Int): BDM[Double] ={
    val output = BDM.zeros[Double](num_features,num_features)

    val indexKey: RDD[(Long, Array[Int])] =input.zipWithIndex().map{ x => (x._2,x._1)}

    indexKey.cache()

    output.mapPairs { (coor, x) =>
      if (coor._1 >= coor._2) {

        val a: Array[Int] =indexKey.lookup(coor._1).flatten.toArray
        val b: Array[Int] =indexKey.lookup(coor._2).flatten.toArray
        //a.foreach(print)
        //println("\\")
        //b.foreach(print)
        //println("\\")
        output(coor._1,coor._2) = computeMutualInformation(a,  num_state1,b, num_state2)
      }

    }

    output.mapPairs { (coor, x) =>
      if (coor._1 < coor._2) {
        output(coor._1,coor._2) = output(coor._2,coor._1)
      }

    }

    output
  }





}
