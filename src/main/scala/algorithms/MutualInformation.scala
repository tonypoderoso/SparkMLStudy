package algorithms



import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import breeze.numerics.abs
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRowMatrix, MatrixEntry, RowMatrix}

import scala.collection.Map
import scala.collection.immutable.IndexedSeq
import scala.collection.mutable.ListBuffer
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
    val mentry: RDD[MatrixEntry] = rdd.zipWithIndex.flatMap {
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number: Double, columnIndex: Int) => MatrixEntry(columnIndex, rowIndex, number)
      }
    }
    new CoordinateMatrix(mentry).toRowMatrix().rows
  }

  def rddTranspose2(rdd: RDD[Vector]): RDD[Vector] = {
    // Split the matrix into one number per line.
    val sc = rdd.sparkContext
    val mentry: RDD[MatrixEntry] = rdd.zipWithIndex.flatMap {
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number: Double, columnIndex: Int) => MatrixEntry(columnIndex, rowIndex, number)
      }
    }
    new CoordinateMatrix(mentry).toRowMatrix().rows
  }

  def featureFromDataset(inputs: RDD[LabeledPoint], normConst: Double): RDD[Vector] = {
    inputs.map { v =>
      val x: BDV[Double] = new BDV(v.features.toArray)
      val y: BDV[Double] = new BDV(Array(v.label))
      Vectors.dense(BDV.vertcat(x, y).toArray)
    }
  }

  def normalizeToUnitwithTranspose(inputs: RDD[LabeledPoint], normConst: Double): RDD[Vector] = {

    //val features: RDD[Vector] = inputs.map(_.features)
    val features: RDD[Vector] = inputs.map { v =>
      val x: BDV[Double] = new BDV(v.features.toArray)
      val y: BDV[Double] = new BDV(Array(v.label))
      Vectors.dense(BDV.vertcat(x, y).toArray)
    }
    val features1: RDD[Vector] = rddTranspose2(features)
    val normalized = (vec: Vector) => {
      val Max: Double = max(vec.toArray)
      val Min: Double = min(vec.toArray)
      val FinalMax: Double = max(abs(Max), abs(Min))
      Vectors.dense((BDV(vec.toArray) :/ BDV.fill(vec.toArray.length) {
        FinalMax
      }).toArray)
    }
    features1.map(normalized)
  }

  def normalizeToUnitT(inputs: RDD[Vector]): RDD[Vector] = {

    val normalized = (vec: Vector) => {
      val Max: Double = max(vec.toArray)
      val Min: Double = min(vec.toArray)
      val FinalMax: Double = max(abs(Max), abs(Min))
      Vectors.dense((BDV(vec.toArray) :/ BDV.fill(vec.toArray.length) {
        FinalMax
      }).toArray)
    }
    inputs.map(normalized)
  }


  def rddTransposeWithPartition(rdd: RDD[Vector]): RDD[Vector] = {
    // Split the matrix into one number per line.
    val byColumnAndRow: RDD[(Int, (Long, Double))] = rdd.zipWithIndex.flatMap {
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number: Double, columnIndex: Int) => {
          columnIndex ->(rowIndex, number)
        }
      }
    }.cache()
    // Build up the transposed matrix. Group and sort by column index first.
    val byColumn: RDD[Iterable[(Long, Double)]] = byColumnAndRow.groupByKey().sortByKey().values
    // Then sort by row index.
    byColumn.map {
      indexedRow => Vectors.dense(indexedRow.toArray.sortBy(_._1).map(_._2))
    }
  }


  def rddTranspose(rdd: RDD[Vector]): RDD[Vector] = {
    // Split the matrix into one number per line.
    val byColumnAndRow: RDD[(Int, (Long, Double))] = rdd.zipWithIndex.flatMap {
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number: Double, columnIndex: Int) => {
          columnIndex ->(rowIndex, number)
        }
      }
    }
    // Build up the transposed matrix. Group and sort by column index first.
    val byColumn: RDD[Iterable[(Long, Double)]] = byColumnAndRow.groupByKey.sortByKey().values
    // Then sort by row index.
    byColumn.map {
      indexedRow => Vectors.dense(indexedRow.toArray.sortBy(_._1).map(_._2))
    }
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

  def discretizeVector1(input: RDD[Vector], level: Int): RDD[Array[Int]] = {

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

  def computeMutualInformation(vec1: Array[Int], num_state1: Int, vec2: Array[Int], num_state2: Int): Double = {
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


  def computeMIMatrixRDD2(in: RDD[Array[Int]], num_features: Int, num_state1: Int,
                          num_state2: Int, num_new_partitions: Int)= {

    val sc = in.sparkContext

    val a: RDD[(Array[Int], Long)] = in.zipWithIndex().cache()
    val a1=in.zipWithIndex().cache()
    val buf = new ListBuffer[((Long, Long), Double)]()
    a.foreach{ b =>
      //val b: (Array[Int], Long) = a.first()
      val onerow = sc.broadcast(b)
      val computeMutualInformation2: ((Array[Int], Long)) => ListBuffer[((Long, Long), Double)]
      = (input: ((Array[Int], Long))) => {

        val recrow: (Array[Int], Long) = onerow.value
        val col = recrow._2
        val row = input._2
        var res: MatrixEntry = MatrixEntry(-1, -1, -1)
        if (row >= col) {
          val output = BDM.zeros[Int](num_state1, num_state2)
          val rsum: BDV[Int] = BDV.zeros[Int](num_state1)
          val csum: BDV[Int] = BDV.zeros[Int](num_state2)
          val msum: Int = recrow._1.length
          recrow._1.zip(input._1).map { x =>
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
          val tmp = MI / msum

          buf += (((row, col), tmp))  //MatrixEntry(row, col, tmp)
        }
        buf
      }
      a1.flatMap(computeMutualInformation2)
    }
    //reduce(_.union(_))
    buf//reduce(_.union(_))
  }


    def computeMIMatrixRDD1(in: RDD[Array[Int]], num_features: Int, num_state1: Int,
                            num_state2: Int, num_new_partitions: Int)= {

      val sc = in.sparkContext




      val computeMutualInformation1 = (input: ((Array[Int], Long), (Array[Int], Long))) => {


        val row = input._1._2
        val col = input._2._2
        var res:((Long,(Long,Double)))= ((-1,(-1,-1.0)))//: MatrixEntry = MatrixEntry(-1, -1, -1)
        if (row > col) {

          val output = BDM.zeros[Int](num_state1, num_state2)
          val rsum: BDV[Int] = BDV.zeros[Int](num_state1)
          val csum: BDV[Int] = BDV.zeros[Int](num_state2)
          val msum: Int = input._1._1.length

          input._1._1.zip(input._2._1).map { x =>
            output(x._1-1, x._2-1) = output(x._1-1, x._2-1) + 1
            rsum(x._1-1) = rsum(x._1-1) + 1
            csum(x._2-1) = csum(x._2-1) + 1
          }

          val MI = output.mapPairs { (coo, x) =>
            if (x > 0) {
              val tmp = msum.toDouble / (rsum(coo._1) * csum(coo._2))
              x * math.log(x * tmp) / math.log(2)
            } else
              0
          }.toArray.reduce(_ + _)

          val tmp = MI / msum
          res = ((row,(col,tmp))) //MatrixEntry(row, col, tmp)
        }
        res
      }

      val a: RDD[(Array[Int], Long)] = in.zipWithIndex().cache()
      //val b=in.zipWithIndex()
      //Method 1
      //a.cartesian(a).repartition(num_new_partitions).map(computeMutualInformation1)
      a.cartesian(a).map(computeMutualInformation1)
      //Method 2
      /*
    //val b: Map[Array[Int], Long] =a.collectAsMap

    sc.union(
    b.map{ x =>
      val belem = sc.broadcast(x)
      //a.mapPartitions { ys =>
        a.map { y =>
          //val tmp: (Array[Int], Long) =y
          //val tmp2: (Array[Int], Long) =belem.val\
          //(y,belem.value)computeMutualInformation1)
          computeMutualInformation1(y, belem.value)
      //  }
      }
    }.toSeq)*/

    }

    def computeMIMatrixRDD(input: RDD[Array[Int]], num_features: Int, num_state1: Int, num_state2: Int)
    : RDD[MatrixEntry] = {

      val sc = input.sparkContext

      val in: Array[Array[Int]] = input.collect()

      sc.parallelize(
        (0 until num_features).flatMap { row =>
          (row until num_features).flatMap { col =>
            //println("Computhing " +row.toString+ "-th row  and "+ col.toString + "-th column")
            val a: Array[Int] = in(row) //indexKey.lookup(row).flatten.toArray
          val b: Array[Int] = in(col) //indexKey.lookup(col).flatten.toArray
          val tmp: Double = computeMutualInformation(a, num_state1, b, num_state2)
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

    def computeMIMatrix(input: RDD[Array[Int]], num_features: Int, num_state1: Int, num_state2: Int): BDM[Double] = {
      val output = BDM.zeros[Double](num_features, num_features)

      val indexKey: RDD[(Long, Array[Int])] = input.zipWithIndex().map { x => (x._2, x._1) }

      indexKey.cache()

      output.mapPairs { (coor, x) =>
        if (coor._1 >= coor._2) {

          val a: Array[Int] = indexKey.lookup(coor._1).flatten.toArray
          val b: Array[Int] = indexKey.lookup(coor._2).flatten.toArray
          //a.foreach(print)
          //println("\\")
          //b.foreach(print)
          //println("\\")
          output(coor._1, coor._2) = computeMutualInformation(a, num_state1, b, num_state2)
        }

      }

      output.mapPairs { (coor, x) =>
        if (coor._1 < coor._2) {
          output(coor._1, coor._2) = output(coor._2, coor._1)
        }

      }

      output
    }



}
