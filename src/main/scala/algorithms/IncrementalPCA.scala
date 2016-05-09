package algorithms

import breeze.linalg.{ DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors, _}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}


/**
  * Created by tonyp on 2016-05-04.
  */
class IncrementalPCA {

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

  def RowMatrixMultiply(rm1: RowMatrix, rm2: RowMatrix): RowMatrix = {
    println("Inside RowMatrixMultiply")
    /*rm2.rows.first().toArray.foreach(print)

    val tmp: Array[Double] = rm2.rows.collect.map{ v=>v.toArray}.flatten

    val rm22: Matrix = Matrices.dense(rm2.numRows().toInt,rm2.numCols().toInt,tmp)
    println("")

    (0 until rm2.numRows.toInt).foreach { row => (0 until rm2.numCols.toInt).foreach{ col=>
      print( rm22(row,col).toString + " , ")}
      println("")
    }*/

    val rm22: Matrix = Matrices.dense(rm2.numRows.toInt, rm2.numCols.toInt, toBreeze(rm2).toArray)
    rm1.multiply(rm22)
  }

  def printArrayVector(arr: Array[Vector], str: String) = {

    println("*********************************************************")
    println(str)
    println("*********************************************************")

    arr.foreach { x =>
      println(x.toArray.toList) //.foreach(y => print(y.toString.substring(0, 4) + ","))
      //println("")
    }
    println("*********************************************************")
  }

  def ArrayVectorRowMatrixConcatenate(rm1: Array[Vector], rm2: RowMatrix): RDD[Vector] = {
    //if (rm1.length == rm2.numRows()){
    rm2.rows.zipWithIndex().map { x =>
      val tmp: BDV[Double] = BDV.vertcat(BDV(rm1(x._2.toInt).toArray), BDV(x._1.toArray))
      // tmp.foreach(print)
      // println("")
      Vectors.dense(tmp.toArray.toSeq.toArray)
    }
  }

  def RowMatrixTranspose(rm: RowMatrix): RowMatrix = {

    val res: RDD[Vector] = rddTranspose(rm.rows)
    new RowMatrix(res)
    //val tmp: RDD[MatrixEntry] = rm.rows.zipWithIndex.flatMap{ case (vec, row) =>
    //  vec.toArray.zipWithIndex.map { case (entry, col) => MatrixEntry(col, row, entry)
    //  }
    //}
    //new CoordinateMatrix(tmp, rm.numCols, rm.numRows()).toRowMatrix()
  }

  def RowMatrixPrint(rm: RowMatrix, str: String) = {

    println("*********************************************************")
    println(str)
    println("*********************************************************")
    val row = rm.numRows.toInt
    val col = rm.numCols.toInt
    val tmp = toBreeze(rm)
    println(" The Matrix is of size with rows : " + row.toString + " and cols : " + col.toString)

    (0 until row).foreach { row =>
      (0 until col).foreach { col =>
        //print(tmp(row, col).toString.substring(0,min(7,tmp(row,col).toString.length)) + ", ")
        print(tmp(row, col).toString + ", ")
      }
      println(" ; ")
    }
    println("*********************************************************")
  }

  /*def RowMatrixPrint(rm : RowMatrix,str : String)={

    println("*********************************************************")
    println(str)
    println("*********************************************************")
    val tmp  =rm.rows.collect()
    println(" The Matrix is of size with rows : "+ rm.numRows.toString + " and cols : "+ rm.numRows.toString)
    tmp.zipWithIndex.foreach { case (vect, index) =>
        print( "\nRow " + index.toString + " : ")
        vect.toArray.foreach(x  =>  print( x.toString.substring(0,min(x.toString.length,7))+ " , " ))
    }
    println("*********************************************************")
  }*/

  def takeColsFrom0To(rm: RowMatrix, k: Int): RowMatrix = {
    val sc = rm.rows.sparkContext
    val rmt = RowMatrixTranspose(rm)
    val tmp: Array[Vector] = rmt.rows.collect.take(k)
    RowMatrixTranspose(new RowMatrix(sc.parallelize(tmp)))
  }


  def rddTranspose(rdd: RDD[Vector]): RDD[Vector] = {
    // Split the matrix into one number per line.
    val byColumnAndRow: RDD[(Int, (Long, Double))] = rdd.zipWithIndex.flatMap {
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number: Double, columnIndex: Int) => {
          columnIndex.->(rowIndex, number)
        }
      }
    }
    // Build up the transposed matrix. Group and sort by column index first.
    val byColumn: RDD[Iterable[(Long, Double)]] = byColumnAndRow.groupByKey.sortByKey().values

    byColumn.map {
      indexedRow =>
        val ttmmpp: Array[Double] = indexedRow.toArray.sortBy(_._1).map(_._2)
        Vectors.dense(ttmmpp)
    }
  }



  def fit(data_in: RDD[Vector], U0: RDD[Vector] = null, D0: Array[Double] = null,
          mu0: Array[Double] = null, n0 : Array[Double] = Array(-1.0),
          ff: Double= 1.0, K: Array[Int] = Array(-1)):
  (RDD[Vector], Array[Double], Array[Double], Double) = {

    val sc = data_in.sparkContext
    val datain_collect: Array[Vector] = data_in.collect()
    val N = datain_collect.length.toDouble
    var n = datain_collect(0).toArray.length.toDouble
    var data:RDD[Vector] = null
    var keep = 0
    //var mu:Array[Double]=null

    if (U0 == null) {
      val mu: Array[Double] = datain_collect.map { x => sum(x.toArray) / n }
      data = sc.parallelize(datain_collect.zipWithIndex.map { x =>
        val ttt: Array[Double] = x._1.toArray.map(i => i - mu(x._2))
        Vectors.dense(ttt.toArray)
      })
      val data_row = new RowMatrix(data)
      val svd: SingularValueDecomposition[RowMatrix, Matrix] =
        data_row.computeSVD(data_row.numCols.toInt, computeU = true)

      val D: Array[Double] = svd.s.toArray
      var res=(svd.U.rows, D,mu,n)


      /* nargin >= 7
      keep = 1:min(K,length(D));
      D = D(keep);
      U = U(:,keep);
      end*/

      if (K(0) != -1){
        keep = min(K(0), D.length)
        val U: RowMatrix = takeColsFrom0To(svd.U, keep)
        val DD: Array[Double] = svd.s.toArray.take(keep)
        res = (U.rows,DD,mu,n)
      }

      res

    }
    else {


      var mu:Array[Double]=Array(-1)
      if (n0(0) == -1.0) {
        n0(0) = n
      }
      if (D0 != null && mu0 != null) {
        val mu1: Array[Double] = datain_collect.map { x => x.toArray.reduce(_ + _) / n }
        val cnt = sqrt(n * n0(0) / (n + n0(0)))
        val mu_new: Array[Double] = mu0.zipWithIndex.map { i => cnt * (i._1 - mu1(i._2)) }
        data = sc.parallelize(datain_collect.zipWithIndex.map { x =>
          val ttt: Array[Double] = x._1.toArray.map(i => i - mu1(x._2))
          val tttt: BDV[Double] = BDV.vertcat(BDV(ttt), BDV(mu_new(x._2)))
          Vectors.dense(tttt.toArray)
        })

        val a1: BDV[Double] = n0(0) * ff * BDV(mu0)
        val a2: BDV[Double] = n * BDV(mu1)
        n = n + ff * n0(0)
        mu = ((a1 + a2) / n).toArray
      }


      val data_collect = data.collect()
      val data_prj: RowMatrix = RowMatrixMultiply(new RowMatrix(rddTranspose(U0)), new RowMatrix(data))
      val uodata_proj: RowMatrix = RowMatrixMultiply(new RowMatrix(U0), data_prj)

      //data_res = data - data_prj;
      var i = -1
      val data_res_rdd: Array[Vector] = uodata_proj.rows.collect.map { x =>
        i = i + 1
        Vectors.dense(data_collect(i).toArray - x.toArray)
      }

      val data_res: RowMatrix = new RowMatrix(sc.parallelize(data_res_rdd))
    //  val data_res_t = RowMatrixTranspose(data_res)
      val result: QRDecomposition[RowMatrix, Matrix] =
        data_res.tallSkinnyQR(true)
      //Q = [U0 q];
       val Q: RDD[Vector] = ArrayVectorRowMatrixConcatenate(U0.collect, result.Q)

      // top = ff*diag(D0) data_proj;
      //bottom =zeros([size(data,2) length(D0)]) q'*data_res]
      //R = [top ; bottom ];

      val diag0: Array[Vector] = (0 until D0.length).map { row =>
        val tmp: BDV[Double] = BDV.zeros[Double](D0.length)
        tmp(row) = D0(row)
        Vectors.dense(tmp.toArray)
      }.toArray

      val top: Array[Vector] = ArrayVectorRowMatrixConcatenate(diag0, data_prj).collect

      val bottomleft: Array[Vector] = (1 to data.first.toArray.length).map { i =>
        Vectors.dense(BDV.zeros[Double](D0.length).toArray)
      }.toArray

      val bottomright: RowMatrix = RowMatrixMultiply(RowMatrixTranspose(result.Q), data_res)

      val bottom = ArrayVectorRowMatrixConcatenate(bottomleft, bottomright).collect

      val R: Array[Vector] = Array.concat(top, bottom).toSeq.toArray

      //    [U,D,V] = svd(R, 0);
      //   D = diag(D);

      val RR = new RowMatrix(sc.parallelize(R))
      val svd: SingularValueDecomposition[RowMatrix, Matrix] = RR.computeSVD(R.length, computeU = true)

      /* if nargin < 7
      cutoff = sum(D.^2) * 1e-6;
      keep = find(D.^2 >= cutoff);
      else
      keep = 1:min([K,length(D),n]);
      end */
      //var keep = 0
      val DD = BDV(svd.s.toArray)
      if (K(0) == -1){
        val cutoff: Double = sum(DD :* DD) * 1e-6
        val sumsq: BDV[Int] = DD.map(i =>
          if (i*i >= cutoff) 1 else 0)
        keep = sum(sumsq)
      }else
        {
          keep = min(K(0),DD.length,n.toInt)
        }
      //val Keep = 16

      val U: RowMatrix = takeColsFrom0To(RowMatrixMultiply(new RowMatrix(Q), svd.U), keep)
      val D: Array[Double] = svd.s.toArray.take(keep)

      (U.rows, D,mu,n)


    }


  }

}

