package algorithms

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{Matrix, Vector,Vectors,Matrices,_}
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}


/**
  * Created by tonyp on 2016-04-27.
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
          ff: Double= 1.0, K: Array[Int] = Array(-1)): (RowMatrix, Array[Double]) = {

    val sc = data_in.sparkContext;
    val datain_collect: Array[Vector] = data_in.collect()
    val N = datain_collect.length.toDouble
    var n = datain_collect(0).toArray.length.toDouble
    var data:RDD[Vector] = null

    if (U0 == null) {

    } else {
      if (n0(0) == -1.0) {
        n0(0) = n
        println("inside n0")
      }
      if (D0 != null && mu0 != null) {
          println("data transformation")
          val mu1: Array[Double] = datain_collect.map { x => x.toArray.reduce(_ + _) / n }
          println(" The mu1 vector : " + mu1.map { i => i.toString + "," }.reduce(_ + _))
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
          val mu: BDV[Double] = (a1 + a2) / n
          println(" The mu vector : " + mu.map { i => i.toString + "," }.reduce(_ + _))

      }
    }

    val data_collect = data.collect()
    printArrayVector(data_collect, "The modified data")
      //RowMatrixPrint(RowMatrixTranspose(new RowMatrix(data)), "The input data")
      //RowMatrixPrint(RowMatrixTranspose(new RowMatrix(U0)), " The U0 Matrix")

      /* val data_prj: RowMatrix = new CoordinateMatrix(sc.parallelize(
      data_collect.zipWithIndex.map { case (v1, ind1) =>
      rddTranspose(U0).zipWithIndex.map{ case (v2, ind2) =>
        MatrixEntry(ind1, ind2, BDV(v1.toArray).t * BDV(v2.toArray))
      }.toLocalIterator
    }.toSeq.flatten)).toRowMatrix() */

      val data_prj: RowMatrix = RowMatrixMultiply(new RowMatrix(rddTranspose(U0)), new RowMatrix(data))

      //RowMatrixPrint(data_prj, " Computing data_projection")

      //RowMatrixPrint(RowMatrixTranspose(data_prj), "Data Projection Transopsed")

      //uodata_prj = U0*data_proj;

      /*val uo: Array[Vector] = U0.collect
    val uodata_proj: RowMatrix = new CoordinateMatrix(sc.parallelize(
      data_prj.rows.zipWithIndex.map{ case (v1,ind1) =>
        uo.zipWithIndex.map{case (v2,ind2)=>
        MatrixEntry(ind1,ind2,BDV(v1.toArray).t * BDV(v2.toArray))
    }.toSeq
    }.toLocalIterator.flatten.toIndexedSeq)).toRowMatrix()  */

      val uodata_proj: RowMatrix = RowMatrixMultiply(new RowMatrix(U0), data_prj)
      //RowMatrixPrint(uodata_proj, "Computing U0 multiplied by data projection ")
      println("data_collect : row:" + data_collect.length + " col: " + data_collect(0).toArray.length)
      println("uodata_proj : row:" + uodata_proj.numRows + " col: " + uodata_proj.numCols)


      //data_res = data - data_prj;
      var i = -1;
      val data_res_rdd: Array[Vector] = uodata_proj.rows.collect.map { x =>
        i = i + 1
        Vectors.dense(data_collect(i).toArray - x.toArray)
      }



      val data_res: RowMatrix = new RowMatrix(sc.parallelize(data_res_rdd))

      println("data_res : row:" + data_res.numRows + " col: " + data_res.numCols)
      println("data : row:" + data_collect.length + " col: " + data_collect(0).toArray.length)

     // RowMatrixPrint(data_res, " The data result")

      val data_res_t = RowMatrixTranspose(data_res)
      //RowMatrixPrint(data_res_t, "Computing data result transposed")
      //[q, dummy] = qr(data_res, 0);
      val result: QRDecomposition[RowMatrix, Matrix] =
        data_res.tallSkinnyQR(true)

      //RowMatrixPrint(RowMatrixTranspose(result.Q), " The Q Matrix")

      //Q = [U0 q];

      val Q: RDD[Vector] = ArrayVectorRowMatrixConcatenate(U0.collect, result.Q)


      //RowMatrixPrint(new RowMatrix(Q), " The Concatenated Vector")


      // top = ff*diag(D0) data_proj;
      //bottom =zeros([size(data,2) length(D0)]) q'*data_res]
      //R = [top ; bottom ];


      val diag0: Array[Vector] = (0 until D0.length).map { row =>
        val tmp: BDV[Double] = BDV.zeros[Double](D0.length)
        tmp(row) = D0(row)
        Vectors.dense(tmp.toArray)
      }.toArray

      //printArrayVector(diag0, "The diagonal Matrix")



      println("The size of diag0 is ==> Row: " + diag0.length + " Col : " + diag0(0).toArray.length)
      println("the size of data_prj is ==> Row: " + data_prj.numRows + " Col : " + data_prj.numCols)

      //RowMatrixPrint(RowMatrixTranspose(data_prj), " The transpose of data_proj")

      val top: Array[Vector] = ArrayVectorRowMatrixConcatenate(diag0, data_prj).collect

      //printArrayVector(top, "The Top ArrayVector")

      //println(" The size to top part is : " +top.length +" cols: "+ top(0).toArray.length)

      //RowMatrixPrint(new RowMatrix(top.toList), "The Top Matrix")


      val bottomleft: Array[Vector] = (1 to data.first.toArray.length).map { i =>
        Vectors.dense(BDV.zeros[Double](D0.length).toArray)
      }.toArray

      val bottomright: RowMatrix = RowMatrixMultiply(RowMatrixTranspose(result.Q), data_res)

      //RowMatrixPrint(bottomright, " The bottom rigth part ")

      val bottom = ArrayVectorRowMatrixConcatenate(bottomleft, bottomright).collect

      //println(" The size to top part is : " +bottom.length +" cols: "+ bottom(0).toArray.length)

      //printArrayVector(bottom, " The Bottom ArrayVector")

      //RowMatrixPrint(new RowMatrix(bottom), "The Bottom Matrix")

      val R: Array[Vector] = Array.concat(top, bottom).toSeq.toArray

      //printArrayVector(R, " The Array R ")


      //println("The rowsize of top is : " + top.length + "and the bottom is : " + bottom.length + " and the rowsize of R is : "+R.length)

      //    [U,D,V] = svd(R, 0);
      //   D = diag(D);
      R.foreach { x => x.toArray.foreach(print)
        println("")
      }
      val RR = new RowMatrix(sc.parallelize(R))

      //RowMatrixPrint(RR, "THe R computed")

      val svd: SingularValueDecomposition[RowMatrix, Matrix] = RR.computeSVD(R.length, computeU = true)

      print(" THe diagonal entries are: ")
     // println(svd.s.toArray.map(x => x.toString + ",").reduce(_ + _))
      println(" ")


      println("The size of U matrix is : " + svd.U.numRows + ", " + svd.U.numCols)
      //RowMatrixPrint(svd.U, " The U Marix")
      //RowMatrixPrint(new RowMatrix(sc.parallelize(svd.V))," The V matrix")

      val Keep = 16

      val U: RowMatrix = takeColsFrom0To(RowMatrixMultiply(new RowMatrix(Q), svd.U), Keep)

      RowMatrixPrint(U, " The final U Marix")

      val D: Array[Double] = svd.s.toArray.take(Keep)

      (U, D)





      //diagonal matrix
      //println("Make a diagonal matrix")
      //val diagentry: IndexedSeq[MatrixEntry] = (0 until D0.length ).map{ row=>
      // MatrixEntry(row,row,D0(row))
      //}
      //val DD: CoordinateMatrix = new CoordinateMatrix(sc.parallelize(diagentry))


      //val D = diag(D0)
      //val data_proj= data.zipWithIndex.map{case (x,y)=>
      //    data


    //val res = data - U0.t*data_proj.t


  }

}

//val uot: Array[Vector] =rddTranspose(U0).collect()
//println("Number of columns of UO : "+ uot(0).toArray.length)
// println("Number of rows of UO : " +uot.length)

//val data_prj_tmp: IndexedSeq[MatrixEntry] = rddTranspose(data).zipWithIndex.map { case (v1, ind1) =>
//  uot.zipWithIndex.map{ case (v2, ind2) =>
//    MatrixEntry(ind1, ind2, BDV(v1.toArray).t * BDV(v2.toArray))
//  }.toSeq
//}.toLocalIterator.flatten.toIndexedSeq
//val data_prj: RowMatrix = new CoordinateMatrix(sc.parallelize( data_prj_tmp))
//  .toRowMatrix()