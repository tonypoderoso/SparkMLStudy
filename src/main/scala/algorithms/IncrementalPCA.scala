package algorithms

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{Matrix, Vector, _}
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}

import scala.collection.immutable.IndexedSeq

/**
  * Created by tonyp on 2016-04-27.
  */
class IncrementalPCA {

  def ArrayVectorRowMatrixConcatenate(rm1:Array[Vector],rm2:RowMatrix): RDD[Vector] = {
    //if (rm1.length == rm2.numRows()){
    rm2.rows.zipWithIndex().map { x =>
      Vectors.dense(BDV.vertcat(BDV(rm1(x._2.toInt).toArray),BDV(x._1.toArray)).toArray)
    }
  }

  def RowMatrixTranspose(rm : RowMatrix): RowMatrix = {

    val larray: RDD[Vector] = rm.rows
    val tmp: RDD[MatrixEntry] = larray.zipWithIndex.flatMap { case (vec, row) =>
      vec.toArray.zipWithIndex.map { case (entry, col) => MatrixEntry(col, row, entry)
      }
    }
    new CoordinateMatrix(tmp, rm.numCols, rm.numRows()).toRowMatrix()
  }

  def RowMatrixPrint(rm : RowMatrix,str : String)={

    println("*********************************************************")
    println(str)
    println("*********************************************************")
    val tmp: Array[Vector] =rm.rows.collect()
    println(" The Matrix is of size with rows : "+ tmp.length + " and cols : "+ tmp(0).toArray.length)
    tmp.zipWithIndex.foreach { case (vect, index) =>
        print( "\nRow " + index.toString + " : ")
        vect.toArray.foreach(x  =>  print( x.toString.substring(0,5)+ " , " ))
    }
    println("*********************************************************")
  }

  def RowMatrixMultiply(rm1:RowMatrix,rm2:RowMatrix): RowMatrix = {
    println("Inside RowMatrixMultiply")
    rm2.rows.first().toArray.foreach(print)

    val rm22: Matrix = Matrices.dense( rm2.numRows().toInt,rm2.numCols().toInt, rm2.rows.flatMap{ v =>v.toArray}.toArray())
    println("")
    rm22.toArray.foreach(print)
    rm1.multiply(rm22)
  }

  def rddTranspose(rdd: RDD[Vector]): RDD[Vector] = {
    // Split the matrix into one number per line.
    val byColumnAndRow: RDD[(Int, (Long, Double))] = rdd.zipWithIndex.flatMap{
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number: Double, columnIndex: Int) => {columnIndex.->(rowIndex, number)}
      }
    }
    // Build up the transposed matrix. Group and sort by column index first.
    val byColumn: RDD[Iterable[(Long, Double)]] = byColumnAndRow.groupByKey.sortByKey().values

    byColumn.map {
      indexedRow =>
        val ttmmpp: Array[Double] =indexedRow.toArray.sortBy(_._1).map(_._2)
        Vectors.dense(ttmmpp)
    }
  }

  def fit(data : RDD[Vector], U0 :RDD[Vector] , D0 : Array[Double]): Unit ={
    /*val n=data.rows.first().toArray.length
    val n0=data.numCols()
    val ff=1.0
    val mu1: RDD[Double] = data.rows.map(x=>sum(x.toArray)/n)
    val ndata: RDD[Array[Double]] = data.rows.map{ x =>
        val y: Double = sum(x.toArray)/n
        x.toArray.map(z=>z/y)
    }
    */
    val sc = data.sparkContext;


    val data_collect: Array[Vector] = rddTranspose(data).collect()

    val data_prj: RowMatrix = new CoordinateMatrix(sc.parallelize(
      data_collect.zipWithIndex.map { case (v1, ind1) =>
      rddTranspose(U0).zipWithIndex.map{ case (v2, ind2) =>
        MatrixEntry(ind1, ind2, BDV(v1.toArray).t * BDV(v2.toArray))
      }.toLocalIterator
    }.toSeq.flatten)).toRowMatrix()

    RowMatrixPrint(data_prj," Computing data_projection")


    //uodata_prj = U0*data_proj;
    val uo: Array[Vector] =U0.collect()
    val uodata_proj: RowMatrix = new CoordinateMatrix(sc.parallelize(
      data_prj.rows.zipWithIndex.map{ case (v1,ind1) =>
        uo.zipWithIndex.map{case (v2,ind2)=>
        MatrixEntry(ind1,ind2,BDV(v1.toArray).t * BDV(v2.toArray))
    }.toSeq
    }.toLocalIterator.flatten.toIndexedSeq)).toRowMatrix()

    RowMatrixPrint(uodata_proj, "Computing U0 multiplied by data projection ")


    //data_res = data - data_prj;
    val data_res: RowMatrix = new RowMatrix(uodata_proj.rows.zipWithIndex.map{ x =>
      val tmp = (BDV(data_collect(x._2.toInt).toArray) - BDV(x._1.toArray)).toArray
      Vectors.dense(tmp)
    })

    RowMatrixPrint(data_res, "Computing data result")

    //[q, dummy] = qr(data_res, 0);
    val result: QRDecomposition[RowMatrix, Matrix] =
      RowMatrixTranspose(data_res).tallSkinnyQR(computeQ =true)

    RowMatrixPrint(result.Q," The Q Matrix")

    //Q = [U0 q];



    val Q: RDD[Vector] = ArrayVectorRowMatrixConcatenate(uo, result.Q)


    RowMatrixPrint(new RowMatrix(Q) , " The Concatenated Vector")


    // top = ff*diag(D0) data_proj;
    //bottom =zeros([size(data,2) length(D0)]) q'*data_res]
    //R = [top ; bottom ];

    val diag0: Array[Vector] = (0 until D0.length ).map{ row=>
       val tmp: BDV[Double] = BDV.zeros[Double](D0.length)
       tmp(row)=D0(row)
       Vectors.dense(tmp.toArray)
    }.toArray

    val top: Array[Vector] = ArrayVectorRowMatrixConcatenate(diag0,data_prj).toArray

    println(" The size to top part is : " +top.length +" cols: "+ top.take(0).length)

    val bottomleft: Array[Vector] = (1 to data.first.toArray.length).map { i =>
      Vectors.dense( BDV.zeros[Double](D0.length).toArray)
    }.toArray


    val bottomright: RowMatrix = RowMatrixMultiply(data_res,result.Q)

    RowMatrixPrint(bottomright, " The bottom rigth part ")

    val bottom: Array[Vector] = ArrayVectorRowMatrixConcatenate(bottomleft,bottomright).toArray

    println(" The size to top part is : " +bottom.length +" cols: "+ bottom.take(0).length)
    val R: Array[Vector] = Array(top,bottom).flatten

    println("The rowsize of top is : " + top.length + "and the bottom is : " + bottom.length + " and the rowsize of R is : "+R.length)

    //    [U,D,V] = svd(R, 0);
    //   D = diag(D);
    val RR: RowMatrix =new RowMatrix(sc.parallelize(R))

    RowMatrixPrint(RR, "THe R computed")

    //val svd: SingularValueDecomposition[RowMatrix, Matrix] = RR.computeSVD(R.length, computeU = true)



    //print(" THe diagonal entries are: " )
    //svd.s.toArray.map(x=>x.toString+ ",")
    //println(" ")

    //println("The size of U matrix is : " + svd.U.numRows +", "+ svd.U.numCols)
    ///RowMatrixPrint( svd.U , " The U Marix")
    //RowMatrixPrint(new RowMatrix(sc.parallelize(svd.V))," The V matrix")





    //diagonal matrix
    //println("Make a diagonal matrix")
    //val diagentry: IndexedSeq[MatrixEntry] = (0 until D0.length ).map{ row=>
     // MatrixEntry(row,row,D0(row))
    //}
    //val DD: CoordinateMatrix = new CoordinateMatrix(sc.parallelize(diagentry))



    //val D = diag(D0)
    //val data_proj= data.zipWithIndex.map{case (x,y)=>
    //    data

    }
    //val res = data - U0.t*data_proj.t


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