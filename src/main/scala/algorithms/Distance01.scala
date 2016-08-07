package algorithms


import breeze.linalg.{Axis, min, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{AccumulatorParam, FutureAction, SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer
import preprocessing._

import scala.collection.immutable.{IndexedSeq, Range}




/**
  * Created by tonypark on 2016. 6. 26..
  */
object Distance01 {

  def computeMeanShiftedMatrix(r:RDD[LabeledPoint],N: Int): RDD[Array[Float]] =
  {
    val colMean= r
      .map( x => x.features.toArray )
      .reduce( _ + _ )
      .map( x => x / N )

    r.map( x => ( x.features.toArray - colMean ).map(elem => elem.toFloat))

  }

  def computeDifference(r:RDD[LabeledPoint],N: Int): RDD[Array[Float]] =
  {
    val colMean= r
      .map( x => x.features.toArray )
      .reduce( _ + _ )
      .map( x => x / N )

    r.map( x => ( x.features.toArray - colMean ).map(elem => elem.toFloat))

  }


  def exPartitionMap(MeanShiftedMat:RDD[Array[Float]],num_partitions:Int,num_features:Int)
  : RDD[(Int, BDM[Float])] = {
    val Buff1 = new ListBuffer[(Int, BDM[Float])]
    val rownum= num_features/num_partitions

    MeanShiftedMat
      .zipWithIndex
      .keyBy(_._2)
      .partitionBy(new MyPartitioner( num_partitions, num_features))
      .mapPartitionsWithIndex({ (indx, itr) => {

        val bbb: BDM[Float] =BDM.zeros[Float](rownum,num_features)
        itr.foreach{ case (row , arr)=>
          //println("The itr index:" + row + " The other index : " + arr._2 + "The matrix index : " + row%rownum)
          val ccc: BDV[Float] = BDV(arr._1)
          bbb((row%rownum).toInt,::) := ccc.t
        }
        Buff1 += ((indx, bbb))
      }.toIterator
      } , true)
  }


  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)

    val sc = new SparkContext(new SparkConf())//.setMaster("local[*]").setAppName("PCAExampleTest"))

    var num_features: Int = 1000
    var num_samples: Int = 1000
    var num_bins: Int = 5
    var num_new_partitions: Int = 100
    var num_partsum:Int = 10

    if (!args.isEmpty) {

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt
      num_partsum=args(4).toString.toInt

    }

    val recordsPerPartition: Int = num_samples / num_new_partitions
    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitionsWithIndex { (idx, iter) => {
        val gauss: Gaussian = new Gaussian(0.0, 1.0)
        (0 until recordsPerPartition).map { i =>
          new LabeledPoint(idx*recordsPerPartition+i, Vectors.dense(gauss.sample(num_features).toArray))
        }
      }.toIterator
      }.cache()


    val start = System.currentTimeMillis()

    //val msm: RDD[Array[Float]] = computeMeanShiftedMatrix(lds, num_samples)
    val msm: RDD[Array[Float]] = lds.map{ elem =>
      val aaa: Array[Float] =elem.features.toArray.map{ a=>
        a.toFloat}
      aaa}

    val cc: RDD[(Int, BDM[Float])] =
      exPartitionMap(msm, num_new_partitions, num_features).cache()

    var dd: Seq[(Int, BDM[Float])] = cc.collectAsync().get().sortBy(x=>x._1)

    if (num_partsum>1) {
      dd = (0 until dd.length by num_partsum).par.map { i =>
        val tmp: BDM[Float] = (0 until num_partsum).map{ j =>
          dd(i + j)._2
        }.reduce(BDM.vertcat(_, _))
        (i + num_partsum-1  ,tmp )
      }.toList
    }


    def distcompute1(a:BDM[Float],b:BDM[Float]) = {
      (0 until a.rows).par.map {i=>
        val tmp = b  - BDM.tabulate[Float](b.rows,a.cols){
          case (_,j) => a(i,j)}
        sum(tmp :* tmp,Axis._1).toDenseMatrix
      }.reduce((x,y) =>
        BDM.vertcat(x,y))
    }

    def distcompute2(a:BDM[Float],b:BDM[Float]): BDM[Float] = {
      val tmp=BDM.tabulate[Float](b.rows*a.rows,a.cols){
        case(i,j)=>a(i/b.rows,j)-b(i%b.rows,j)
      }
      sum(tmp :* tmp,Axis._1).toDenseMatrix.t.reshape(a.rows,b.rows)
    }
    def distcompute3(a:BDM[Float],b:BDM[Float]) = {
      (0 until a.rows).par.map {i=>
        val tmp: BDM[Float] = (b  :- a(i,::))
        val res: BDV[Float] =sum(tmp :* tmp,Axis._1)
        res.toDenseMatrix.t
      }.reduce((x,y) =>
        BDM.vertcat(x,y))
    }
    val res: List[(Int, BDM[Float])] = dd.par.flatMap { case (part1: Int, seq1: BDM[Float]) => {
      val bro: Broadcast[(BDM[Float], Int)] = sc.broadcast(seq1, part1)
      val Buff1 = new ListBuffer[(Int, BDM[Float])]

      val block: (Int, BDM[Float]) = cc.map{
        case (part2: Int, seq2: BDM[Float]) =>
          if (bro.value._2 >= part2) {
            (part2,distcompute1(bro.value._1,seq2))
          } else {
            (part2,BDM.zeros[Float](bro.value._1.rows,1))
          }
      }.collectAsync.get.sortBy(_._1)
        .reduce { (a, b) =>
        (0, BDM.horzcat(a._2, b._2))
      }
      Buff1 += ((part1, block._2))
    }.toIterator
    }.toList

    println("Elased Time in seconds : "+(System.currentTimeMillis - start)/1000.0)

    println("*******************************************\n")
    println(" Number of features : " + num_features)
    println(" Number of partitions : " + num_new_partitions)

    sc.getConf.getAll.foreach(println)
    sc.stop()

  }

}

