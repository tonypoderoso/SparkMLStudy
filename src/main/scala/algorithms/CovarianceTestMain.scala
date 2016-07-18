package algorithms

import algorithms.common.PrintWrapper
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, cov => brzCov}
import preprocessing._
import breeze.stats.distributions.Gaussian
import breeze.storage.Storage
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{FutureAction, Partition, SparkConf, SparkContext}

import scala.collection.immutable.IndexedSeq
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
/**
  * Created by tonypark on 2016. 6. 7..
  *
  *
  */


object CovarianceTestMain {

  def computeCovarianceRDDNew(r:RDD[LabeledPoint],N: Int, P: Int,num_partitions:Int): RDD[(Int, Int, BDM[Double])] = {

    val colMean: Array[Double] = r
      .map( x => x.features.toArray )
      .reduce( _ + _ )
      .map( x => x / N )

    val MeanShiftedMat: RDD[Array[Double]] = r
      .map( x => ( x.features.toArray - colMean )  )


    // The number of partitions
    val K: Int = ( if (P < 10000) 2 else if (P < 40000) 4 else 10 )


    val blokRDD: RDD[(Long, Array[Double])] = MeanShiftedMat
      .zipWithIndex().map{case (v,k)=>(k,v)}
      .partitionBy(new MyPartitioner( num_partitions, P))



    val Buff1= new ListBuffer[(Int,Iterator[(Long,Array[Double])])]
    val res1: RDD[(Int, Iterator[(Long, Array[Double])])]
    = blokRDD.mapPartitionsWithIndex{ (indx, itr) =>
      {
        Buff1 += ((indx,itr))
      }.toIterator.take(num_partitions)
    }

    val Buff2= new ListBuffer[(Int,Iterator[(Long,Array[Double])])]
    val res2: RDD[(Int, Iterator[(Long, Array[Double])])]
    = blokRDD.mapPartitionsWithIndex{ (indx, itr) =>
      {
        Buff2 += ((indx,itr))
      }.toIterator.take(num_partitions)
    }

    var cnt = 0

    res1.cartesian(res2).flatMap{ case ((indx1, itr1), (indx2, itr2)) =>
      val Buf = new ListBuffer[(Int,Int, BDM[Double])]
      cnt += 1
      //println("Iter: " + cnt + " , "  + ", The index: "  +
      //  ", The partition index is: " + indx1 +"," + indx2 + "\n" )//+ aa)
    val aa: BDM[Double] = itr1.zip(itr2).map { case (x1, x2) =>
        BDV(x1._2) * BDV(x2._2).t
      }.reduce(_ :+ _)
      Buf .+= ((indx1, indx2, aa))
    }
  }

  def computeCovarianceRDDbyBlock(r:RDD[LabeledPoint],N: Int, P: Int,num_partitions:Int)
  = {

    val colMean: Array[Double] = r
      .map( x => x.features.toArray )
      .reduce( _ + _ )
      .map( x => x / N )

    val MeanShiftedMat: RDD[Array[Double]] = r
      .map( x => ( x.features.toArray - colMean )  )


    // The number of partitions
    val K: Int = ( if (P < 10000) 2 else if (P < 40000) 4 else 10 )


    val blokRDD: RDD[(Long, Array[Double])] = MeanShiftedMat
      .zipWithIndex().map{case (v,k)=>(k,v)}
      .partitionBy(new MyPartitioner( num_partitions, P))

    val Buff1= new ListBuffer[(Int,Iterator[(Long,Array[Double])])]
    val res1: RDD[(Int, Iterator[(Long, Array[Double])])]
    = blokRDD.mapPartitionsWithIndex{ (indx, itr) =>
      {
        Buff1 += ((indx,itr))
      }.toIterator.take(num_partitions)
    }

    val Buff2= new ListBuffer[(Int,Iterator[(Long,Array[Double])])]
    val res2: RDD[(Int, Iterator[(Long, Array[Double])])]
    = blokRDD.mapPartitionsWithIndex{ (indx, itr) =>
      {
        Buff2 += ((indx,itr))
      }.toIterator.take(num_partitions)
    }


    res1.cartesian(res2).flatMap{ case ((indx1, itr1), (indx2, itr2)) =>
      val Buf = new ListBuffer[(Int,Int, BDM[Float])]
      val elemPerPartition: Int =P/num_partitions
      val resultMat = BDM.zeros[Float](P/num_partitions,P/num_partitions)
      val seq1 = itr1.toSeq
      val seq2 = itr2.toSeq
      seq1.foreach { case (x1)  =>
        seq2.foreach { case (x2) =>
          print( " i & j : "+ x1._1.toInt,x2._1.toInt)
          resultMat(x1._1.toInt % elemPerPartition, x2._1.toInt % elemPerPartition)
            = (BDV(x1._2).t * BDV(x2._2)).toFloat
        }
      }
      Buf .+= ((indx1, indx2, resultMat))
    }

  }

  def computeCovarianceRDDbyBlock1(r:RDD[LabeledPoint],N: Int, P: Int,num_partitions:Int)
  = {

    val colMean: Array[Double] = r
      .map( x => x.features.toArray )
      .reduce( _ + _ )
      .map( x => x / N )

    val MeanShiftedMat: RDD[Array[Float]] = r
      .map( x => ( x.features.toArray - colMean ).map(elem => elem.toFloat)  )


    val blokRDD: RDD[(Long, Array[Float])] = MeanShiftedMat
      .zipWithIndex().map{case (v,k)=>(k,v)}
      .partitionBy(new MyPartitioner( num_partitions, P)).persist



    val Buff1= new ListBuffer[(Int,Iterator[(Long,Array[Float])])]
    val res1: RDD[(Int, Iterator[(Long, Array[Float])])]
    = blokRDD.mapPartitionsWithIndex{ (indx, itr) =>
      {
        Buff1 += ((indx,itr))
      }.toIterator.take(num_partitions)
    }

    val Buff2= new ListBuffer[(Int,Iterator[(Long,Array[Float])])]
    val res2: RDD[(Int, Iterator[(Long, Array[Float])])]
    = blokRDD.mapPartitionsWithIndex{ (indx, itr) =>
      {
        Buff2 += ((indx,itr))
      }.toIterator.take(num_partitions)
    }


    res1.cartesian(res2).toLocalIterator.flatMap{ case ((indx1, itr1), (indx2, itr2)) =>
      val Buf = new ListBuffer[(Int, Int, BDM[Float])]
      if (indx1>indx2) {

        val elemPerPartition: Int = P / num_partitions
        val mat1: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr1.toIndexedSeq.map { x => x._2 }.flatten.toArray)
        val mat2: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr2.toIndexedSeq.map { x => x._2 }.flatten.toArray)
        Buf.+=((indx1, indx2, mat1.t * mat2))
      } else if (indx1==indx2) {
        //val Buf = new ListBuffer[(Int, Int, BDM[Float])]
        val elemPerPartition: Int = P / num_partitions
        val mat1: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr1.toIndexedSeq.map { x => x._2 }.flatten.toArray)
        //val mat2: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr2.toIndexedSeq.map { x => x._2 }.flatten.toArray)
        Buf.+=((indx1, indx1, mat1.t * mat1))

      } else { Buf.+=((indx1, indx1, BDM.zeros[Float](1,1)))}

    }

  }

  def computeMeanShiftedMatrix(r:RDD[LabeledPoint],N: Int): RDD[Array[Float]] =
  {
    val colMean: Array[Double] = r
      .map( x => x.features.toArray )
      .reduce( _ + _ )
      .map( x => x / N )

   r.map( x => ( x.features.toArray - colMean ).map(elem => elem.toFloat))

  }
  def computeCovarianceRDDbyBlock2(MeanShiftedMat:RDD[Array[Float]],N: Int, P: Int,num_partitions:Int)
  : RDD[(Int, Int, BDM[Float])] = {



    val blokRDD: RDD[(Long, Array[Float])] = MeanShiftedMat
      .zipWithIndex().map{case (v,k)=>(k,v)}
      .partitionBy(new MyPartitioner( num_partitions, P)).persist(StorageLevel.MEMORY_AND_DISK)



    val Buff1= new ListBuffer[(Int,Iterator[(Long,Array[Float])])]
    val res1: RDD[(Int, Iterator[(Long, Array[Float])])]
    = blokRDD.mapPartitionsWithIndex({ (indx, itr) =>
      {
        Buff1 += ((indx,itr))
      }.toIterator.take(num_partitions)
    },true)

    val Buff2= new ListBuffer[(Int,Iterator[(Long,Array[Float])])]
    val res2: RDD[(Int, Iterator[(Long, Array[Float])])]
    = blokRDD.mapPartitionsWithIndex({ (indx, itr) =>
      {
        Buff2 += ((indx,itr))
      }.toIterator.take(num_partitions)
    },true)


    res1.cartesian(res2).flatMap{ case ((indx1, itr1), (indx2, itr2)) =>

      if (indx1<indx2) {
        val Buf = new ListBuffer[(Int, Int, BDM[Float])]
        val elemPerPartition: Int = P / num_partitions
        val mat1: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr1.toIndexedSeq.map { x => x._2 }.flatten.toArray)
        val mat2: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr2.toIndexedSeq.map { x => x._2 }.flatten.toArray)
        Buf.+=((indx1, indx2, mat1.t * mat2))
      } else if (indx1==indx2) {
        val Buf = new ListBuffer[(Int, Int, BDM[Float])]
        val elemPerPartition: Int = P / num_partitions
        val mat1: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr1.toIndexedSeq.map { x => x._2 }.flatten.toArray)
        //val mat2: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr2.toIndexedSeq.map { x => x._2 }.flatten.toArray)
        Buf.+=((indx1, indx1, mat1.t * mat1))

      } else Iterator()
    }

  }

  def computeCovarianceRDDbyBlockBroadcast(MeanShiftedMat:RDD[Array[Float]],N: Int, P: Int,num_partitions:Int):
  RDD[(Int, Int, BDM[Float])]
  = {
    val blokRDD: RDD[(Long, Array[Float])] = MeanShiftedMat
      .zipWithIndex().map{case (v,k)=>(k,v)}
      .partitionBy(new MyPartitioner( num_partitions, P)).persist(StorageLevel.MEMORY_AND_DISK)


    val Buff1= new ListBuffer[(Int,Iterator[(Long,Array[Float])])]
    val res1
    = blokRDD.mapPartitionsWithIndex({ (indx, itr) =>
      {
        Buff1 += ((indx,itr))
      }.toIterator.take(num_partitions)
    },true)

    val Buff2= new ListBuffer[(Int,Iterator[(Long,Array[Float])])]
    val res2
    = blokRDD.mapPartitionsWithIndex({ (indx, itr) =>
      {
        Buff2 += ((indx,itr))
      }.toIterator.take(num_partitions)
    },true)

    val sc = blokRDD.context
    val Buf = new ListBuffer[(Int, Int, BDM[Float])]

    //val rr: RDD[(Int, Int, BDM[Float])] = res1.flatMap { (part1: (Int, Iterator[(Long, Array[Float])])) =>
    val computelocalmatrix = ( part1:(Int, Iterator[(Long, Array[Float])])) =>{
        val onepart = sc.broadcast(part1)

        res2.flatMap { case (indx2, itr2) =>

          val indx1 = onepart.value._1
          val itr1 = onepart.value._2

          if (indx1 < indx2) {

            val elemPerPartition: Int = P / num_partitions
            val mat1: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr1.toIndexedSeq.map { x => x._2 }.flatten.toArray)
            val mat2: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr2.toIndexedSeq.map { x => x._2 }.flatten.toArray)
            Buf.+=((indx1, indx2, mat1.t * mat2))
          } else if (indx1 == indx2) {
            //val Buf = new ListBuffer[(Int, Int, BDM[Float])]
            val elemPerPartition: Int = P / num_partitions
            val mat1: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr1.toIndexedSeq.map { x => x._2 }.flatten.toArray)
            //val mat2: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr2.toIndexedSeq.map { x => x._2 }.flatten.toArray)
            Buf.+=((indx1, indx1, mat1.t * mat1))

          } else Iterator()
        }.toLocalIterator
      }
    res1.flatMap(computelocalmatrix)
  }

  def computeCovarianceRDDbyBlockBroadcast1Backup(MeanShiftedMat:RDD[Array[Float]],N: Int, P: Int,num_partitions:Int)
  = {
    val blokRDD: RDD[(Long, Array[Float])] = MeanShiftedMat
      .zipWithIndex().map{case (v,k)=>(k,v)}
      .partitionBy(new MyPartitioner( num_partitions, P)).persist(StorageLevel.MEMORY_AND_DISK)

    val sc = blokRDD.sparkContext

    val Buff1= new ListBuffer[(Int,Seq[(Long,Array[Float])])]
    val res1: RDD[(Int, Seq[(Long, Array[Float])])]
    = blokRDD.mapPartitionsWithIndex({ (indx, itr) =>
      {
        Buff1 += ((indx,itr.toSeq))
      }.toIterator.take(num_partitions)
    },true)

    val Buff2= new ListBuffer[(Int,Seq[(Long,Array[Float])])]
    val res2: RDD[(Int, Seq[(Long, Array[Float])])]
    = blokRDD.mapPartitionsWithIndex({ (indx, itr) =>
      {
        Buff2 += ((indx,itr.toSeq))
      }.toIterator.take(num_partitions)
    },true)

    res1.flatMap{ (part1: (Int, Seq[(Long, Array[Float])])) =>
      val onepart = sc.broadcast(part1)
      val broadp = sc.broadcast(P)
      val broadnumpart = sc.broadcast(num_partitions)

      val innerloop
      = (part2 :( Int,Seq[(Long, Array[Float])])) => {
          val indx1 = onepart.value._1
          val itr1 = onepart.value._2

          val indx2=part2._1
          val itr2 = part2._2
          val P = broadp.value
          val num_partitions = broadnumpart.value
          val Buf = new ListBuffer[(Int, Int, BDM[Float])]

          if (indx1 < indx2) {

            val elemPerPartition: Int = P / num_partitions
            val mat1: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr1.toIndexedSeq.map { x => x._2 }.flatten.toArray)
            val mat2: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr2.toIndexedSeq.map { x => x._2 }.flatten.toArray)
            Buf.+=((indx1, indx2, mat1.t * mat2))
          } else if (indx1 == indx2) {

            val elemPerPartition: Int = P / num_partitions
            val mat1: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr1.toIndexedSeq.map { x => x._2 }.flatten.toArray)
            Buf.+=((indx1, indx1, mat1.t * mat1))
          } else{ Buf.+=((indx1, indx1, BDM.zeros[Float](1,1)))}
        }

      res2.flatMap(innerloop).collect
      //res2.toLocalIterator

      }

  }



  def computeCovarianceRDDbyBlockBroadcast2(MeanShiftedMat:RDD[Array[Float]],N: Int, P: Int,num_partitions:Int)
  = {
    val blokRDD: RDD[(Long, Array[Float])] = MeanShiftedMat
      .zipWithIndex().map{case (v,k)=>(k,v)}
      .partitionBy(new MyPartitioner( num_partitions, P)).persist(StorageLevel.MEMORY_ONLY)



    val Buff2= new ListBuffer[(Int,Iterator[(Long,Array[Float])])]
    val res2
    = blokRDD.mapPartitionsWithIndex({ (indx, itr) =>
      {
        Buff2 += ((indx,itr))
      }.toIterator.take(num_partitions)
    },true)

    val sc = blokRDD.context

    val parts=blokRDD.partitions

    for (p<-parts) {
      val idx = p.index

       blokRDD.mapPartitionsWithIndex ({ case (indx1: Int, itr1: Iterator[(Long, Array[Float])]) =>
        if (idx == indx1) {
          val bindx1 = sc.broadcast(indx1)
          val bitr1 = sc.broadcast(itr1)
          val broadp = sc.broadcast(P)
          val broadnumpart = sc.broadcast(num_partitions)

          val innerloop
          = (part2: (Int, Iterator[(Long, Array[Float])])) => {
            val indx1 = bindx1.value
            val itr1 = bitr1.value

            val indx2 = part2._1
            val itr2 = part2._2
            val P = broadp.value
            val num_partitions = broadnumpart.value
            val Buf = new ListBuffer[(Int, Int, BDM[Float])]

            if (indx1 < indx2) {

              val elemPerPartition: Int = P / num_partitions
              val mat1: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr1.toIndexedSeq.map { x => x._2 }.flatten.toArray)
              val mat2: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr2.toIndexedSeq.map { x => x._2 }.flatten.toArray)
              Buf.+=((indx1, indx2, mat1.t * mat2))
            } else if (indx1 == indx2) {

              val elemPerPartition: Int = P / num_partitions
              val mat1: BDM[Float] = new BDM(elemPerPartition, elemPerPartition, itr1.toIndexedSeq.map { x => x._2 }.flatten.toArray)
              Buf.+=((indx1, indx1, mat1.t * mat1))
            } else {
              Buf.+=((indx1, indx1, BDM.zeros[Float](1, 1)))
            }
          }

          //res2.toLocalIterator.flatMap(innerloop)
          res2.toLocalIterator
        }
        else Iterator()

      },true)
    }

  }

  def toBreeze(matin : CoordinateMatrix): BDM[Double] = {
    val m = matin.numRows().toInt
    val n = matin.numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    matin.entries.collect().foreach { case MatrixEntry(i, j, value) =>
      mat(i.toInt, j.toInt) = value
    }
    mat
  }


  def exPartitionBy(MeanShiftedMat:RDD[Array[Float]],N: Int, P: Int,num_partitions:Int)
  = {
    val blokRDD: RDD[(Long, Array[Float])] = MeanShiftedMat
      .zipWithIndex().map{case (v,k)=>(k,v)}
      .partitionBy(new MyPartitioner( num_partitions, P)).persist(StorageLevel.MEMORY_AND_DISK)



    blokRDD
  }

  def exPartitionMap(blokRDD:RDD[(Long, Array[Float])],num_partitions:Int)= {
    val sc = blokRDD.sparkContext

    val Buff1 = new ListBuffer[(Int, Seq[(Long, Array[Float])])]
    val res1
    = blokRDD.mapPartitionsWithIndex({ (indx, itr) => {
      Buff1 += ((indx, itr.toSeq))
    }.toIterator.take(num_partitions)
    }, true)
    res1
  }


  def exPartitionMap1(blokRDD:RDD[(Long, Array[Float])],num_partitions:Int,num_features:Int)= {
    val sc = blokRDD.sparkContext

    val Buff1 = new ListBuffer[(Int, BDM[Float])]
    val rownum= num_features/num_partitions
    val res1
    = blokRDD.mapPartitionsWithIndex({ (indx, itr) => {
      val bbb: BDM[Float] =BDM.zeros[Float](rownum,num_features)
      itr.foreach{ case (row,arr)=>
          val ccc: BDV[Float] =BDV(arr)
          bbb((row%rownum).toInt,::) := ccc.t
      }
      Buff1 += ((indx, bbb))
    }.toIterator.take(num_partitions)
    }, true)
    res1
  }


  def main(args: Array[String]): Unit = {


    //Logger.getLogger("org").setLevel(Level.OFF)

    val sc = new SparkContext(new SparkConf()
      //.setMaster("local[*]").setAppName("PCAExampleTest")
      .set("spark.driver.maxResultSize", "90g")
      .set("spark.akka.timeout","2000000")
      .set("spark.worker.timeout","5000000")
      .set("spark.storage.blockManagerSlaveTimeoutMs","5000000")
      .set("spark.akka.frameSize", "2047")
      .set("spark.akka.threads","12")
      .set("spark.network.timeout","600")
      .set("spark.rpc.askTimeout","600")
      .set("spark.rpc.lookupTimeout","600")
      .set("spark.network.timeout", "10000000")
      .set("spark.executor.heartbeatInterval","10000000"))

    //val sc = new SparkContext(new SparkConf().setMaster("local[*]"))

/*
    val csv: RDD[String] = sc.textFile("/Users/tonypark/ideaProjects/SparkMLStudy/src/test/resources/pcatest.csv")

    val dataRDD: RDD[Vector] = csv.map(line => Vectors.dense( line.split(",").map(elem => elem.trim.toDouble)))
    val mat = new RowMatrix(dataRDD)
    val num_samples=mat.numRows().toInt
    val num_features=mat.numCols().toInt
    val num_new_partitions=dataRDD.getNumPartitions

    val lds: RDD[LabeledPoint] = dataRDD.map{ x=>
      new LabeledPoint(0, x)
    }

*/
    var num_features:Int =10000
    var num_samples:Int =10000
    var num_bins:Int = 20
    var num_new_partitions:Int = 50

    if (!args.isEmpty){

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt

    }

    val recordsPerPartition: Int = num_samples / num_new_partitions
    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitionsWithIndex { (idx,iter) => {
        val gauss: Gaussian = new Gaussian(0.0, 1.0)
        (1 to recordsPerPartition).map { i =>
          new LabeledPoint(idx, Vectors.dense(gauss.sample(num_features).toArray))
        }
      }.toIterator
      }

    //lds.cache()
   // val start = System.currentTimeMillis
     // val aa: RowMatrix =lds.computeCovarianceRDD(num_samples, num_features)

   //val bb: Array[Vector] = aa.rows.takeSample(false,1000)

   // println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start)/1000.0))

     println(" ************* begin *********************")
    //val start = System.currentTimeMillis
    //val aa: RDD[(Int,Int, BDM[Double])] =computeCovarianceRDDNew(lds,num_samples, num_features,num_new_partitions)
    //val aa: RDD[(Int,Int, BDM[Float])] =computeCovarianceRDDbyBlock1(lds,num_samples, num_features,num_new_partitions)
    val msm: RDD[Array[Float]] =computeMeanShiftedMatrix(lds,num_samples)

    val bb: RDD[(Long, Array[Float])] =
      exPartitionBy(msm,num_samples, num_features,num_new_partitions)

    /*
    bb.mapPartitionsWithIndex{ case(idx,iter)=>
      iter.map {
        case(ind1,arr) =>
          "Part Num : " + idx + " array num : " + ind1 + " array value" + arr.map{z=>z.toString+","}.reduce(_+_)+ "\n"
        }
      }.foreach(println)

    println(" ***********************PartitionMap ****************" )
*/
   // val cc: RDD[(Int, Seq[(Long, Array[Float])])] = exPartitionMap(bb,num_new_partitions)
    val cc: RDD[(Int, BDM[Float])] = exPartitionMap1(bb,num_new_partitions,num_features)
    //val ee = exPartitionMap(bb,num_new_partitions)
/*

// One Partition exmaple
    cc.mapPartitionsWithIndex{ case(idx,iter)=>
      iter.map {
        case(ind1,arr) =>
           arr.map {
             case (ind2, arr2) => {
               ("Part Num : " + idx + " Orig Part : " + ind1.toString
               + " array indx" + ind2.toString + "Array Content" + arr2.map { z => z.toString + "," }.reduce(_ + _) + "\n")
             }
           }.reduce(_+_)
      }
    }.foreach(println)


    //Cartesian Partitino exmaple --> by index

    cc.cartesian(ee).mapPartitionsWithIndex{case(cartindex, cartseq)=>
      cartseq.map{case((part1,seq1),(part2,seq2))=>
      seq1.map { case (idx1, arr1) =>
        seq2.map { case (idx2, arr2) =>
          (idx1.toInt%(num_features/num_new_partitions) + "," + idx2.toInt%(num_features/num_new_partitions) +
            ","+ arr1.zip(arr2).map{x=>x._1*x._2}.reduce(_+_)+" - ")
          //("Cartesian ind : "+cartindex + " Part1 : "+part1+ " Part2 : "+ part2
          //  + " x: "+idx1 + " y :" + idx2 + " Inner Product: "+ arr1.zip(arr2).map{x=>x._1*x._2}.reduce(_+_) +"\n" )
        }.reduce(_+_)
      }.reduce(_+_)
      }
    }.foreach(print)

    */

  // view how cartesian produce behave with indexing
    /*
    val res: RDD[(Int, Int, BDM[Float])] = cc.cartesian(ee).mapPartitionsWithIndex{case(cartindex, cartseq)=>
      cartseq.map{case((part1,seq1),(part2,seq2))=>
        val localMat=BDM.zeros[Float](num_features/num_new_partitions,num_features/num_new_partitions)
        seq1.foreach { case (idx1, arr1) =>
          seq2.foreach { case (idx2, arr2) =>
            //println(idx1.toInt%(num_features/num_new_partitions))
            localMat(idx1.toInt%(num_features/num_new_partitions),idx2.toInt%(num_features/num_new_partitions))
              = arr1.zip(arr2).map{ x => x._1 * x._2}.reduce(_+_)
          }//.reduce(_+_)
        }//.reduce(_+_)
        (part1,part2,localMat)
      }
    }

    */

/*
    val res: RDD[(Int, Int, BDM[Float])] = cc.cartesian(ee).mapPartitions{case( cartseq)=>
      cartseq.map{case((part1,seq1),(part2,seq2))=>
        val localMat = BDM.zeros[Float](num_features / num_new_partitions, num_features / num_new_partitions)
          if (part1>=part2) {
        seq1.foreach { case (idx1, arr1) =>
            seq2.foreach { case (idx2, arr2) =>
              if (idx1 >= idx2) {
                //println(idx1.toInt%(num_features/num_new_partitions))
                localMat((idx1 % (num_features / num_new_partitions)).toInt, (idx2 % (num_features / num_new_partitions)).toInt)
                  = arr1.zip(arr2).map { x => x._1 * x._2 }.reduce(_ + _)
              }
            }
          }
        }
        (part1,part2,localMat)
      }
    }



    val res: RDD[(Long, (Int, String))] = cc.cartesian(cc).mapPartitionsWithIndex { case (partidx, cartseq) =>
  cartseq.map { case ((part1, seq1), (part2, seq2: Seq[(Long, Array[Float])])) =>
    val localMat = BDM.zeros[Float](num_features / num_new_partitions, num_features / num_new_partitions)
    //if (part1<=part2) {
    seq1.foreach { case (idx1, arr1: Array[Float]) =>
      seq2.foreach { case (idx2, arr2: Array[Float]) =>
        //if (idx1 >= idx2) {
        //println(idx1.toInt%(num_features/num_new_partitions))
        localMat((idx1 % (num_features / num_new_partitions)).toInt, (idx2 % (num_features / num_new_partitions)).toInt)
          = arr1.zip(arr2).map { x => x._1 * x._2 }.reduce(_ + _)
        // }
      }
    }
    (part2.toLong,(part1, localMat.toArray.map(i => i.toString + ",").reduce(_ + _)))
    }//.toIterator
  }.partitionBy(new MyPartitioner( num_new_partitions, num_features*num_features)).persist(StorageLevel.MEMORY_AND_DISK)

    val ee: Array[(Int, BDM[Float])] =cc.collect

    //ee.foreach(println)

    val res  = ee.map{case (part1, seq1)=>
      val bro =sc.broadcast(seq1)
      val bropart1 = sc.broadcast(part1)
      val Buf = new ListBuffer[(Long,(Int,BDM[Float]))]
      cc.flatMap{ case (part2, seq2) =>
        //val localMat = BDM.zeros[Float](num_features / num_new_partitions, num_features / num_new_partitions)
        //if (part1<=part2) {
        val seq: BDM[Float] = bro.value
        val mul: BDM[Float] =seq * seq1.t

        Buf += ((part2.toLong,(bropart1.value, mul)))
      }
      //Buf
    }

*/


    //This code is the working one with 100000 and 1000 partitions fast
    val ee: Array[(Int, BDM[Float])] =cc.collect

    //ee.foreach(println)

    val res: Array[RDD[(Long, (Int, BDM[Float]))]] = ee.map{case (part1, seq1)=>
      val bro =sc.broadcast(seq1)
      val bropart1 = sc.broadcast(part1)
      val Buf = new ListBuffer[(Long,(Int,BDM[Float]))]
      cc.flatMap{ case (part2, seq2) =>
        //val localMat = BDM.zeros[Float](num_features / num_new_partitions, num_features / num_new_partitions)
        //if (part1<=part2) {
        if (bropart1.value <= part2) {
          val seq: BDM[Float] = bro.value
          val mul: BDM[Float] = seq * seq1.t
          Buf += ((part2.toLong,(bropart1.value, mul)))
        }else
        { val mul = BDM.zeros[Float](1,1)
          Buf += ((part2.toLong,(bropart1.value, mul)))}
        //Buf += ((part2.toLong,(bropart1.value, mul)))
      }
      //Buf
    }
    //val res = Buf.flatten




    /*
   val res = ee.flatMap{case (part1, seq1)=>
      val bro =sc.broadcast(seq1)
      val bropart1 = sc.broadcast(part1)
      val Buf = new ListBuffer[(Long,(Int,String))]
      cc.flatMap{ case (part2, seq2) =>
        val localMat = BDM.zeros[Float](num_features / num_new_partitions, num_features / num_new_partitions)
        //if (part1<=part2) {
        val seq: Seq[(Long, Array[Float])] = bro.value
        seq.foreach{ case (idx1: Long, arr1: Array[Float]) =>
          seq2.foreach{ case (idx2, arr2: Array[Float]) =>
            //if (idx1 >= idx2) {
            localMat((idx1 % (num_features / num_new_partitions)).toInt,
              (idx2 % (num_features / num_new_partitions)).toInt)
              = arr1.zip(arr2).map { x => x._1 * x._2 }.reduce(_ + _)
            // }
          }
        }
        Buf += ((part2.toLong,(bropart1.value, localMat.toArray.map(i => i.toString + ",").reduce(_ + _))))
      }
      Buf.toIterator
    }//.foreach(println)

*/


    val start=System.currentTimeMillis()
    var cnt=0
    res.foreach{x =>
      x.saveAsTextFile("result"+start.toString + "_" + cnt.toString)
      cnt +=1
    }

    //println("\n\nThe res buffer : " + res.length)
    //res.saveAsTextFile("temp"+ System.currentTimeMillis)

    //res.collectFirst{case(i,(j,arr)) => println("x:" + i.toString + " y:" + j.toString+ "array:" +arr.toString)}//.foreach(println)

   /*
    res.foreach{x =>
      val elem: Array[(Long, (Int, BDM[Float]))] = x.collect
      elem.foreach{case(i,(j,arr)) => println("x:" + i.toString + " y:" + j.toString+ "array:" +arr.toString)}//.foreach(println)
    }

    */
    //while(res.hasNext){
    //  val tmp=res.next
    //  println("x:" + tmp._1 + " y:" + tmp._2+ "array:" +tmp._3)
    //}

//  error on nested RDD map
 /*   cc.mapPartitionsWithIndex { case (idx1, iter1) =>
      iter1.map {
        case (ind1, arr1) =>
          ee.mapPartitionsWithIndex { case (idx2, iter2) =>
            iter2.map {
              case (ind2, arr2) =>
                arr1.map {
                  case (rownum1, rowval1) =>
                    arr2.map {
                      case (rownum2, rowval2) => {
                        ("Part Num 1 : " + idx1 + "Part Num 2 : "+idx2 + " Orig Part : " + ind1 + "Orig Par2 : "+ ind2
                          + " array indx 1 :" + rownum1.toString + "array indx" + rownum2
                          + "Array Content 1" + rowval1.map { z => z.toString + "," }.reduce(_ + _)
                          + "Array Content 2" + rowval2.map { z => z.toString + "," }.reduce(_ + _)
                          + "\n")
                      }
                    }.reduce(_ + _)

                }
            }
          }
      }
    }.foreach(println)

    val dd = cc.collect
    (0 until dd.length).map{i =>
       val parind=dd(i)._1
       dd(i)._2.toArray.map{case(ind1,arr1)=>
         "Part Num : " + parind + " array num : " + ind1 + " array value" + arr1.map{z=>z.toString+","}.reduce(_+_)+ "\n"
       }
    }.foreach(println)
    */




    //println(bb.collect.map{case(ind,seq)=>
    //seq.map{case(ind1,arr)=> "Part Num : " + ind + " array num : "
    //  + ind1 + " array value" + arr.map{z=>z.toString+","}.reduce(_+_)+"\n"}})

    //  bb.foreach{case (indx1,indx2,arr)=> print( " (" + indx1 +"," + indx2+ ")," )}

    //bb.saveAsTextFile( start.toString + "_covarianceTest")

    //println("The number of Partitinos :" + bb.getNumPartitions)
    println("*******************************************\n")
    //val cc = aa.collect.map{case (x,y,v)=>
//    println(bb.map{ case (indx1,indx2, arr) =>
//       " Part Index: "+ indx1 +" , row : " +indx1+" , col : " +indx2+"\n" + arr +"\n"
//     }.reduce(_+_))


/*
    val part=bb.partitions

    for (p<-part){
      val idx = p.index

      val partRdd = bb.mapPartitionsWithIndex({case(indx1,iter) =>
        if (indx1 == idx) iter else Iterator()}, true)
      //The second argument is true to avoid rdd reshuffling
      val data: Array[(Int, Int, BDM[Float])] = partRdd.collect //data contains all values from a single partition
      //in the form of array
      //Now you can do with the data whatever you want: iterate, save to a file, etc.

      data.foreach { case (indx1, indx2, arr) =>
        println(" Part Index: " + idx + " , row : " + indx1+" , col : " + indx2 + "\n" + arr + "\n")
      }
    }   */

    //println(bb.mapPartitionsWithIndex{ case(indx,iter) => iter.take(1).map{case (indx1,indx2, arr) =>
    //  " P: "+ indx +", r: " +indx1+" ,c: " +indx2+ "M(0,0): " + arr(0,0)+ "\n"
    //}}.reduce(_+_))

    //println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start)/1000.0))
    println(" Number of features : " + num_features)
    println(" Number of partitions : " + num_new_partitions)

    sc.stop()

  }

}
