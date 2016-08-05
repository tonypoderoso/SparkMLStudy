
package algorithms


import breeze.linalg.{min, DenseMatrix => BDM, DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{AccumulatorParam, FutureAction, SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer
import preprocessing._

import scala.collection.immutable.{IndexedSeq,Range}
import scala.collection.parallel.ParSeq

class myStruct1(xStart:Int,yStart:Int,xLength:Int,yLength:Int,inputmat:BDM[Float]) extends Serializable{
  val x: Int = xStart
  val y: Int = yStart
  val xoff:Int=xLength
  val yoff:Int=yLength
  var mat:BDM[Float]=inputmat
  def reset = {
    mat=BDM.zeros[Float](xoff,yoff)
    this
  }
  val set = (newone:myStruct) => {
    mat(newone.x until newone.x+newone.xoff,
      newone.y until newone.y+newone.yoff) :=newone.mat
    this
  }

}

object VectorAccumulatorParam1 extends AccumulatorParam[myStruct] {
  def zero(initialValue:myStruct ) = {
    initialValue.reset
  }
  def addInPlace(v1: myStruct,v2:myStruct) ={
    v1.set(v2)
  }
}

/**
  * Created by tonypark on 2016. 6. 26..
  */
object CovarianceMain3 {

  def computeMeanShiftedMatrix(r:RDD[LabeledPoint],N: Int): RDD[Array[Float]] =
  {
    val colMean= r
      .map( x => x.features.toArray )
      .reduce( _ + _ )
      .map( x => x / N )

    r.map( x => ( x.features.toArray - colMean ).map(elem => elem.toFloat))

  }

  def exPartitionBy(MeanShiftedMat:RDD[Array[Float]]
                    ,N: Int, P: Int,num_partitions:Int)
  : RDD[(Long, Array[Float])] = {
    MeanShiftedMat
      .zipWithIndex
      .map{case (v,k)=>(k,v)}
      .partitionBy(new MyPartitioner( num_partitions, P))
      .persist(StorageLevel.MEMORY_ONLY)
  }

  def exPartitionMap1(blokRDD:RDD[(Long, Array[Float])],
                      num_partitions:Int,num_features:Int)
  : RDD[(Int, BDM[Float])] = {
    val sc = blokRDD.sparkContext

    val Buff1 = new ListBuffer[(Int, BDM[Float])]
    val rownum= num_features/num_partitions

    blokRDD.mapPartitionsWithIndex({ (indx, itr) => {
      val bbb: BDM[Float] =BDM.zeros[Float](rownum,num_features)
      itr.foreach{ case (row,arr)=>
        val ccc: BDV[Float] =BDV(arr)
        bbb((row%rownum).toInt,::) := ccc.t
      }
      Buff1 += ((indx, bbb))
    }.toIterator//.take(num_partitions)
    }, true)
  }

  def main(args: Array[String]): Unit = {


    Logger.getLogger("org").setLevel(Level.WARN)

    val sc = new SparkContext(new SparkConf())
    //)
      //.setMaster("local[*]").setAppName("PCAExampleTest"))
      //.set("spark.scheduler.mode", "FAIR")



    var num_features: Int = 10000
    var num_samples: Int = 10000
    var num_bins: Int = 20
    var num_new_partitions: Int = 100
    var num_partsum:Int = 10

    if (!args.isEmpty) {

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt
      num_partsum=args(4).toString.toInt

    }

    val acc = sc.accumulator( new myStruct(0,0,num_samples,num_features,BDM.zeros[Float](num_samples,num_features)))(VectorAccumulatorParam)
    val recordsPerPartition: Int = num_samples / num_new_partitions
    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitionsWithIndex { (idx, iter) => {
        val gauss: Gaussian = new Gaussian(0.0, 1.0)
        (1 to recordsPerPartition).map { i =>
          new LabeledPoint(idx, Vectors.dense(gauss.sample(num_features).toArray))
        }
      }.toIterator
      }

    //println("the number of partiotion of lds : "+lds.getNumPartitions)
    //println(" ************* begin *********************")
    val start = System.currentTimeMillis()

    val msm: RDD[Array[Float]] = computeMeanShiftedMatrix(lds, num_samples)

    val bb: RDD[(Long, Array[Float])] =
      exPartitionBy(msm, num_samples, num_features, num_new_partitions)

    val cc: RDD[(Int, BDM[Float])] =
      exPartitionMap1(bb, num_new_partitions, num_features)


    var dd: Seq[(Int, BDM[Float])] = cc.collectAsync().get().sortBy(x=>x._1)

    //println("number of dd rows " + dd.length)

    if (num_partsum>1) {
      dd = (0 until dd.length by num_partsum).par.map { i =>
        val tmp: BDM[Float] = (0 until num_partsum).map{ j =>
          //println("The i : " + i + " The j : " + j + " The numpartsum : "+ num_partsum + " The index : "+ (i + j))
          dd(i + j)._2
        }.reduce(BDM.vertcat(_, _))
        (i + num_partsum - 1,tmp )
      }.toList
    }


    val res: ParSeq[(Int, BDM[Float])] =  dd.par.map { case (part1, seq1) =>
      val bro = sc.broadcast(seq1, part1)
      val block: BDM[Float] = cc.map { case (part2, seq2) =>
        if (bro.value._2 >= part2) {
          bro.value._1 * seq2.t
        } else {
          BDM.zeros[Float](bro.value._1.rows,1)//seq2.rows)
        }
      }.reduce((a, b) =>
        BDM.horzcat(a, b)
      )
      (part1, block)
    }
    println("Elased Time in seconds : "+(System.currentTimeMillis - start)/1000.0)

     sc.parallelize(res.toList).saveAsTextFile("result" + System.currentTimeMillis().toString)



    println("*******************************************\n")
    println(" Number of features : " + num_features)
    println(" Number of partitions : " + num_new_partitions)

    sc.stop()

  }

}


