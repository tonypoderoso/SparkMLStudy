
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

class myStruct(xStart:Int,yStart:Int,xLength:Int,yLength:Int,inputmat:BDM[Float]) extends Serializable{
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

object VectorAccumulatorParam extends AccumulatorParam[myStruct] {
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
object CovarianceMain2 {

  def computeMeanShiftedMatrix(r:RDD[LabeledPoint],N: Int): RDD[Array[Float]] =
  {
    val colMean: Array[Double] = r
      .map( x => x.features.toArray )
      .reduce( _ + _ )
      .map( x => x / N )

    r.map( x => ( x.features.toArray - colMean ).map(elem => elem.toFloat))

  }

  def exPartitionBy(MeanShiftedMat:RDD[Array[Float]]
                    ,N: Int, P: Int,num_partitions:Int)
  : RDD[(Long, Array[Float])] = {
    MeanShiftedMat
      .zipWithIndex().map{case (v,k)=>(k,v)}
      .partitionBy(new MyPartitioner( num_partitions, P))
    //.persist(StorageLevel.MEMORY_AND_DISK)
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

    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]").setAppName("PCAExampleTest")
      //.set("spark.scheduler.mode", "FAIR")
      .set("spark.driver.maxResultSize", "220g")
      .set("spark.akka.timeout", "2000000")
      .set("spark.worker.timeout", "5000000")
      .set("spark.storage.blockManagerSlaveTimeoutMs", "5000000")
      .set("spark.akka.frameSize", "2047")
      .set("spark.akka.threads", "16")
      .set("spark.network.timeout", "7200000")
      .set("spark.rpc.askTimeout", "7200000")
      .set("spark.rpc.lookupTimeout", "7200000")
      .set("spark.network.timeout", "10000000")
      .set("spark.executor.heartbeatInterval", "10000")
      .set("scala.concurrent.context.maxThreads","8"))






    var num_features: Int = 5000
    var num_samples: Int = 5000
    var num_bins: Int = 20
    var num_new_partitions: Int = 100
    var num_partsum:Int = 1

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
    //val start = System.currentTimeMillis()

    val msm: RDD[Array[Float]] = computeMeanShiftedMatrix(lds, num_samples)

    val bb: RDD[(Long, Array[Float])] =
      exPartitionBy(msm, num_samples, num_features, num_new_partitions)

    val cc: RDD[(Int, BDM[Float])] =
      exPartitionMap1(bb, num_new_partitions, num_features).cache()
    //println("the number of partiotion of cc : "+lds.getNumPartitions)

    //val cc1 = exPartitionMap1(bb,num_new_partitions/num_partsum,num_features)

    var dd: Seq[(Int, BDM[Float])] = cc.collectAsync().get().sortBy(x=>x._1)

    if (num_partsum>1) {
      dd = (0 until dd.length by num_partsum).par.map { i =>
        (i + num_partsum - 1, (0 until num_partsum).map(j =>
          dd(i + j)._2).reduce(BDM.vertcat(_, _)))
      }.toList
    }


    //println(" the length of dd " + dd.length)
    //val res: RDD[(Int, BDM[Float])] = sc.parallelize({
    val res =  dd.map { case (part1, seq1) =>
      val bro = sc.broadcast(seq1, part1)
        cc.map{ case (part2, seq2) =>
       if (bro.value._2 >= part2) {
          val tmp: BDM[Float] =bro.value._1 * seq2.t
          println("Part1: "+part1+" , Part2: "+part2+ ", Rows:" + tmp.rows+ ", Cols : " +tmp.rows)
          acc.add( new myStruct(part1*tmp.rows, part2*tmp.cols, tmp.rows,tmp.cols, tmp))
        }
      }.collectAsync()
    }.toList
    //}.toList//)
    //implicit val context = ExecutionContext.Implicits.global
    //println(res.getNumPartitions)

    println(acc.value.mat)
    //println("The number of res partitions: " + res.getNumPartitions)
    //sc.parallelize(,num_new_partitions).saveAsTextFile("result" + System.currentTimeMillis().toString)
    //res.foreachAsync{ z =>
    // println("\n  x:" + z._1.toString + "\n" + z._2 + "\n")
    //}.get()
    // res.foreachPartitionAsync{dum=>dum.map{ z =>
    //   println("\n  x:" + z._1.toString + "\n" + z._2 + "\n")
    // }}.get()


    //println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start)/1000.0))

    println("*******************************************\n")
    println(" Number of features : " + num_features)
    println(" Number of partitions : " + num_new_partitions)

    sc.stop()

  }

}


