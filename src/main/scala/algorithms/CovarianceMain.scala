package algorithms


import breeze.linalg.{min, DenseMatrix => BDM, DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{FutureAction, SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer
import preprocessing._

import scala.collection.immutable.IndexedSeq
import scala.collection.parallel.ParSeq



/**
  * Created by tonypark on 2016. 6. 26..
  */
object CovarianceMain {

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
      .persist(StorageLevel.MEMORY_AND_DISK)
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
    }.toIterator.take(num_partitions)
    }, true)
  }

  def main(args: Array[String]): Unit = {


    Logger.getLogger("org").setLevel(Level.WARN)

    val sc = new SparkContext(new SparkConf()
      //.setMaster("local[*]").setAppName("PCAExampleTest")
      //.set("spark.scheduler.mode", "FAIR")
      .set("spark.driver.maxResultSize", "90g")
      .set("spark.akka.timeout", "2000000")
      .set("spark.worker.timeout", "5000000")
      .set("spark.storage.blockManagerSlaveTimeoutMs", "5000000")
      .set("spark.akka.frameSize", "2047")
      .set("spark.akka.threads", "12")
      //.set("spark.network.timeout", "7200000")
      .set("spark.rpc.askTimeout", "7200000")
      .set("spark.rpc.lookupTimeout", "7200000")
      .set("spark.network.timeout", "10000000")
      .set("spark.executor.heartbeatInterval", "1000")
      .set("scala.concurrent.context.maxThreads","2"))


    var num_features: Int = 1000
    var num_samples: Int = 1000
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

    val recordsPerPartition: Int = num_samples / num_new_partitions
    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitionsWithIndex { (idx, iter) => {
        val gauss: Gaussian = new Gaussian(0.0, 1.0)
        (1 to recordsPerPartition).map { i =>
          new LabeledPoint(idx, Vectors.dense(gauss.sample(num_features).toArray))
        }
      }.toIterator
      }

     println("the number of partiotion of lds : "+lds.getNumPartitions)
    //println(" ************* begin *********************")
    val start = System.currentTimeMillis()

    val msm: RDD[Array[Float]] = computeMeanShiftedMatrix(lds, num_samples)

    val bb: RDD[(Long, Array[Float])] =
      exPartitionBy(msm, num_samples, num_features, num_new_partitions)

    val cc: RDD[(Int, BDM[Float])] =
      exPartitionMap1(bb, num_new_partitions, num_features)
    println("the number of partiotion of cc : "+lds.getNumPartitions)

    //val cc1 = exPartitionMap1(bb,num_new_partitions/num_partsum,num_features)

    var dd: Seq[(Int, BDM[Float])] = cc.collectAsync().get().sortBy(x=>x._1)

    if (num_partsum>1) {
       dd = (0 until dd.length by num_partsum).map { i =>
        (i + num_partsum - 1, (0 until num_partsum).map(j =>
          dd(i + j)._2).reduce(BDM.vertcat(_, _)))
      }
      //ee.foreach{case(i,j)=> println(i);println(j)}
    }

    val aaaaa: ParSeq[(Int, BDM[Float])] = dd.par

    println(" the length of dd " + dd.length)
     val res: RDD[(Int, BDM[Float])] = sc.parallelize({
       dd.par.map { case (part1, seq1) =>
         val bro = sc.broadcast(seq1, part1)
         val block: BDM[Float] = cc.map { case (part2, seq2) =>
           if (bro.value._2 <= part2) {
             bro.value._1 * seq2.t
           } else null
         }.reduce((a, b) =>
           if (b.isInstanceOf[BDM[Float]] && a.isInstanceOf[BDM[Float]]) BDM.horzcat(a, b)
           else if (!a.isInstanceOf[BDM[Float]] && b.isInstanceOf[BDM[Float]]) b
           else if (!b.isInstanceOf[BDM[Float]] && a.isInstanceOf[BDM[Float]]) a
           else null
         )
         (part1, block)
       }
     }.toList, dd.length)
    //implicit val context = ExecutionContext.Implicits.global
    println(res.getNumPartitions)

    println("The number of res partitions: " + res.getNumPartitions)
    res.saveAsTextFile("result" + System.currentTimeMillis().toString)
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

