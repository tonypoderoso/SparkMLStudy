
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



/**
  * Created by tonypark on 2016. 6. 26..
  */
object CovarianceMain4 {

  def computeMeanShiftedMatrix(r:RDD[LabeledPoint],N: Int): RDD[Array[Float]] =
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

    //val acc = sc.accumulator( new myStruct(0,0,num_samples,num_features,BDM.zeros[Float](num_samples,num_features)))(VectorAccumulatorParam)
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

    //val msm: RDD[Array[Float]] = computeMeanShiftedMatrix(lds, num_samples)
    val msm: RDD[Array[Float]] = lds.map{ elem => elem.features.toArray.map{ a=>a.toFloat}}

    val cc: RDD[(Int, BDM[Float])] =
      exPartitionMap(msm, num_new_partitions, num_features).cache()


    var dd: Seq[(Int, BDM[Float])] = cc.collectAsync().get().sortBy(x=>x._1)

    //println("number of dd rows " + dd.length)

    if (num_partsum>1) {
      dd = (0 until dd.length by num_partsum).par.map { i =>
        val tmp: BDM[Float] = (0 until num_partsum).map{ j =>
          //println("The i : " + i + " The j : " + j + " The numpartsum : "+ num_partsum + " The index : "+ (i + j))
          dd(i + j)._2
        }.reduce(BDM.vertcat(_, _))
        //println ( " The index : "+ (i+num_partsum).toString+"The row size : " + (i*tmp.rows).toString)
        (i + num_partsum-1  ,tmp )
      }.toList
    }


    val res: List[(Int, BDM[Float])] = dd.par.flatMap { case (part1, seq1) => {
      val bro = sc.broadcast(seq1, part1)
      val Buff1 = new ListBuffer[(Int, BDM[Float])]
      val block: BDM[Float] = cc.map {
        case (part2, seq2) =>
        if (bro.value._2 >= part2) {
          bro.value._1 * seq2.t
        } else {
         BDM.zeros[Float](bro.value._1.rows, 1) //seq2.rows)
        }
      }.reduce((a, b) =>
        BDM.horzcat(a, b)
      )
      Buff1 += ((part1, block))
    }.toIterator
    }.toList

/*   val collen: Int =cc.first._2.rows

    val res=  dd.map { case (part1, seq1) => {
      val bro = sc.broadcast(seq1, part1)
      println("The pAART NUMBER : " + part1 + " The Size of the Matrix is => Rows :" + seq1.rows + " Cols: " + part1 * collen)
      val block = BDM.zeros[Float](seq1.rows, part1 * collen)
      cc.foreach { case (part2, seq2) =>
        if (bro.value._2 > part2) {
          println("the value of part: " + part2 + " start: " + part2 * seq2.rows + " and end : " + (part2 + 1) * seq2.rows)
          val tmp = bro.value._1 * seq2.t
          println("The value of part1 : " + bro.value._2 + "The tmp rows: " + tmp.rows + " cols : " + tmp.cols)
          block(::, part2 * seq2.rows until (part2 + 1) * seq2.rows) := tmp
          println(tmp)

        } //.toList
        //.toList
        //println(block)
      }
      (part1, block)
    }
    }*/

    println("Elased Time in seconds : "+(System.currentTimeMillis - start)/1000.0)

    //sc.parallelize(res.toList).saveAsTextFile("result" + System.currentTimeMillis().toString)



    println("*******************************************\n")
    println(" Number of features : " + num_features)
    println(" Number of partitions : " + num_new_partitions)

    sc.stop()

  }

}


