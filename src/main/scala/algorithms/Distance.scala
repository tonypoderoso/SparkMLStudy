

package algorithms


import breeze.linalg.{Axis, min, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{AccumulatorParam, FutureAction, SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer
import preprocessing._

import scala.collection.immutable.{IndexedSeq, Range}
import scala.collection.parallel.ParSeq
import scala.reflect.ClassTag
import scala.tools.scalap.scalax.rules.Zero



/**
  * Created by tonypark on 2016. 6. 26..
  */
object Distance{

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

    val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("PCAExampleTest"))

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

    //al ldscol: Array[(Double, Vector)] =lds.collect.map{ elem=>
     // (elem.label,elem.features)}

    //println("\n Step1  : After making lds ")
    //ldscol.foreach(x=>print("ind: "+ x._1.toInt + "=>" +x._2+"\n"))


    val start = System.currentTimeMillis()

    //val msm: RDD[Array[Float]] = computeMeanShiftedMatrix(lds, num_samples)
    //println("\n Step2 : Before matrix msm")
    val msm: RDD[Array[Float]] = lds.map{ elem =>
      val aaa: Array[Float] =elem.features.toArray.map{ a=>
          a.toFloat}
      //println("idx1 : " +elem.label.toInt+ " => "+ aaa.map{x=>x + ","}.reduce(_+_))

    aaa}
    ///msm.collect
    //println("\n  Step2 : After matrix msm")

    val cc: RDD[(Int, BDM[Float])] =
      exPartitionMap(msm, num_new_partitions, num_features).cache()

    //println("  Step3 : After matrix cc")
    var dd: Seq[(Int, BDM[Float])] = cc.collectAsync().get().sortBy(x=>x._1)

    //println("\n  Step4 : After matrix dd")
//    for (i<- 0 until dd.length) {
//      println(dd(i)._1 + " : " + dd(i)._2)
//    }


    //println("number of dd rows " + dd.length)

    if (num_partsum>1) {
      dd = (0 until dd.length by num_partsum).par.map { i =>
        val tmp: BDM[Float] = (0 until num_partsum).map{ j =>
          dd(i + j)._2
        }.reduce(BDM.vertcat(_, _))
        (i + num_partsum-1  ,tmp )
      }.toList
    }

    //println("\n  Step4 : After matrix new dd")
//    for (i<- 0 until dd.length) {
//      println(dd(i)._1 + " : " + dd(i)._2)
//    }



    def distcompute1(a:BDM[Float],b:BDM[Float]) = {
      //println( "the a : " +a.rows+" , "+ a.cols +" and "+ b.rows+ " , "+ b.cols)
      val res1 =(0 until a.rows).map {i=>
        val tmp = b  - BDM.tabulate[Float](b.rows,a.cols){
          case (_,j) => a(i,j)}
       // println(tmp)
        //println("ind: "+i+ " ," +tmp.rows +" , "+tmp.cols)
        val res: BDM[Float] = sum(tmp :* tmp,Axis._1).toDenseMatrix
        //println(res.size )
        res
      }.reduce((x,y) =>
      BDM.vertcat(x,y))
      //println("the vertcat result: "+res1.rows +"," +res1.cols)
      res1
    }


    //println(" Before Distance Computation: " )
    val res: List[(Int, BDM[Float])] = dd.flatMap { case (part1: Int, seq1: BDM[Float]) => {
      val bro: Broadcast[(BDM[Float], Int)] = sc.broadcast(seq1, part1)
      val Buff1 = new ListBuffer[(Int, BDM[Float])]
      val block: (Int, BDM[Float]) = cc.map {
        case (part2: Int, seq2: BDM[Float]) =>
          if (bro.value._2 >= part2) {
            val rrr= distcompute1(bro.value._1,seq2)
            //println("\n The part : " +part1 + ","+ part2 +"\n matA \n"+bro.value._1+"\n matB \n"+seq2+"\nDistance \n"+rrr)
            (part2,rrr)
          } else {
            (part2,BDM.zeros[Float](bro.value._1.rows,seq2.rows))
          }
      }.collect.sortBy(_._1).reduce { (a, b) =>
        (0, BDM.horzcat(a._2, b._2))
      }
      Buff1 += ((part1, block._2))
    }.toIterator
    }.toList

    //println(" After Distance Computation: " )
    println("Elased Time in seconds : "+(System.currentTimeMillis - start)/1000.0)

//    val finalmat: BDM[Float] =res.map{case(idx,mat)=>
//      println(idx)
//      (0 until mat.rows).foreach{i=>
//        (0 until mat.cols).foreach{j=>
//        print(mat(i,j)+",")}
//      println}
//      mat}.reduce((a,b)=>BDM.vertcat(a,b))
//
//    val inputmatrix =dd.sortBy(_._1).map{case(idx,mat)=>
//      //println(idx)
//      mat}.reduce((a,b)=>BDM.vertcat(a,b))
//
//
//
//    //sc.parallelize(res.toList).saveAsTextFile("result" + System.currentTimeMillis().toString)
//
//    println(finalmat)
//    println("+++++++++++++++++++++++++++++++++++++++++++++++")
//    println(inputmatrix.toString(100,100))

    println("*******************************************\n")
    println(" Number of features : " + num_features)
    println(" Number of partitions : " + num_new_partitions)

    sc.getConf.getAll.foreach(println)
    sc.stop()

  }

}

/*    def repVec[T:ClassTag:Zero](in: BDV[T], nRow: Int): BDM[T] =
    {
      BDM.tabulate[T](nRow, in.size)({case (_, j) => in(j)})
    }

    def repMat[T:ClassTag:Zero](in: BDM[T], nRep: Int): BDM[T] =
    {
      BDM.tabulate[T](nRep, in.size)({case (_, j) => in(j)})
    }*/

/*def distcompute2(a:BDM[Float],b:BDM[Float]): BDM[Float] = {
  var tmp=BDM.zeros[Float](,a.cols)
  (1 to a.rows).map {i=>
    tmp = BDM.vertcat(tmp,b)
    val tmp: BDM[Float] = b //:- a(i)
    tmp * tmp.t
  }.reduce((x,y) =>
    BDM.horzcat(x,y))
}
*/