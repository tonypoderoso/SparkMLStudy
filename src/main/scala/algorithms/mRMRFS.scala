package algorithms

import breeze.linalg.{Axis, DenseMatrix => BDM, DenseVector => BDV, argmax => brzArgmax, argmin => brzArgmin, sum => brzSum}
import breeze.stats.distributions.Gaussian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import scala.collection.immutable.IndexedSeq
import scala.collection.mutable.ListBuffer

/**
  * Created by tonypark on 2016. 6. 13..
  */
object mRMRFS {

  def computeMIVector(b: (Array[Int],Long),a1: RDD[(Array[Int], Long)],
                      num_state1: Int,  num_state2: Int): (Long, Array[Double]) = {

    val sc = a1.context
    val onerow = sc.broadcast(b)

    val computeMutualInformation2: ((Array[Int], Long)) => ListBuffer[Double]
    = (input: ((Array[Int], Long))) => {
      val buf = new ListBuffer[(Long, Double)]()
      val col = input._2
      //if (onerow.value._2 > col) {
        val output = BDM.zeros[Int](num_state1, num_state2)
        val rsum: BDV[Int] = BDV.zeros[Int](num_state1)
        val csum: BDV[Int] = BDV.zeros[Int](num_state2)
        val msum: Int = onerow.value._1.length
        onerow.value._1.zip(input._1).map { x =>
          output(x._1 - 1, x._2 - 1) = output(x._1 - 1, x._2 - 1) + 1
          rsum(x._1 - 1) = rsum(x._1 - 1) + 1
          csum(x._2 - 1) = csum(x._2 - 1) + 1
        }
        buf += ((col, output.mapPairs { (coo, x) =>
          if (x > 0) {
            val tmp = msum.toDouble / (rsum(coo._1) * csum(coo._2))
            x * math.log(x * tmp) / math.log(2)
          } else
            0
        }.toArray.sum / msum ))
        //buf.+=((col, MI)) //MatrixEntry(row, col, tmp)
      //}buf

    }.sorted.map(x => x._2)

    // val mapper = //.sorted.map(x=>x._2)
    (onerow.value._2, a1.flatMap(computeMutualInformation2).collect)
    //a1.flatMap(computeMutualInformation2)
  }

  def computeMIVectorWithLookup(b: (Long, Array[Int]),a1: RDD[(Long, Array[Int])],
                      num_state1: Int,  num_state2: Int): (Long, Array[Double]) = {

    val sc = a1.context
    val onerow = sc.broadcast(b)

    val computeMutualInformation2: ((Long,Array[Int])) => ListBuffer[Double]
    = (input: ((Long, Array[Int]))) => {
      val buf = new ListBuffer[(Long, Double)]()
      val col = input._1
      //if (onerow.value._2 > col) {
      val output = BDM.zeros[Int](num_state1, num_state2)
      val rsum: BDV[Int] = BDV.zeros[Int](num_state1)
      val csum: BDV[Int] = BDV.zeros[Int](num_state2)
      val msum: Int = onerow.value._2.length
      onerow.value._2.zip(input._2).map{ x =>
        output(x._1 - 1, x._2 - 1) = output(x._1 - 1, x._2 - 1) + 1
        rsum(x._1 - 1) = rsum(x._1 - 1) + 1
        csum(x._2 - 1) = csum(x._2 - 1) + 1
      }
      buf += ((col, output.mapPairs { (coo, x) =>
        if (x > 0) {
          val tmp = msum.toDouble / (rsum(coo._1) * csum(coo._2))
          x * math.log(x * tmp) / math.log(2)
        } else
          0
      }.toArray.sum / msum ))
      //buf.+=((col, MI)) //MatrixEntry(row, col, tmp)
      //}buf

    }.sorted.map(x => x._2)

    // val mapper = //.sorted.map(x=>x._2)
    (onerow.value._1, a1.flatMap(computeMutualInformation2).collect)
    //a1.flatMap(computeMutualInformation2)
  }

  def computemRMR_Init(in: RDD[Array[Int]],number_of_features: Int, num_state1: Int,
                          num_state2: Int)= {

    val sc = in.sparkContext
    val a1: RDD[(Array[Int], Long)] = in.zipWithIndex().persist
    val a: Array[(Array[Int], Long)] = a1.collect
    val num_feat = a.length
    val mRMRSorting: BDV[Int] = BDV.zeros[Int](num_feat)
    val selected: BDV[Double] = BDV.ones[Double](num_feat)
    val MIScore = BDM.zeros[Double](number_of_features,num_feat)



    val res: (Long, Array[Double]) = computeMIVector(a(0), a1, num_state1, num_state2)
    res._2(0) = Double.MaxValue
    MIScore(0,::) := BDV(res._2).t
    print(" Line => "+res._1 + " : ")
    res._2.map { x => x.toString + " , " }.reduce(_ + _).foreach(print)
    println("")


    // ******************* First Features : Max Relevance ******************
    mRMRSorting(0) = brzArgmax(res._2)
    println("the maximum argument is : " + brzArgmax(res._2))
    selected(mRMRSorting(0)) = Double.MaxValue
    println("The selected vector is : " + selected.map(x => x.toString + ","))
    println("")

    (1 until number_of_features).map { indx =>

      val res1: (Long, Array[Double]) = computeMIVector(a(mRMRSorting(indx-1)), a1, num_state1, num_state2)
      res1._2(0)=Double.MaxValue

      val rel = mRMRSorting(0)
      val red = BDV(res1._2).t*selected
      MIScore(indx,::) := BDV(res1._2).t


      print(" Line => "+res1._1 + " : ")
      res1._2.map { x => x.toString + " , " }.reduce(_ + _).foreach(print)
      mRMRSorting(indx) = brzArgmax(BDV(res1._2) :* selected )
      println("the maximum argument is : " + brzArgmax(res1._2))
      selected(mRMRSorting(indx)) = 0
      println("The selected vector is : " + selected.map(x => x.toString + ","))

     }

    (0 until number_of_features).map{ i =>
      (0 until num_feat).map{ j =>
        MIScore(i,j) + ","
      }.reduce(_+_)+ "\n"

    }.foreach(println)



  }

  def computemRMR(in: RDD[Array[Int]],number_of_features: Int, num_state1: Int,
                       num_state2: Int)= {

    val sc = in.sparkContext
    val a1: RDD[(Array[Int], Long)] = in.zipWithIndex().persist
    val a: Array[(Array[Int], Long)] = a1.collect
    val num_feat = a.length
    val mRMRSorting: BDV[Int] = BDV.zeros[Int](number_of_features)
    val selected: BDV[Double] = BDV.ones[Double](num_feat)
    val MIScore = BDM.zeros[Double](number_of_features-1,num_feat)


    // ******************* First Features : Max Relevance ******************
    val res: (Long, Array[Double]) = computeMIVector(a(0), a1, num_state1, num_state2)
    res._2(0) = 0
    mRMRSorting(0) = brzArgmax(res._2)
    selected(mRMRSorting(0)) = Double.MaxValue

    print(" Line => "+res._1 + " : ")
    res._2.map { x => x.toString + " , " }.reduce(_ + _).foreach(print)
    println("")
    println("the maximum argument is : " + brzArgmax(res._2))

    println("The selected vector is : " + selected.map(x => x.toString + ","))
    println("")



    (1 until number_of_features).map { indx =>

      val res1: (Long, Array[Double]) = computeMIVector(a(mRMRSorting(indx-1)), a1, num_state1, num_state2)
      res1._2(0)=Double.MaxValue

      MIScore(indx-1,::) := (BDV(res1._2) :* selected).t
      println("The selected vector is : " + (BDV(res1._2) :* selected).map(x => x.toString + ",").reduce(_+_))

      val relvector = brzSum(MIScore(0 until indx,::),Axis._0).toArray.map{x=>x/indx}

      mRMRSorting(indx) = brzArgmin(BDV(relvector) :* selected )

      selected(mRMRSorting(indx)) = Double.MaxValue

      //println("The selected vector is : " + selected.map(x => x.toString + ",").reduce(_+_))

      //println(relvector.map(elem =>elem.toString.take(5) + ", ").reduce(_+_))

    }

    (0 until number_of_features-1).map{ i =>
      (0 until num_feat).map{ j =>
        MIScore(i,j).toString.take(5) + ","
      }.reduce(_+_)+ "\n"
    }.foreach(println)

    println("The Important features are : " + mRMRSorting.map(x => x.toString + ",").reduce(_+_))

  }

  def computemRMRWithLookup(in: RDD[Array[Int]],num_feat:Int,number_of_features: Int, num_state1: Int,
                  num_state2: Int)= {

    val sc = in.sparkContext
    val a1: RDD[(Long, Array[Int])] = in.zipWithIndex.map{ case (k,v) => (v,k) }.persist
    //val num_feat = a1.count().toInt
    val a: Array[Int] = a1.lookup(0).head
    val mRMRSorting: BDV[Int] = BDV.zeros[Int](number_of_features)
    val selected: BDV[Double] = BDV.ones[Double](num_feat)
    val MIScore = BDM.zeros[Double](number_of_features-1,num_feat)


    // ******************* First Features : Max Relevance ******************
    val res: (Long, Array[Double]) = computeMIVectorWithLookup( (0,a) , a1, num_state1, num_state2)
    res._2(0) = 0
    mRMRSorting(0) = brzArgmax(res._2)
    selected(mRMRSorting(0)) = Double.MaxValue

    print(" Line => "+res._1 + " : ")
    res._2.map { x => x.toString + " , " }.reduce(_ + _).foreach(print)
    println("")
    println("the maximum argument is : " + brzArgmax(res._2))

    println("The selected vector is : " + selected.map(x => x.toString + ","))
    println("")



    (1 until number_of_features).map { indx =>

      val a = a1.lookup(mRMRSorting(indx-1)).head

      val res1: (Long, Array[Double]) = computeMIVectorWithLookup((mRMRSorting(indx-1), a), a1, num_state1, num_state2)
      res1._2(0)=Double.MaxValue

      MIScore(indx-1,::) := (BDV(res1._2) :* selected).t
      println("The selected vector is : " + (BDV(res1._2) :* selected).map(x => x.toString + ",").reduce(_+_))

      val relvector = brzSum(MIScore(0 until indx,::),Axis._0).toArray.map{x=>x/indx}

      mRMRSorting(indx) = brzArgmin(BDV(relvector) :* selected )

      selected(mRMRSorting(indx)) = Double.MaxValue

      //println("The selected vector is : " + selected.map(x => x.toString + ",").reduce(_+_))

      //println(relvector.map(elem =>elem.toString.take(5) + ", ").reduce(_+_))

    }

    (0 until number_of_features-1).map{ i =>
      (0 until num_feat).map{ j =>
        MIScore(i,j).toString.take(5) + ","
      }.reduce(_+_)+ "\n"
    }.foreach(println)

    println("The Important features are : " + mRMRSorting.map(x => x.toString + ",").reduce(_+_))

  }

  def main(args:Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]")
      .setAppName("MutualINformationMain")
      .set("spark.driver.maxResultSize", "90g")
      .set("spark.akka.timeout", "200000")
      .set("spark.worker.timeout", "500000")
      .set("spark.storage.blockManagerSlaveTimeoutMs", "5000000")
      .set("spark.akka.frameSize", "1024"))
    //.set("spark.akka.heartbeat.interval","4000s")
    //.set("spark.akka.heartbeat.pauses","2000s"))
    //val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("Test"))
    var num_features: Int = 100
    var num_samples: Int = 100000
    var num_bins: Int = 200
    var num_new_partitions: Int = 5
    var num_selection = 10

    if (!args.isEmpty) {

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_selection=args(3).toString.toInt
      num_new_partitions = args(4).toString.toInt

    }

    Logger.getLogger("org").setLevel(Level.OFF)


    // Distributed processing
    //************************************************
    val recordsPerPartition: Int = num_samples / num_new_partitions


    val noise = 0.1
    val gauss = new Gaussian(0.0, 1.0)
    val weights: Array[Double] = gauss.sample(num_features).toArray
    val w = BDV(weights)

    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitions { _ => {
        val gauss = new Gaussian(0.0, 1.0)
        (1 to recordsPerPartition).map { _ =>
          val x = BDV(gauss.sample(num_features).toArray)
          val l = x.dot(w) + gauss.sample() * noise
          new LabeledPoint(l, Vectors.dense(x.toArray))
        }
      }.toIterator
      }

    val mi = new MutualInformation

    val ffd = mi.featureFromDataset(lds, 1)
    //println("2.ffd : " +ffd.getNumPartitions)

    val ffdt = mi.rddTranspose2(ffd)
    //println("3.ffdt: "+ffdt.getNumPartitions)

    val trans = mi.normalizeToUnitT(ffdt)

    //println("4. trans: "+  trans.getNumPartitions)

    val dvec: RDD[Array[Int]] = mi.discretizeVector1(trans, num_bins)

    val start= System.currentTimeMillis()

    computemRMR(dvec,num_selection,num_bins,num_bins)

    println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start)/1000.0))

    val start1= System.currentTimeMillis()

    computemRMRWithLookup(dvec,num_features+1,num_selection,num_bins,num_bins)

    println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start1)/1000.0))

    sc.stop()
  }

}
