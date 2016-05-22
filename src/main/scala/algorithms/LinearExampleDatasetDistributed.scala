package algorithms
import breeze.linalg._
import breeze.numerics._
import breeze.linalg.{DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.IndexedSeq
import scala.collection.immutable.Range.Inclusive
/**
  * Created by tonypark on 2016. 5. 12..
  */
object LinearExampleDatasetDistributed {

  def main(args:Array[String]): Unit = {




    def average[T](ts: Iterable[T])(implicit num: Numeric[T]) = {
      val filteringTS = ts.filter(v => !v.asInstanceOf[Double].equals(Double.NaN))
      num.toDouble(filteringTS.sum) / filteringTS.size
    }

    def columnCheck(inputs: RDD[LabeledPoint]): RDD[Array[Double]] = {

      val nrow: Long = inputs.count()
      val countGood = inputs.map{ v =>
        val x = new BDV(v.features.toArray)
        x.map(x => if (java.lang.Double.isNaN(x)) 0.0 else 1.0)
      }.reduce(_ + _)

      val colSum = inputs.map{ v =>
        val x = new BDV(v.features.toArray)
        x.map ( x => if (java.lang.Double.isNaN(x)) 0.0 else x)
      }.reduce(_ + _)

      val Mean: BDV[Double] =colSum :/countGood


      //countNanNMean.foreach(print)
      //countNan.foreach(print)
      val th=BDV.fill[Double](countGood.length){nrow.toDouble / 2.0}
      val tmp: Array[Int] = (th :< countGood ).toArray.map(b => if (b == true) 1 else 0)
      val elem = sum(tmp)

      val bb = new Array[Int](elem)
      var i = 0
      var cnt = 0
      while (i < tmp.length) {
        if (tmp(i) == 1) {
          bb(cnt) = i
          cnt += 1
        }
          i += 1
      }
      bb.foreach(print)
      val meanNew: Array[Double] =Mean(bb.toList).toArray
      inputs.map{v=>
        //val out=
        val a: Array[Double] = new BDV(v.features.toArray)(bb.toList).toArray

        println("Length of a : " + a.length + "Length of b : " + meanNew.length)

        (0 until a.length).map { i =>
          if (java.lang.Double.isNaN(a(i)))
            a(i) = meanNew(i)
        }
        a
      }

    }

    def columnCheck2(inputs: RDD[LabeledPoint]): RDD[Array[Double]] = {

      val nrow: Long = inputs.count()

      // Count the number of NaNs
      val countGood = inputs.map{ v =>
        val x = new BDV(v.features.toArray)
        x.map(x => if (java.lang.Double.isNaN(x)) 0.0 else 1.0)
      }.reduce(_ + _)

      val th=BDV.fill[Double](countGood.length){nrow.toDouble / 2.0}
      val tmp: Array[Int] = (th :< countGood ).toArray.map(b => if (b == true) 1 else 0)
      val elem = sum(tmp)

      val colLeft = new Array[Int](elem)
      var i = 0
      var cnt = 0
      while (i < tmp.length) {
        if (tmp(i) == 1) {
          colLeft(cnt) = i
          cnt += 1
        }
        i += 1
      }
      //colLeft.foreach(print)
      val colSum = inputs.map{ v =>
        val x = new BDV(v.features.toArray)
        x.map ( x => if (java.lang.Double.isNaN(x)) 0.0 else x)
      }.reduce(_ + _)

      val Mean: BDV[Double] =colSum :/countGood
      val meanNew: Array[Double] =Mean(colLeft.toList).toArray


      inputs.map { v =>
        //val out=
        val a: Array[Double] = new BDV(v.features.toArray)(colLeft.toList).toArray
        var count = 0
        (0 until a.length).map { i =>
          if (java.lang.Double.isNaN(a(i))) {
            a(i) = meanNew(i)
            count += 1
          }
        }
        if (count<a.length/4) a else null
      }
    }.filter(x=>x!=null)


    def filterDataSet(rawData: RDD[(LabeledPoint, Long)], groupAs: Array[Long], groupBs: Array[Long]) = {



      val featuresIndex: RDD[(Double, Int)] = rawData.flatMap(row => row._1.features.toArray.zipWithIndex)
      val transFeatures: RDD[(Iterable[Double], Long)] = featuresIndex.groupBy(_._2).values.map(_.map(_._1)).zipWithIndex()

      val limitOfMissingFeatures: Int = rawData.count().toInt / 2
      val fiteringFeatures: RDD[(Iterable[Double], Long)] = transFeatures.filter(v => v._1.toArray.count(_.isNaN) <= limitOfMissingFeatures)
      val avgs: RDD[(Int, Double)] = fiteringFeatures.map(v => average(v._1)).zipWithIndex().map({
        case (k, v) => (v.toInt, k)
      })
      val filteringFeaturesIndex: RDD[(Double, Int)] = fiteringFeatures.flatMap(row => row._1.zipWithIndex)
      val transFilteringFeatures: RDD[(Int, Iterable[Double])] = filteringFeaturesIndex.groupBy(_._2).map(
        row => (row._1, row._2.map(_._1))
      )

      val limitOfMissingRows: Int = fiteringFeatures.count().toInt / 4
      val filteringRows: RDD[(Int, Iterable[Double])] = transFilteringFeatures.filter(v => v._2.toArray.count(_.isNaN) <= limitOfMissingRows).sortByKey(true)
      val filteringRowsIndex: RDD[(Int, Double)] = filteringRows.flatMap(row => (row._2.zipWithIndex)).map({
        case (k, v) => (v, k)
      })

      val meanWithFilteringRowsIndex: RDD[(Double, Int)] = filteringRowsIndex.join(avgs).map({
        case (k, (v1, v2)) => (
          if (!v1.equals(Double.NaN)) {
            v1
          } else {
            v2
          }, k)
      })

      val meanCols: RDD[Iterable[Double]] = meanWithFilteringRowsIndex.groupBy(_._2).values.map(_.map(_._1))
      val meanRows: RDD[(Int, Iterable[Double])] = meanCols.flatMap(row => row.zipWithIndex).groupBy(_._2).map(
        row => (row._1, row._2.map(_._1))
      ).sortByKey(true)

      val joinRows: RDD[(Int, ((Int, Iterable[Double]), ((Int, Iterable[Double]), Long)))] = meanRows.keyBy(_._1).join(filteringRows.zipWithIndex().keyBy(_._2.toInt))
     /* val filterRows: RDD[(LabeledPoint, Long)] = joinRows.map { case row =>
        val label: Int = {
          if (groupAs.contains(row._2._2._1._1)) 1
          else if (groupBs.contains(row._2._2._1._1)) 2
          else -1
        }
        val features: Iterable[Double] = row._2._1._2
        val pos: Int = row._1
        (LabeledPoint(label, Vectors.dense(features.toArray)), pos.toLong)
      }
      (filterRows, fiteringFeatures.map(v => v._2))*/
    }


    val sc = new SparkContext(new SparkConf()
      //.setMaster("local[*]")
      .set("spark.driver.maxResultSize", "90g")
      .set("spark.akka.timeout","200000")
      .set("spark.worker.timeout","500000")
      .set("spark.storage.blockManagerSlaveTimeoutMs","5000000")
      .set("spark.akka.frameSize", "1024"))
    //val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("Test"))

    var num_features:Int =1000
    var num_samples:Int =100000
    var num_bins:Int = 200
    var num_new_partitions:Int = 5*16

    if (!args.isEmpty){

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt

    }

    val recordsPerPartition: Int = num_samples/num_new_partitions


    val noise = 0.1
    val gauss = new Gaussian(0.0,1.0)
    val weights: Array[Double] = gauss.sample(num_features).toArray
    val w = BDV(weights)
    w(3)=0.0

    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](),num_new_partitions)
      .mapPartitions { _ => {
        val gauss=new Gaussian(0.0,1.0)
        val r= scala.util.Random
        (1 to recordsPerPartition).map { _ =>
          val x = BDV(gauss.sample(num_features).toArray)
          x(3)=NaN


          val l = x.dot(w) + gauss.sample() * noise
          (1 to r.nextInt(num_features)).map(i=> x(r.nextInt(num_features))=Double.NaN)
          new LabeledPoint(l, Vectors.dense(x.toArray))
        }
      }.toIterator
      }


    //val res: Unit = filterDataSet(lds.zipWithIndex(),Array(0),Array(1))

    //lds.collect.foreach{x=>
    //  x.features.toArray.foreach(print)
    //  println}


    val x: BDV[Double] = new BDV(lds.first().features.toArray)
    println(x(3))
    val aaa: Double = x(3)
    println(x(3)==Double.NaN)
    println(Double.NaN.equals(Double.NaN))

    (BDV.fill[Double](5){1.0}:==BDV.fill[Double](5){1.0}).foreach(print)
    println()
    val x1=BDV.fill[Double](x.length) {Double.NaN}
    x1.foreach(print)
    println()
    val y =(x :== x1).toArray
    y.foreach(print)
    println()
    //val mi = new MutualInformation

    //val ffd = featureFromDataset(lds,1)
    val ffd: RDD[Array[Double]] = columnCheck2(lds)

    //ffd.collect.foreach{x=>
    //  x.foreach(print)
    //  println}

    ffd.saveAsTextFile("/tmp/output_preprocessing.txt")


    //ffd.foreach(println)

    //ffd.saveAsTextFile("feature_data.txt")
        val a: BDV[Double] =  new BDV(Array(1.0, 3, 5, 7,9,11,13,15,17))
        a.foreach(println)
        val ind: Inclusive = 2 to 4
        val b = a(Array(1,2,4,5).toList)
        b.foreach(println)


    sc.stop()



  }
}
