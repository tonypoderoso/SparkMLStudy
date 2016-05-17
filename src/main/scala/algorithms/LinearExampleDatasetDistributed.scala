package algorithms

import breeze.linalg.{DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.IndexedSeq
/**
  * Created by tonypark on 2016. 5. 12..
  */
object LinearExampleDatasetDistributed {

  def main(args:Array[String]): Unit = {

    def average[T](ts: Iterable[T])(implicit num: Numeric[T]) = {
      val filteringTS = ts.filter(v => !v.asInstanceOf[Double].equals(Double.NaN))
      num.toDouble(filteringTS.sum) / filteringTS.size
    }


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
      .setMaster("local[*]")
      .setAppName("Dataset Genration")
      .set("spark.driver.maxResultSize", "4g")
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

    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](),num_new_partitions)
      .mapPartitions { _ => {
        val gauss=new Gaussian(0.0,1.0)
        (1 to recordsPerPartition).map { _ =>
          val x = BDV(gauss.sample(num_features).toArray)
          val l = x.dot(w) + gauss.sample() * noise
          new LabeledPoint(l, Vectors.dense(x.toArray))
        }
      }.toIterator
      }


    val res: Unit = filterDataSet(lds.zipWithIndex(),Array(0),Array(1))


    //val mi = new MutualInformation

    //val ffd = mi.featureFromDataset(lds,1)

    //ffd.saveAsTextFile("feature_data.txt")



    sc.stop()



  }
}
