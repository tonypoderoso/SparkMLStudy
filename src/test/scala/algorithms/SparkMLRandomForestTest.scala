package algorithms

//import org.apache.spark
import breeze.linalg.{DenseVector=>BDV}
import breeze.stats.distributions.Gaussian
import org.apache.spark.SparkContext
import org.apache.spark.ml.attribute.{AttributeGroup, NominalAttribute, NumericAttribute}
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.FunSuite
//import org.apache.spark.ml.param
import org.apache.spark.ml.regression._
//import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}

/**
  * Created by tonyp on 2016-04-26.
  */
class SparkMLRandomForestTest extends FunSuite{
  def setMetadata(
                   data: RDD[LabeledPoint],
                   categoricalFeatures: Map[Int, Int],
                   numClasses: Int): DataFrame = {
    val sqlContext = SQLContext.getOrCreate(data.sparkContext)
    import sqlContext.implicits._
    val df = data.toDF()


    val numFeatures = data.first().features.size
    val featuresAttributes = Range(0, numFeatures).map { feature =>
      if (categoricalFeatures.contains(feature)) {
        NominalAttribute.defaultAttr.withIndex(feature).withNumValues(categoricalFeatures(feature))
      } else {
        NumericAttribute.defaultAttr.withIndex(feature)
      }
    }.toArray
    val featuresMetadata = new AttributeGroup("features", featuresAttributes).toMetadata()
    val labelAttribute = if (numClasses == 0) {
      NumericAttribute.defaultAttr.withName("label")
    } else {
      NominalAttribute.defaultAttr.withName("label").withNumValues(numClasses)
    }
    val labelMetadata = labelAttribute.toMetadata()
    df.select(df("features").as("features", featuresMetadata),
      df("label").as("label", labelMetadata))
  }


  // 1000 --> 5min
  // 10000 --> 1h3min

  test("Spark ML RandomForest Feature Importance test"){
    val sc = new SparkContext("local","LeastSquaresRegressionTest")
    val num_features = 50000
    val num_samples = 100000
    val num_new_partitions = 50

    // Distributed Data generation
    //************************************************
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


    val rf= new RandomForestRegressor()
      .setImpurity("variance")
      //.setMaxDepth(3)
      //.setNumTrees(100)
      //.setFeatureSubsetStrategy("all")
      //.setSubsamplingRate(1.0)
      .setSeed(123)

    val categoricalFeatures = Map.empty[Int,Int]
    val df:DataFrame = setMetadata(lds,categoricalFeatures,0)
    val model = rf.fit(df)

    val importances: Vector = model.featureImportances
    val mostImportantFeature: Int = importances.argmax
    println("The most important feature  is " + mostImportantFeature)
    println("Importance ranking"+ importances.toArray.map(x => x.toString +","))
    importances.toArray.zipWithIndex.map(x => {
      println(x._2.toString +"th-value : " + x._1)
      })



  }

}
