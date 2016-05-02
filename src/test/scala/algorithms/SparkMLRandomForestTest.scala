package algorithms

//import org.apache.spark
import org.apache.spark.SparkContext
import org.apache.spark.ml.attribute.{AttributeGroup, NominalAttribute, NumericAttribute}
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


  test("Spark ML RandomForest Feature Importance test"){
    val sc = new SparkContext("local","LeastSquaresRegressionTest")
    val num_features = 1000
    val num_samples = 100000
    val dataset: LinearExampleDataset = new LinearExampleDataset(num_samples,num_features-1,0.1)


    //println("//////////////////////////////")
    //dataset.labeledPoints.map(_.label).take(10).foreach(println)
    //dataset.labeledPoints.map(_.features).take(10).foreach(println)
    // println("//////////////////////////////")

    val lds: RDD[LabeledPoint] = sc.parallelize(dataset.labeledPoints)
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
