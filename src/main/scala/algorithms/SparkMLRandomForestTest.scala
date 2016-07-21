package algorithms

//import org.apache.spark
import breeze.linalg.{DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import org.apache.spark.ml.attribute.{AttributeGroup, NominalAttribute, NumericAttribute}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
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
object SparkMLRandomForestTest {  //{extends FunSuite{
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

  //test("Spark ML RandomForest Feature Importance test"){

  def main(args:Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]")
      .setAppName("Random Forest")
      .set("spark.driver.maxResultSize", "40g")
      .set("spark.akka.timeout","20000")
      .set("spark.worker.timeout","50000")
      .set("spark.storage.blockManagerSlaveTimeoutMs","500000")
      .set("spark.akka.frameSize", "1024"))
    //.set("spark.akka.heartbeat.interval","4000s")
    //.set("spark.akka.heartbeat.pauses","2000s"))
    //val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("Test"))

    var num_features:Int =100
    var num_samples:Int =100000
    var num_bins:Int = 20
    var num_new_partitions:Int = 5
    var num_selection=10

    if (!args.isEmpty){

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt

    }
    // Distributed Data generation
    //************************************************
    val recordsPerPartition: Int = num_samples / num_new_partitions

    val r = scala.util.Random
    val weights: BDV[Double] =BDV.zeros[Double](num_features-1)
    val arr=BDV.zeros[Int](num_selection)
    var cnt = 0
    while (cnt < num_selection) {
      val idx = r.nextInt(num_features-1)
      if (weights(idx) == 0){
        weights(idx) = r.nextInt(100)*2 - 100
        arr(cnt)=idx
        cnt += 1
        print(idx + ",")
      }
    }
    val w = weights
    val mask =w.map{ i=>if (i==0) 0.0 else 1}
    val negmask = w.map{i=>if (i==0) 1 else 0.0}



    //val noise = 0.1
    //val gauss = new Gaussian(1.0, 1.0)
    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitions { _ => {
        val gauss = new Gaussian(10.0, 10.0)
        (1 to recordsPerPartition).map { _ =>
          val x = BDV(gauss.sample(num_features-1).toArray)
          val l = x.dot(w) //+ gauss.sample() * noise * 0.01
        val noise = scala.util.Random.nextDouble*1
          val noise1 =BDV( (0 until num_features).map{ _ => scala.util.Random.nextDouble*100}.toArray)
          val feat = Vectors.dense(( ( x:* mask) :+ (noise * negmask)).toArray )
          new LabeledPoint( l , feat )
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


    sc.stop()

  }

}