package algorithms

import breeze.linalg.{max, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import org.apache.spark.SparkContext
import org.apache.spark.ml.attribute.{AttributeGroup, NominalAttribute, NumericAttribute}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.scalatest.FunSuite
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd._
import org.apache.spark.sql.{DataFrame, SQLContext}


/**
  * Created by tonyp on 2016-04-26.
  */
class CrossTabulateTest extends FunSuite{
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

  def normalizeToUnit(inputs: RDD[LabeledPoint], normConst: Double): RDD[LabeledPoint] = {

    //val features: RDD[Vector] = inputs.map(_.features)
    val features: RDD[Vector] = inputs.map { v =>
      val x: BDV[Double] = new BDV(v.features.toArray)
      val y: BDV[Double] = new BDV(Array(v.label))
      new DenseVector(BDV.vertcat(x, y).toArray)
    }
    val featuresRow: RowMatrix = new RowMatrix(features)
    val Max: BDV[Double] = new BDV(featuresRow.computeColumnSummaryStatistics().max.toArray)
    val Min: BDV[Double] = new BDV(featuresRow.computeColumnSummaryStatistics().min.toArray)
    val FinalMax: BDV[Double] = max(abs(Max), abs(Min))

    features.map((v: Vector) =>{
      val y: BDV[Double] = BDV(v.toArray) :/ FinalMax
      new LabeledPoint(y(y.length-1), new DenseVector(y(0 until y.length-1).toArray))
     })
  }

  def discretizeVector(inputs:RDD[LabeledPoint],level: Int): RDD[LabeledPoint] ={
    val input: RDD[Vector] = inputs.map { v =>
      val x: BDV[Double] = new BDV(v.features.toArray)
      val y: BDV[Double] = new BDV(Array(v.label))
      new DenseVector(BDV.vertcat(x, y).toArray)
    }
    val features: RDD[Array[Double]] = input.map { vec =>
      val sorted_vec: BDV[Double] = BDV(vec.toArray.sortWith(_ < _))
      val bin_level: BDV[Int] = BDV((1 to level).toArray) * (sorted_vec.length / level)
      val pos: BDV[Double] = bin_level.map { x => sorted_vec(x - 1) }
      vec.toArray.map { x =>
        sum((BDV.fill(level)(x)
          :> pos).toArray.map(y => if (y == true) 1.0 else 0.0))
      }
    }
    features.foreach(x=>x.foreach(aa=>println(aa.toString+",")))
    features.map(y =>{
      val z = BDV(y)
      new LabeledPoint(y(y.length-1),new DenseVector( z(0 until z.length-1).toArray))
    })

  }

  test("Spark ML RandomForest Feature Importance test"){
    val sc = new SparkContext("local","LeastSquaresRegressionTest")
    val num_features = 3
    val num_samples = 1000
    val dataset: LinearExampleDataset = new LinearExampleDataset(num_samples,num_features-1,0.1)
    val lds: RDD[LabeledPoint] = sc.parallelize(dataset.labeledPoints)

    val lab = dataset.labeledPoints(0).label
    val fea = dataset.labeledPoints(0).features
    println("The Label is : " +lab.toString)
    println("The features : ")
    fea.toArray.foreach(x=> println( x.toString + ","))
    val unitdata: RDD[LabeledPoint] = normalizeToUnit(lds,1)

    println("The unitdata is : "+unitdata.first().label.toString)
    println("The unitdata is : ")
    unitdata.first().features.toArray.map(x=>println(x.toString +","))
    val discdata: RDD[LabeledPoint] = discretizeVector(unitdata,10)

    println("The discdata label : "+ discdata.first().label)
    println("The discdata features : ")
    discdata.first().features.toArray.map(x=> println(x.toString+ ","))
    val categoricalFeatures = Map.empty[Int,Int]
    val df:DataFrame = setMetadata(discdata,categoricalFeatures,0)

    df.collect.map(x=>println(x.toString + ","))

    val ct = new CrossTabulate
    val abcd = ct.crossTabulate(lds,df.columns(0),df.columns(1))

    //abcd.map(x=> println(" the ith row is : "+ x))


  }

}

