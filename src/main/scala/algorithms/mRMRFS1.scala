package algorithms

/**
  * Created by tony on 16. 7. 18.
  */
import breeze.linalg.{Axis, Transpose, argsort, max, min, DenseMatrix => BDM, DenseVector => BDV, argmax => brzArgmax, argmin => brzArgmin, sum => brzSum}
import breeze.linalg.{Axis, DenseMatrix, DenseVector, Transpose}
import breeze.numerics._
import breeze.stats.distributions.Gaussian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.attribute.{AttributeGroup, NominalAttribute, NumericAttribute}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.{Partitioner, SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.collection.immutable.IndexedSeq
import scala.collection.mutable.ListBuffer


class RowPartitioner(parts: Int, num_features: Int) extends Partitioner {
  override def numPartitions: Int = parts

  val num_sample: Int = (num_features / parts).asInstanceOf[Int]

  override def getPartition(key: Any): Int = (key.asInstanceOf[Long] / num_sample).asInstanceOf[Int]
}

/**
  * Created by tonypark on 2016. 6. 13..
  */
object mRMRFS1 {


  def readFromFile(sc:SparkContext,fname:String,lname:String,num_partitions:Int) = {


    val feature: RDD[String] = sc.textFile("/Users/tonypark/ideaProjects/SparkMLStudy/src/test/resources/" + fname)
    val label: RDD[String] = sc.textFile("/Users/tonypark/ideaProjects/SparkMLStudy/src/test/resources/" + lname)

    val featRDD: Array[Vector] = feature.map(line => Vectors.dense(line.split(",").map(elem => elem.trim.toDouble))).collect
    val labelRDD: Array[Double] = label.map(line => line.toDouble).collect

    sc.parallelize((0 until feature.count().toInt).map{ i =>
        new LabeledPoint(labelRDD(i),featRDD(i))
    },num_partitions)
  }

  def generateRegerssionData(sc:SparkContext, num_samples: Int ,num_features:Int,num_selection:Int,num_new_partitions:Int)
  : RDD[LabeledPoint] = {

    val recordsPerPartition: Int = num_samples / num_new_partitions

    val r = scala.util.Random
    val weights: BDV[Double] = BDV.zeros[Double](num_features - 1)
    val arr = BDV.zeros[Int](num_selection)
    var cnt = 0
    while (cnt < num_selection) {
      val idx = r.nextInt(num_features - 1)
      if (weights(idx) == 0) {
        weights(idx) = r.nextInt(100) * 2 - 100
        arr(cnt) = idx
        cnt += 1
        print((idx + 1).toString + ",")
      }
    }
    val w = weights
    val mask = w.map { i => if (i == 0) 0.0 else 1 }
    val negmask: BDV[Double] = w.map { i => if (i == 0) 1 else 0.0 }



    //val noise = 0.1
    //val gauss = new Gaussian(1.0, 1.0)
    sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitions { _ => {
        val gauss = new Gaussian(10.0, 10.0)
        (1 to recordsPerPartition).map { _ =>
          val x = BDV(gauss.sample(num_features - 1).toArray)
          val l = x.dot(w) //+ gauss.sample() * noise * 0.01
        val noise = scala.util.Random.nextDouble * 1
          val noise1: BDV[Double] = BDV((0 until num_features-1).map { _ => scala.util.Random.nextDouble * 100 }.toArray)
          val feat = Vectors.dense(((x :* mask) :+ (noise1 :* negmask)).toArray)
          new LabeledPoint(l, feat)
        }
      }.toIterator
      }
  }


  def computeMIVector(b: (Array[Int], Long), a1: RDD[(Array[Int], Long)],
                      num_state1: Int, num_state2: Int): (Long, Array[Double]) = {

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
      }.toArray.sum / msum))

    }.sorted.map(x => x._2)

    // val mapper = //.sorted.map(x=>x._2)
    (onerow.value._2, a1.flatMap(computeMutualInformation2).collect)
    //a1.flatMap(computeMutualInformation2)
  }

  def computeMIVectorWithLookup(b: (Long, Array[Int]), a1: RDD[(Long, Array[Int])],
                                num_state1: Int, num_state2: Int): (Long, Array[Double]) = {

    val sc = a1.context
    val onerow = sc.broadcast(b)

    val computeMutualInformation: ((Long, Array[Int])) => ListBuffer[Double]
    = (input: ((Long, Array[Int]))) => {
      val buf = new ListBuffer[(Long, Double)]()
      val col = input._1
      //if (onerow.value._2 > col) {
      val output = BDM.zeros[Int](num_state1, num_state2)
      val rsum: BDV[Int] = BDV.zeros[Int](num_state1)
      val csum: BDV[Int] = BDV.zeros[Int](num_state2)
      val msum: Int = onerow.value._2.length
      onerow.value._2.zip(input._2).map { x =>
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
      }.toArray.sum / msum))

    }.sorted.map(x => x._2)

    (onerow.value._1, a1.flatMap(computeMutualInformation).collect)

  }


  def computemRMR(in: RDD[Array[Int]], number_of_features: Int, num_state1: Int,
                  num_state2: Int) = {

    val sc = in.sparkContext
    val a1: RDD[(Array[Int], Long)] = in.zipWithIndex().persist
    val a: Array[(Array[Int], Long)] = a1.collect
    val num_feat = a.length
    val mRMRSorting: BDV[Int] = BDV.zeros[Int](number_of_features+1)
    val selected: BDV[Double] = BDV.ones[Double](num_feat)
    val MIScore = BDM.zeros[Double](number_of_features, num_feat)
    selected(0)=0

    // ******************* First Features : Max Relevance ******************
    val res: (Long, Array[Double]) = computeMIVector(a(0), a1, num_state1, num_state2)
    res._2(0) = 0
    val rel = BDV(res._2)
    mRMRSorting(0) = brzArgmax(res._2)
    selected(mRMRSorting(0)) = 0 //Double.MaxValue


    (1 to number_of_features).map { indx =>


      val res1: (Long, Array[Double]) = computeMIVector(a(mRMRSorting(indx - 1)), a1, num_state1, num_state2)

      MIScore(indx - 1, ::) := BDV(res1._2).t

      println("The selected vector is : " + (BDV(res1._2) :* selected).map(x => x.toString + ",").reduce(_+_))

      val redvector: BDV[Double] = brzSum(MIScore(0 until indx, ::), Axis._0).toDenseVector.map { x => x / indx }

      val relvector: BDV[Double] = ((rel - redvector) :* selected) + (Double.MaxValue * (selected-1.0))

      mRMRSorting(indx) = brzArgmax(relvector )

      selected(mRMRSorting(indx)) = 0

    }

    for (i <- 0 until MIScore.rows){
      for (j<- 0 until MIScore.cols){
        print(MIScore(i,j)+",")
      }
      println()
    }

    println("The Important features are : " + mRMRSorting.map(x => x.toString + ",").reduce(_ + _))

  }


    def computemRMR_backup(in: RDD[Array[Int]], number_of_features: Int, num_state1: Int,
                    num_state2: Int) = {

      val sc = in.sparkContext
      val a1: RDD[(Array[Int], Long)] = in.zipWithIndex().persist
      val a: Array[(Array[Int], Long)] = a1.collect
      val num_feat = a.length
      val mRMRSorting: BDV[Int] = BDV.zeros[Int](number_of_features+1)
      val selected: BDV[Double] = BDV.ones[Double](num_feat)
      val MIScore = BDM.zeros[Double](number_of_features, num_feat)
      selected(0)=0

      // ******************* First Features : Max Relevance ******************
      val res: (Long, Array[Double]) = computeMIVector(a(0), a1, num_state1, num_state2)
      res._2(0) = 0
      val rel = BDV(res._2)
      mRMRSorting(0) = brzArgmax(res._2)
      selected(mRMRSorting(0)) = 0 //Double.MaxValue


      (1 to number_of_features).map { indx =>


        val res1: (Long, Array[Double]) = computeMIVector(a(mRMRSorting(indx - 1)), a1, num_state1, num_state2)

        val temp1: Transpose[BDV[Double]] =(BDV(res1._2) :* selected).t

        MIScore(indx - 1, ::) := temp1

        println("The selected vector is : " + (BDV(res1._2) :* selected).map(x => x.toString + ",").reduce(_+_))

        val redvector: BDV[Double] = brzSum(MIScore(0 until indx, ::), Axis._0).toDenseVector.map { x => x / indx }

        val relvector: BDV[Double] = ((rel - redvector) :* selected) + (Double.MaxValue * (selected-1.0))

        mRMRSorting(indx) = brzArgmax(relvector )

        selected(mRMRSorting(indx)) = 0

      }

    println("The Important features are : " + mRMRSorting.map(x => x.toString + ",").reduce(_ + _))

  }

  def computemRMRWithLookup(in: RDD[Array[Int]], num_feat: Int, number_of_features: Int, num_state1: Int,
                            num_state2: Int) = {

    val sc = in.sparkContext
    val num_part = in.getNumPartitions


    println(" The number of partitions is :" + num_part)

    val a1: RDD[(Long, Array[Int])] = in.zipWithIndex
      .map { case (k, v) => (v, k) }
      .partitionBy(new RowPartitioner(num_part, num_feat)).persist
    val a: Array[Int] = a1.lookup(0).head
    val mRMRSorting: BDV[Int] = BDV.zeros[Int](number_of_features)
    val selected: BDV[Double] = BDV.ones[Double](num_feat)
    val MIScore = BDM.zeros[Double](number_of_features - 1, num_feat)


    // ******************* First Features : Max Relevance ******************
    val res: (Long, Array[Double]) = computeMIVectorWithLookup((0, a), a1, num_state1, num_state2)
    res._2(0) = 0
    val rel = BDV(res._2)
    mRMRSorting(0) = brzArgmax(res._2)
    selected(mRMRSorting(0)) = Double.MaxValue
    println("The selected vector is : " + selected.map(x => x.toString + ","))
    println("")
    (1 until number_of_features).map { indx =>

      val a = a1.lookup(mRMRSorting(indx - 1)).head

      val res1: (Long, Array[Double]) = computeMIVectorWithLookup((mRMRSorting(indx - 1), a), a1, num_state1, num_state2)
      res1._2(0) = 0 //Double.MaxValue

      MIScore(indx - 1, ::) := (BDV(res1._2) :* selected).t

      val redvector = brzSum(MIScore(0 until indx, ::), Axis._0).toDenseVector.map { x => x / indx }
      val relvector: BDV[Double] = rel - redvector

      mRMRSorting(indx) = brzArgmax(relvector :* selected)

      selected(mRMRSorting(indx)) = 0 //Double.MaxValue

    }
    println("The Important features are : " + mRMRSorting.map(x => x.toString + ",").reduce(_ + _))

  }


  def toBreeze(rm: RowMatrix): BDM[Double] = {
    val m = rm.numRows().toInt
    val n = rm.numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    var i = 0
    rm.rows.collect().foreach { vector =>
      vector.foreachActive { case (j, v) =>
        mat(i, j) = v
      }
      i += 1
    }
    mat
  }


  def featureFromDataset(inputs: RDD[LabeledPoint], normConst: Double): RDD[Vector] = {
    inputs.map { v =>
      val x: BDV[Double] = new BDV(v.features.toArray)
      val y: BDV[Double] = new BDV(Array(v.label.toDouble))
      Vectors.dense(BDV.vertcat(y,x).toArray)
    }
  }


  def normalizeToUnitT(inputs: RDD[Vector]): RDD[Vector] = {

    val normalized = (vec: Vector) => {
      val Max: Double = max(vec.toArray)
      val Min: Double = min(vec.toArray)
      val FinalMax: Double = max(abs(Max), abs(Min))
      Vectors.dense((BDV(vec.toArray) :/ BDV.fill(vec.toArray.length) {
        FinalMax
      }).toArray)
    }
    inputs.map(normalized)
  }



  def rddTranspose(rdd: RDD[Vector]): RDD[Vector] = {
    // Split the matrix into one number per line.
    val byColumnAndRow: RDD[(Int, (Long, Double))] = rdd.zipWithIndex.flatMap {
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number: Double, columnIndex: Int) => {
          columnIndex ->(rowIndex, number)
        }
      }
    }
    // Build up the transposed matrix. Group and sort by column index first.
    val byColumn: RDD[Iterable[(Long, Double)]] = byColumnAndRow.groupByKey.sortByKey().values
    // Then sort by row index.
    byColumn.map {
      indexedRow => Vectors.dense(indexedRow.toArray.sortBy(_._1).map(_._2))
    }
  }

  def rddTranspose2(rdd: RDD[Vector]): RDD[Vector] = {
    // Split the matrix into one number per line.
    val sc = rdd.sparkContext
    val mentry: RDD[MatrixEntry] = rdd.zipWithIndex.flatMap {
      case (row: Vector, rowIndex: Long) => row.toArray.zipWithIndex.map {
        case (number: Double, columnIndex: Int) => MatrixEntry(columnIndex, rowIndex, number)
      }
    }
    new CoordinateMatrix(mentry).toRowMatrix().rows
  }



  def discretizeVector(input: RDD[Vector], level: Int): RDD[Array[Int]] = {

    input.map { vec =>
      val sorted_vec: BDV[Double] = BDV(vec.toArray.sortWith(_ < _))

      val bin_level: BDV[Int] = BDV((1 to level).toArray) * (sorted_vec.length / level)

      val pos: BDV[Double] = bin_level.map { x => sorted_vec(x - 1) }

      vec.toArray.map { x =>
        brzSum((BDV.fill(level)(x)
          :> pos).toArray.map(y => if (y == true) 1 else 0))
      }
    }
  }



  def main(args:Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]")
      .setAppName("MutualINformationMain")
      .set("spark.driver.extraJavaOptions", "-Dcom.sun.management.jmxremote  -Dcom.sun.management.jmxremote.port=9990"+
        " -Dcom.sun.management.jmxremote.authenticate=false   -Dcom.sun.management.jmxremote.ssl=false")
      .set("spark.driver.maxResultSize", "90g")
      .set("spark.akka.timeout", "200000")
      .set("spark.akka.threads","8")
      .set("spark.worker.timeout", "500000")
      .set("spark.storage.blockManagerSlaveTimeoutMs", "5000000")
      .set("spark.akka.frameSize", "1024"))


    var num_features: Int = 100
    var num_samples: Int = 10000
    var num_bins: Int = 4
    var num_new_partitions: Int = 5
    var num_selection = 21

    if (!args.isEmpty) {

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_selection=args(3).toString.toInt
      num_new_partitions = args(4).toString.toInt

    }

    Logger.getLogger("org").setLevel(Level.ERROR)


    // Distributed processing
    //************************************************

    val lds = readFromFile(sc,"feat.csv","label.csv",num_new_partitions)
    //val lds: RDD[LabeledPoint] = generateRegerssionData(sc,num_samples,num_features,num_selection,num_new_partitions).cache()


    val ffd: RDD[Vector] = featureFromDataset(lds, 1)


    //println("2.ffd : " +ffd.getNumPartitions)

    val ffdt = rddTranspose(ffd)
    //println("3.ffdt: "+ffdt.getNumPartitions)

    val trans = normalizeToUnitT(ffdt)

    //println("4. trans: "+  trans.getNumPartitions)

    val dvec: RDD[Array[Int]] = discretizeVector(trans, num_bins)


    val start2= System.currentTimeMillis()

    computemRMR(dvec, num_selection ,num_bins,num_bins)
    //computemRMRwithSeparateLabel(labelonly,featureonly,num_selection,num_bins,num_bins)
    println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start2)/1000.0))
    ffd.saveAsTextFile("data"+start2.toString)

    println(" Number of features : " + num_features)
    println(" Number of selection : " + num_selection)
    println(" Number of partitions : " + num_new_partitions)


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

    val importances = model.featureImportances
    val mostImportantFeature: Int = importances.argmax
    println("The most important feature  is " + mostImportantFeature)

    val res= importances.toArray.zipWithIndex.sorted
    println("Importance ranking"+ importances.toArray.map(x => x.toString +","))

    println("Importance sorted ranking"+ res.map(x => x.toString +","))


   // importances.toArray.zipWithIndex.map(x => {
   //   println(x._2.toString +"th-value : " + x._1)
   // })

    println(" The resutl")
    res.map{x=>
      println("The "+x._2 +"-th features : " + x._1)
    }

    sc.stop()
  }

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
}