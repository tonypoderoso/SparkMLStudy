package algorithms

//import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import breeze.linalg.{DenseVector=>BDV}
import breeze.stats.distributions.Gaussian
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by tonypark on 2016. 4. 19..
  */
object MutualInformationMain {

  def main(args:Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf()
      //.setMaster("local[*]")
      .setAppName("MutualINformationMain")
      .set("spark.driver.maxResultSize", "90g")
      .set("spark.akka.timeout","200000")
      .set("spark.worker.timeout","500000")
      .set("spark.storage.blockManagerSlaveTimeoutMs","5000000")
      .set("spark.akka.frameSize", "1024"))
      //.set("spark.akka.heartbeat.interval","4000s")
      //.set("spark.akka.heartbeat.pauses","2000s"))
    //val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("Test"))

    var num_features:Int =100
    var num_samples:Int =100000
    var num_bins:Int = 200
    var num_new_partitions:Int = 5*16

    if (!args.isEmpty){

     num_features = args(0).toString.toInt
     num_samples = args(1).toString.toInt
     num_bins = args(2).toString.toInt
     num_new_partitions = args(3).toString.toInt

  }


    // Single processing
    //************************************************
    //val dataset = new LinearExampleDataset(num_samples,num_features-1,0.1)
    //val lds: RDD[LabeledPoint] = sc.parallelize(dataset.labeledPoints,num_new_partitions)



    //val unitdata: RDD[DenseVector]= mi.normalizeToUnit(lds,1)

    //val trans: RDD[Vector] = mi.rddTranspose1(unitdata)

    //val trans = mi.normalizeToUnitwithTranspose(lds,1)


    // Distributed processing
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

    val mi = new MutualInformation

    val ffd = mi.featureFromDataset(lds,1)
    println("2.ffd : " +ffd.getNumPartitions)

    val ffdt = mi.rddTranspose2(ffd)
    println("3.ffdt: "+ffdt.getNumPartitions)

    val trans = mi.normalizeToUnitT(ffdt)

    //println("4. trans: "+  trans.getNumPartitions)

    val dvec: RDD[Array[Int]] = mi.discretizeVector1(trans,num_bins)//.repartition(num_new_partitions)

    //println("5. dvec : "+dvec.getNumPartitions)
    //val res = dvec.zipWithIndex()

    val MIRDD =mi.computeMIMatrixRDD1(dvec,num_features,num_bins,num_bins,num_new_partitions)

    //println("6. MIRDD : " + MIRDD.getNumPartitions)

    //val res = MIRDD.collect()

    MIRDD.saveAsTextFile("/tmp/output_mutual_info.txt")

    //MIMAT.foreachPair{ (x,y)=>println(x._1 + ", " + x._2 + " --> " + y) }
    //println(MIMAT.rows)
    //println(MIMAT.cols)
    //println("///////////////////////////////////////////")

    //val mrMR=new MaxRelevanceOverMinRedundancyFeatureSelection
    //val H: DenseMatrix[Double] =MIMAT(0 until MIMAT.rows-1,0 until MIMAT.cols-1)
    //val f: BDV[Double] = MIMAT(MIMAT.rows-1,0 until MIMAT.cols-1).t
    //println("The H is a maxrix of size " +H.rows +" rows and  " + H.cols +" columns")
    //println("The f vector is of length"+ f.length)

    //val mrMRFS = mrMR.evaluate( H,f, num_features-1)
    //println("the result is: ")
    //mrMRFS.foreach(println)
    sc.stop()

  }
}
