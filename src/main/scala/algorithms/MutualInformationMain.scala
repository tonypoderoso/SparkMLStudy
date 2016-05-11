package algorithms

//import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector,Vector}
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
      .set("spark.driver.cores","8")
      .set("spark.driver.maxResultSize", "4g")
      .set("spark.akka.frameSize", "512"))
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
    val dataset = new LinearExampleDataset(num_samples,num_features-1,0.1)

    //println("1.dataset: "+dataset.labeledPointsSc.getNumPartitions)

    val lds: RDD[LabeledPoint] = sc.parallelize(dataset.labeledPoints,num_new_partitions)

    val mi = new MutualInformation

    //val unitdata: RDD[DenseVector]= mi.normalizeToUnit(lds,1)

    //val trans: RDD[Vector] = mi.rddTranspose1(unitdata)

    //val trans = mi.normalizeToUnitwithTranspose(lds,1)

    val ffd = mi.featureFromDataset(lds,1)
    println("2.ffd : " +ffd.getNumPartitions)

    val ffdt = mi.rddTranspose2(ffd)
    println("3.ffdt: "+ffdt.getNumPartitions)

    val trans = mi.normalizeToUnitT(ffdt)

    println("4. trans: "+trans.getNumPartitions)

    val dvec: RDD[Array[Int]] = mi.discretizeVector1(trans,num_bins)//.repartition(num_new_partitions)

    println("5. dvec : "+dvec.getNumPartitions)


    val MIRDD =mi.computeMIMatrixRDD1(dvec,num_features,num_bins,num_bins)

    println("6. MIRDD : " + MIRDD.getNumPartitions)

    //MIRDD.foreach{x=>println("r : " + x.i + " c : "+x.j + " value : "+ x.value)}

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
