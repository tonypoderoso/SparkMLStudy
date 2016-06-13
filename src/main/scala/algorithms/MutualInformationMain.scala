package algorithms

//import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import java.io.{FileOutputStream, PrintWriter}
import java.net.URI

import breeze.linalg.{min, DenseMatrix => BDM, DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.{FutureAction, SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.{Map, mutable}
import scala.collection.mutable.ListBuffer

/**
  * Created by tonypark on 2016. 4. 19..
  */
object MutualInformationMain {

  def main(args:Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf()
      //.setMaster("local[*]")
      .setAppName("MutualINformationMain")
      .set("spark.driver.maxResultSize", "90g")
      .set("spark.akka.timeout", "200000")
      .set("spark.worker.timeout", "500000")
      .set("spark.storage.blockManagerSlaveTimeoutMs", "5000000")
      .set("spark.akka.frameSize", "1024"))
    //.set("spark.akka.heartbeat.interval","4000s")
    //.set("spark.akka.heartbeat.pauses","2000s"))
    //val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("Test"))
    var num_features: Int = 10
    var num_samples: Int = 100000
    var num_bins: Int = 200
    var num_new_partitions: Int = 5

    if (!args.isEmpty) {

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt

    }

    Logger.getLogger("org").setLevel(Level.OFF)

    // Single processing
    //************************************************
    //val dataset = new LinearExampleDataset(num_samples,num_features-1,0.1)
    //val lds: RDD[LabeledPoint] = sc.parallelize(dataset.labeledPoints,num_new_partitions)


    //val unitdata: RDD[DenseVector]= mi.normalizeToUnit(lds,1)

    //val trans: RDD[Vector] = mi.rddTranspose1(unitdata)

    //val trans = mi.normalizeToUnitwithTranspose(lds,1)


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

    val dvec: RDD[Array[Int]] = mi.discretizeVector1(trans, num_bins) //.repartition(num_new_partitions)

    //println("5. dvec : "+dvec.getNumPartitions)
    //val res = dvec.zipWithIndex()


    def computeMIMatrixRDD2(in: RDD[Array[Int]], num_features: Int, num_state1: Int,
                            num_state2: Int, num_new_partitions: Int): Array[(Long, Array[Double])] = {

      val sc = in.sparkContext
      val a1: RDD[(Array[Int], Long)] = in.zipWithIndex().cache
      val a: Array[(Array[Int], Long)] = a1.collect


      a.map { b =>
        //val b: (Array[Int], Long) = a.first()
        val onerow = sc.broadcast(b)


        val computeMutualInformation2
        = (input: ((Array[Int], Long))) => {
          val buf = new ListBuffer[(Long, Double)]()
          //val recrow: (Array[Int], Long) = onerow.value
          //val row = recrow._2
          val col = input._2
          var res: MatrixEntry = MatrixEntry(-1, -1, -1)
          if (onerow.value._2 >= col) {
            val output = BDM.zeros[Int](num_state1, num_state2)
            val rsum: BDV[Int] = BDV.zeros[Int](num_state1)
            val csum: BDV[Int] = BDV.zeros[Int](num_state2)
            val msum: Int = onerow.value._1.length
            onerow.value._1.zip(input._1).map { x =>
              output(x._1 - 1, x._2 - 1) = output(x._1 - 1, x._2 - 1) + 1
              rsum(x._1 - 1) = rsum(x._1 - 1) + 1
              csum(x._2 - 1) = csum(x._2 - 1) + 1
            }
            val MI = output.mapPairs { (coo, x) =>
              if (x > 0) {
                val tmp = msum.toDouble / (rsum(coo._1) * csum(coo._2))
                x * math.log(x * tmp) / math.log(2)
              } else
                0
            }.toArray.reduce(_ + _)
            val tmp = MI / msum

            buf.+=((col, tmp)) //MatrixEntry(row, col, tmp)
          }
          buf
        }.sorted.map(x => x._2)
        val mapper: Array[Double] = a1.flatMap(computeMutualInformation2).collect //.sorted.map(x=>x._2)
        (onerow.value._2, mapper)
      }
      //buf//reduce(_.union(_))
    }

    def computeMIMatrixRDD3(in: RDD[Array[Int]], num_features: Int, num_state1: Int,
                            num_state2: Int, num_new_partitions: Int): Array[(Long, Array[Double])] = {

      val sc = in.sparkContext
      val a1: RDD[(Array[Int], Long)] = in.zipWithIndex().cache
      val a: Array[(Array[Int], Long)] = a1.collect


      a.map { b =>
        //val b: (Array[Int], Long) = a.first()
        val onerow = sc.broadcast(b)


        val computeMutualInformation2
        = (input: ((Array[Int], Long))) => {
          val buf = new ListBuffer[(Long, Double)]()
          //val recrow: (Array[Int], Long) = onerow.value
          //val row = recrow._2
          val col = input._2
          var res: MatrixEntry = MatrixEntry(-1, -1, -1)
          if (onerow.value._2 >= col) {
            val output = BDM.zeros[Int](num_state1, num_state2)
            val rsum: BDV[Int] = BDV.zeros[Int](num_state1)
            val csum: BDV[Int] = BDV.zeros[Int](num_state2)
            val msum: Int = onerow.value._1.length
            onerow.value._1.zip(input._1).map { x =>
              output(x._1 - 1, x._2 - 1) = output(x._1 - 1, x._2 - 1) + 1
              rsum(x._1 - 1) = rsum(x._1 - 1) + 1
              csum(x._2 - 1) = csum(x._2 - 1) + 1
            }
            val MI = output.mapPairs { (coo, x) =>
              if (x > 0) {
                val tmp = msum.toDouble / (rsum(coo._1) * csum(coo._2))
                x * math.log(x * tmp) / math.log(2)
              } else
                0
            }.toArray.reduce(_ + _)
            val tmp = MI / msum

            buf.+=((col, tmp)) //MatrixEntry(row, col, tmp)
          }
          buf
        }.sorted.map(x => x._2)
        val mapper: Array[Double] = a1.flatMap(computeMutualInformation2).collect //.sorted.map(x=>x._2)
        (onerow.value._2, mapper)
      }
      //buf//reduce(_.union(_))
    }

    def computeMIMatrixRDD4(in: RDD[Array[Int]], num_features: Int, num_state1: Int,
                            num_state2: Int, num_new_partitions: Int): Array[(Long, Array[Double])] = {

      val sc = in.sparkContext
      val a1: RDD[(Array[Int], Long)] = in.zipWithIndex().persist
      val a: Array[(Array[Int], Long)] = a1.collect
      //sc.parallelize(
        a.map{ b =>
          val onerow = sc.broadcast(b)

          val computeMutualInformation2: ((Array[Int], Long)) => ListBuffer[Double]
          = (input: ((Array[Int], Long))) => {
              val buf = new ListBuffer[(Long, Double)]()
              val col = input._2
              if (onerow.value._2 > col) {
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
              }
              buf
          }.sorted.map(x => x._2)

         // val mapper = //.sorted.map(x=>x._2)
          (onerow.value._2, a1.flatMap(computeMutualInformation2).collect)
          //a1.flatMap(computeMutualInformation2)
      }//)
    }


    def computeMIVector(b: (Array[Int],Long),a1: RDD[(Array[Int], Long)],
                        num_state1: Int,  num_state2: Int): (Long, Array[Double]) = {

        val onerow = sc.broadcast(b)

        val computeMutualInformation2: ((Array[Int], Long)) => ListBuffer[Double]
        = (input: ((Array[Int], Long))) => {
          val buf = new ListBuffer[(Long, Double)]()
          val col = input._2
          if (onerow.value._2 > col) {
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
          }
          buf
        }.sorted.map(x => x._2)

        // val mapper = //.sorted.map(x=>x._2)
        (onerow.value._2, a1.flatMap(computeMutualInformation2).collect)
        //a1.flatMap(computeMutualInformation2)
    }



    //val mrmr= new mRMRFS()
    //mrmr.computemRMR(dvec,num_features,num_bins,num_bins)



/*
    val start1 =System.currentTimeMillis()
    val MIRDD2: RDD[(Long, (Long, Double))] = mi.computeMIMatrixRDD1(dvec,num_features,num_bins,num_bins,num_new_partitions)
    val MIRDD3: RDD[(Long, Iterable[(Long, Double)])] =MIRDD2.groupByKey.sortByKey()
    val MIRDD4: Array[(Long, Array[Double])] =MIRDD3.mapValues{ iter=>
      iter.toArray.sortBy(_._1).map(_._2)//.mapValues{iter=>
    }.collect() //sortByKey()// sort by i-th ind
      //.sortByKey()
//( x => Vectors.dense(x.toArray.sortBy( _._1 ).map( _._2  )) ).collect
    println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start1)/1000.0))

    val start =System.currentTimeMillis()
    val MIRDD: Array[(Long, Array[Double])] = computeMIMatrixRDD4(dvec,num_features,num_bins,num_bins,num_new_partitions)
    println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start)/1000.0))

*/

// ********************************************************************

    //println("6. MIRDD : " + MIRDD.getNumPartitions)

    //val res = MIRDD.collect()

    //println( MIRDD.length)
    //println( "Using Cartesian RDD")
    //MIRDD.map{ x =>
    //  if (x._1.isInstanceOf[Long] && !x._2.isEmpty ) {
    //    "the output is : " + x._1 + " , " + x._2.map(y => (y*100000).toString.take(min(5,y.toString.length)) + ",").reduce(_ + _) + "\n"
    //  }
    //}.foreach(println)

    //println("Using My code")
    //MIRDD4.map{ x =>
    //  "the output is : " + x._1+ " , " + x._2.map(y=>y.toString + "," ).reduce(_+_) + "\n"
    //}.foreach(println)//saveAsTextFile("/tmp/out")
    //println("%d dim %.3f seconds".format(num_features, (System.currentTimeMillis - start)/1000.0))
    /*val conf = new Configuration()

    //하둡에 저장하는 방법
    val pathString = new Path("/out.log")
    val fs = FileSystem.get(URI.create("hdfs://jnn-g07-02:8020"), conf)
    val outFile = fs.makeQualified(pathString)
    if (fs.exists(outFile)) {
      fs.delete(outFile)
    }
    val fos= fs.create(outFile)

    val res= MIRDD.map { xi =>
      //xi.count()
      val xs: Array[MatrixEntry] = xi.collect
      xs.foreach { x =>
        val str = "the output is : " + x.i + " , " + x.j + " , " + x.value +"\n"
        fos.write(str.getBytes())
      }
    }

    fos.flush()
    fos.close()
     */


    //val res1: RDD[String] = sc.parallelize(res)
    //res1.saveAsTextFile("/tmp/output_mutual_info.txt")


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
