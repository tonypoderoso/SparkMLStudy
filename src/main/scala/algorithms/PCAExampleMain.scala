package algorithms

import java.util.Arrays

import algorithms.common.{BreezeConversion, PrintWrapper}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, svd => brzSvd}
import breeze.stats.distributions.Gaussian
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
/**
  * Created by tonypark on 2016. 5. 30..
  */
object PCAExampleMain {

  def asBreeze(mat:Matrix): BM[Double] = {
    if (!mat.isTransposed) {
      new BDM[Double](mat.numRows, mat.numCols, mat.toArray)
    } else {
      val breezeMatrix = new BDM[Double](mat.numCols, mat.numRows, mat.toArray)
      breezeMatrix.t
    }
  }

  /**
    * Computes the top k principal components and a vector of proportions of
    * variance explained by each principal component.
    * Rows correspond to observations and columns correspond to variables.
    * The principal components are stored a local matrix of size n-by-k.
    * Each column corresponds for one principal component,
    * and the columns are in descending order of component variance.
    * The row data do not need to be "centered" first; it is not necessary for
    * the mean of each column to be 0.
    *
    * Note that this cannot be computed on matrices with more than 65535 columns.
    *
    * @param k number of top principal components.
    * @return a matrix of size n-by-k, whose columns are principal components, and
    * a vector of values which indicate how much variance each principal component
    * explains
    */

  def computePrincipalComponentsAndExplainedVariance(mat:RowMatrix,k: Int): (Matrix, Vector) = {
    val n = mat.numCols().toInt
    require(k > 0 && k <= n, s"k = $k out of range (0, n = $n]")

    val Cov = asBreeze(mat.computeCovariance()).asInstanceOf[BDM[Double]]

    val brzSvd.SVD(u: BDM[Double], s: BDV[Double], _) = brzSvd(Cov)

    val eigenSum = s.data.sum
    val explainedVariance = s.data.map(_ / eigenSum)

    if (k == n) {
      (Matrices.dense(n, k, u.data), Vectors.dense(explainedVariance))
    } else {
      (Matrices.dense(n, k, Arrays.copyOfRange(u.data, 0, n * k)),
        Vectors.dense(Arrays.copyOfRange(explainedVariance, 0, k)))
    }
  }




  def main(args: Array[String]): Unit = {
    def featureFromDataset(inputs: RDD[LabeledPoint], normConst: Double): RDD[Vector] ={
      inputs.map { v =>
        val x: BDV[Double] = new BDV(v.features.toArray)
        val y: BDV[Double] = new BDV(Array(v.label))
        Vectors.dense(BDV.vertcat(x, y).toArray)
      }
    }



      /// preprocessing /////////
      val sc = new SparkContext(new SparkConf()
        //.setMaster("local[*]").setAppName("PCAExampleTest")
        .set("spark.driver.maxResultSize", "90g")
        .set("spark.akka.timeout","200000")
        .set("spark.worker.timeout","500000")
        .set("spark.storage.blockManagerSlaveTimeoutMs","5000000")
        .set("spark.akka.frameSize", "1024"))
    //val sc = new SparkContext(new SparkConf().setMaster("local[*]"))

    var num_features:Int =1000
    var num_samples:Int =100000
    var num_bins:Int = 200
    var num_new_partitions:Int = 50

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
    //w(3)=0.0

    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](),num_new_partitions)
      .mapPartitions { _ => {
        val gauss=new Gaussian(0.0,1.0)
        val r= scala.util.Random
        (1 to recordsPerPartition).map { _ =>
          val x = BDV(gauss.sample(num_features).toArray)
          //x(3)=NaN


          val l = x.dot(w) + gauss.sample() * noise
          //(1 to r.nextInt(num_features)).map(i=> x(r.nextInt(num_features))=Double.NaN)
          new LabeledPoint(l, Vectors.dense(x.toArray))
        }
      }.toIterator
      }


      val dataRDD: RDD[Vector] = featureFromDataset(lds,1)

     /////////////////////////////


      val k = dataRDD.first().toArray.length-1
      //val pca = new PCA(k).fit(dataRDD)

      val mat = new RowMatrix(dataRDD)
      val (pc, explainedVariance) = computePrincipalComponentsAndExplainedVariance(mat,k)

      //val pca_transform: Array[Vector] = pca.transform(dataRDD).collect()
      //val mat_multiply: Array[Vector] = mat.multiply(pc).rows.collect()
      val mat_multiply: RowMatrix = mat.multiply(pc)

      val result: BDM[Double] = BreezeConversion.toBreeze(mat_multiply)

      println(result(0,0))
      println(result(1,1))
      println(result(-1,-1))

    //pw.printArrayVector(pca_transform,"PCA Transformations")
    //PrintWrapper.RowMatrixPrint(mat_multiply,"Matrix Multiplication Result")

    }
}