package algorithms

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV, axpy => brzAxpy, svd => brzSvd}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.storage.StorageLevel
import java.util.Arrays

import algorithms.common.{BreezeConversion, PrintWrapper}
import breeze.stats.distributions.Gaussian
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
/**
  * Created by tonypark on 2016. 5. 30..
  */
object SVDExampleMain {

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

def asBreeze(mat:Matrix): BM[Double] = {
    if (!mat.isTransposed) {
      new BDM[Double](mat.numRows, mat.numCols, mat.toArray)
    } else {
      val breezeMatrix = new BDM[Double](mat.numCols, mat.numRows, mat.toArray)
      breezeMatrix.t
    }
  }

def VectorToBreezeVector(v: Vector): BV[Double] =  new BDV(v.toArray)

  /**
    * Creates a Matrix instance from a breeze matrix.
    *
    * @param breeze a breeze matrix
    * @return a Matrix instance
    */
  def fromBreeze(breeze: BM[Double]): Matrix = {
    breeze match {
      case dm: BDM[Double] =>
        new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)
      case sm: BSM[Double] =>
        // Spark-11507. work around breeze issue 479.
        val mat = if (sm.colPtrs.last != sm.data.length) {
          val matCopy = sm.copy
          matCopy.compact()
          matCopy
        } else {
          sm
        }
        // There is no isTranspose flag for sparse matrices in Breeze
        new SparseMatrix(mat.rows, mat.cols, mat.colPtrs, mat.rowIndices, mat.data)
      case _ =>
        throw new UnsupportedOperationException(
          s"Do not support conversion from type ${breeze.getClass.getName}.")
    }
  }
  /**
    * Multiplies the Gramian matrix `A^T A` by a dense vector on the right without computing `A^T A`.
    *
    * @param v a dense vector whose length must match the number of columns of this matrix
    * @return a dense vector representing the product
    */
  def multiplyGramianMatrixBy(mat: RowMatrix, v: BDV[Double]): BDV[Double] = {
    val n = mat.numCols().toInt
    val vbr = mat.rows.context.broadcast(v)
    mat.rows.treeAggregate(BDV.zeros[Double](n))(
      seqOp = (U, r) => {
        val rBrz = VectorToBreezeVector(r)
        val a = rBrz.dot(vbr.value)
        rBrz match {
          // use specialized axpy for better performance
          case _: BDV[_] => brzAxpy(a, rBrz.asInstanceOf[BDV[Double]], U)
          case _: BSV[_] => brzAxpy(a, rBrz.asInstanceOf[BSV[Double]], U)
          case _ => throw new UnsupportedOperationException(
            s"Do not support vector operation from type ${rBrz.getClass.getName}.")
        }
        U
      }, combOp = (U1, U2) => U1 += U2)
  }


  /**
    * The actual SVD implementation, visible for testing.
    *
    * @param k        number of leading singular values to keep (0 &lt; k &lt;= n)
    * @param computeU whether to compute U
    * @param rCond    the reciprocal condition number
    * @param maxIter  max number of iterations (if ARPACK is used)
    * @param tol      termination tolerance (if ARPACK is used)
    * @param mode     computation mode (auto: determine automatically which mode to use,
    *                 local-svd: compute gram matrix and computes its full SVD locally,
    *                 local-eigs: compute gram matrix and computes its top eigenvalues locally,
    *                 dist-eigs: compute the top eigenvalues of the gram matrix distributively)
    * @return SingularValueDecomposition(U, s, V). U = null if computeU = false.
    */

  def computeSVD( mat:RowMatrix,
                  k: Int,
                  computeU: Boolean,
                  rCond: Double,
                  maxIter: Int,
                  tol: Double,
                  mode: String): SingularValueDecomposition[RowMatrix, Matrix] = {


    val n = mat.numCols().toInt
    require(k > 0 && k <= n, s"Requested k singular values but got k=$k and numCols=$n.")

    object SVDMode extends Enumeration {
      val LocalARPACK, LocalLAPACK, DistARPACK = Value
    }

    val computeMode = mode match {
      case "auto" =>
        if (k > 5000) {
          println(s"computing svd with k=$k and n=$n, please check necessity")
        }

        // TODO: The conditions below are not fully tested.
        if (n < 100 || (k > n / 2 && n <= 15000)) {
          // If n is small or k is large compared with n, we better compute the Gramian matrix first
          // and then compute its eigenvalues locally, instead of making multiple passes.
          if (k < n / 3) {
            SVDMode.LocalARPACK
          } else {
            SVDMode.LocalLAPACK
          }
        } else {
          // If k is small compared with n, we use ARPACK with distributed multiplication.
          SVDMode.DistARPACK
        }
      case "local-svd" => SVDMode.LocalLAPACK
      case "local-eigs" => SVDMode.LocalARPACK
      case "dist-eigs" => SVDMode.DistARPACK
      case _ => throw new IllegalArgumentException(s"Do not support mode $mode.")
    }

    // Compute the eigen-decomposition of A' * A.
    val (sigmaSquares: BDV[Double], u: BDM[Double]) = computeMode match {
      case SVDMode.LocalARPACK =>
        require(k < n, s"k must be smaller than n in local-eigs mode but got k=$k and n=$n.")
        val G = asBreeze(mat.computeGramianMatrix()).asInstanceOf[BDM[Double]]
        EigenValueDecomposition.symmetricEigs(mat, (mat,v) => G * v, n, k, tol, maxIter)
      case SVDMode.LocalLAPACK =>
        // breeze (v0.10) svd latent constraint, 7 * n * n + 4 * n < Int.MaxValue
        require(n < 17515, s"$n exceeds the breeze svd capability")
        val G = asBreeze(mat.computeGramianMatrix()).asInstanceOf[BDM[Double]]
        val brzSvd.SVD(uFull: BDM[Double], sigmaSquaresFull: BDV[Double], _) = brzSvd(G)
        (sigmaSquaresFull, uFull)
      case SVDMode.DistARPACK =>
        if (mat.rows.getStorageLevel == StorageLevel.NONE) {
          println("The input data is not directly cached, which may hurt performance if its"
            + " parent RDDs are also uncached.")
        }
        require(k < n, s"k must be smaller than n in dist-eigs mode but got k=$k and n=$n.")
        EigenValueDecomposition.symmetricEigs(mat,multiplyGramianMatrixBy, n, k, tol, maxIter)
    }

    val sigmas: BDV[Double] = brzSqrt(sigmaSquares)

    // Determine the effective rank.
    val sigma0 = sigmas(0)
    val threshold = rCond * sigma0
    var i = 0
    // sigmas might have a length smaller than k, if some Ritz values do not satisfy the convergence
    // criterion specified by tol after max number of iterations.
    // Thus use i < min(k, sigmas.length) instead of i < k.
    if (sigmas.length < k) {
     println(s"Requested $k singular values but only found ${sigmas.length} converged.")
    }
    while (i < math.min(k, sigmas.length) && sigmas(i) >= threshold) {
      i += 1
    }
    val sk = i

    if (sk < k) {
     println(s"Requested $k singular values but only found $sk nonzeros.")
    }

    // Warn at the end of the run as well, for increased visibility.
    if (computeMode == SVDMode.DistARPACK && mat.rows.getStorageLevel == StorageLevel.NONE) {
      println("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    val s = Vectors.dense(Arrays.copyOfRange(sigmas.data, 0, sk))
    val V = Matrices.dense(n, sk, Arrays.copyOfRange(u.data, 0, n * sk))

    if (computeU) {
      // N = Vk * Sk^{-1}
      val N = new BDM[Double](n, sk, Arrays.copyOfRange(u.data, 0, n * sk))
      var i = 0
      var j = 0
      while (j < sk) {
        i = 0
        val sigma = sigmas(j)
        while (i < n) {
          N(i, j) /= sigma
          i += 1
        }
        j += 1
      }
      val U: RowMatrix = mat.multiply(fromBreeze(N))
      SingularValueDecomposition(U, s, V)
    } else {
      SingularValueDecomposition(null, s, V)
    }
  }


  def main(args: Array[String]): Unit = {
    def featureFromDataset(inputs: RDD[LabeledPoint], normConst: Double): RDD[Vector] = {
      inputs.map { v =>
        val x: BDV[Double] = new BDV(v.features.toArray)
        val y: BDV[Double] = new BDV(Array(v.label))
        Vectors.dense(BDV.vertcat(x, y).toArray)
      }
    }



    /// preprocessing /////////
    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]")
      .setAppName("SVDExample")
      .set("spark.driver.maxResultSize", "90g")
      .set("spark.akka.timeout", "200000")
      .set("spark.worker.timeout", "500000")
      .set("spark.storage.blockManagerSlaveTimeoutMs", "5000000")
      .set("spark.akka.frameSize", "1024"))
    //val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("Test"))

    var num_features: Int = 1000
    var num_samples: Int = 1000
    var num_bins: Int = 200
    var num_new_partitions: Int = 5
    var num_singularvalues = num_features - 1

    if (!args.isEmpty) {

      num_features = args(0).toString.toInt
      num_samples = args(1).toString.toInt
      num_bins = args(2).toString.toInt
      num_new_partitions = args(3).toString.toInt
      num_singularvalues = args(4).toString.toInt

    }

    val recordsPerPartition: Int = num_samples / num_new_partitions


    val noise = 0.1
    val gauss = new Gaussian(0.0, 1.0)
    val weights: Array[Double] = gauss.sample(num_features).toArray
    val w = BDV(weights)
    //w(3)=0.0

    val lds: RDD[LabeledPoint] = sc.parallelize(IndexedSeq[LabeledPoint](), num_new_partitions)
      .mapPartitions { _ => {
        val gauss = new Gaussian(0.0, 1.0)
        val r = scala.util.Random
        (1 to recordsPerPartition).map { _ =>
          val x = BDV(gauss.sample(num_features).toArray)
          //x(3)=NaN


          val l = x.dot(w) + gauss.sample() * noise
          //(1 to r.nextInt(num_features)).map(i=> x(r.nextInt(num_features))=Double.NaN)
          new LabeledPoint(l, Vectors.dense(x.toArray))
        }
      }.toIterator
      }


    val dataRDD: RowMatrix = new RowMatrix(featureFromDataset(lds, 1))

    /**
      * The actual SVD implementation, visible for testing.
      *
      * @param k        number of leading singular values to keep (0 &lt; k &lt;= n)
      * @param computeU whether to compute U
      * @param rCond    the reciprocal condition number
      * @param maxIter  max number of iterations (if ARPACK is used)
      * @param tol      termination tolerance (if ARPACK is used)
      * @param mode     computation mode (auto: determine automatically which mode to use,
      *                 local-svd: compute gram matrix and computes its full SVD locally,
      *                 local-eigs: compute gram matrix and computes its top eigenvalues locally,
      *                 dist-eigs: compute the top eigenvalues of the gram matrix distributively)
      * @return SingularValueDecomposition(U, s, V). U = null if computeU = false.
      */

    val svd = computeSVD(dataRDD , num_singularvalues, true, 1e-6,
                        300, 1e-6,"dist-eigs")

    val U: RowMatrix = svd.U
    val s: Vector = svd.s
    val V: Matrix = svd.V

    val res = toBreeze(U)

    PrintWrapper.RowMatrixPrint(U,"U Matrix")




    //U.rows.saveAsTextFile("/SVD_Result")

  }
}