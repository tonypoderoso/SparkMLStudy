package algorithms.common

import breeze.linalg.{DenseMatrix=>BDM}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

/**
  * Created by tonypark on 2016. 5. 30..
  */
object BreezeConversion {
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
}
