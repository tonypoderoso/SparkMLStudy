package algorithms.common

import breeze.linalg.min
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix

/**
  * Created by tonypark on 2016. 5. 30..
  */
object  PrintWrapper {
  def printArrayVector(arr: Array[Vector], str: String) = {

    println("*********************************************************")
    println(str)
    println("*********************************************************")

    arr.foreach { x =>
      println(x.toArray.toList) //.foreach(y => print(y.toString.substring(0, 4) + ","))
      //println("")
    }
    println("*********************************************************")
  }

  def RowMatrixPrint(rm: RowMatrix, str: String, rowin:Int = -1, colin:Int = -1) = {

    println("*********************************************************")
    println(str)
    println("*********************************************************")
    var row=rm.numRows.toInt
    if (rowin != -1) {
      row = rowin
    }
    var col = rm.numCols.toInt
    if (colin != -1) {
       col = colin
    }
    val tmp = BreezeConversion.toBreeze(rm)
    println(" The Matrix is of size with rows : " + row.toString + " and cols : " + col.toString)

    (0 until row).foreach { row =>
      (0 until col).foreach { col =>
        //print(tmp(row, col).toString.substring(0,min(7,tmp(row,col).toString.length)) + ", ")
        print(tmp(row, col).toString.take(6) + ", ")
      }
      println(" ; ")
    }
    println("*********************************************************")
  }


}
