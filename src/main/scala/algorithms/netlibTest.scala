package algorithms

import com.github.fommil.netlib.{ARPACK, LAPACK}
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by tonypark on 2016. 6. 4..
  */
object netlibTest{
 def main(args:Array[String]){

   val conf = new SparkConf()
     .setAppName("FirstExample")
     .set("spark.driver.maxResultSize", "90g")
   //.setMaster("local[*]")
   val sc = new SparkContext(conf)


   val arpack = ARPACK.getInstance()
   println("Tony Test " + arpack.toString)
   val lapack = LAPACK.getInstance()
   println("Tony Test " + lapack.toString)

   sc.stop()


 }
}
