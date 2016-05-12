package algorithms

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSuite

/**
  * Created by tony on 16. 5. 7.
  */
class mapPartitionsTest extends FunSuite {
  val sc = new SparkContext(new SparkConf().setMaster("local[4]").setAppName("mapPartitionsTest"))

  /*
  mapPartitionsWithIndex
Similar to mapPartitions, but takes two parameters. The first parameter is
 the index of the partition and the second is an iterator through all the
 items within this partition. The output is an iterator containing the list
 of items after applying whatever transformation the function encodes.

Listing Variants
def mapPartitionsWithIndex[U: ClassTag](f: (Int, Iterator[T]) =>
Iterator[U], preservesPartitioning: Boolean = false): RDD[U]

   */

  test("Test of mapPartitionsWithIndex") {
    val x = sc.parallelize(List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 3)
    def myfunc(index: Int, iter: Iterator[Int]): Iterator[String] = {
      iter.toList.map(x => index + "," + x).iterator
    }
    val out: Array[String] = x.mapPartitionsWithIndex(myfunc).collect()
    println(out.map(x => x.toString + " ; ").reduce(_ + _))
  }


  /*
mapPartitions

This is a specialized map that is called only once for each partition. The entire content of the respective partitions is available as a sequential stream of values via the input argument (Iterarator[T]). The custom function must return yet another Iterator[U]. The combined result iterators are automatically converted into a new RDD. Please note, that the tuples (3,4) and (6,7) are missing from the following result due to the partitioning we chose.


Listing Variants

def mapPartitions[U: ClassTag](f: Iterator[T] => Iterator[U], preservesPartitioning: Boolean = false): RDD[U]


 */

  test("Test of mapPartiions 1"){
    val a = sc.parallelize(1 to 9, 3)
    def myfunc[T](iter: Iterator[T]) : Iterator[(T, T)] = {
      var res = List[(T, T)]()
      var pre = iter.next
      while (iter.hasNext)
      {
        val cur = iter.next;
        res .::= (pre, cur)
        pre = cur;
      }
      res.iterator
    }
    val res: Array[(Int, Int)] = a.mapPartitions(myfunc).collect

  }

  test("Test of mapPartitions2") {
    val x = sc.parallelize(List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 3)
    def myfunc(iter: Iterator[Int]): Iterator[Int] = {
      var res = List[Int]()
      while (iter.hasNext) {
        val cur = iter.next;
        res = res ::: List.fill(scala.util.Random.nextInt(10))(cur)
      }
      res.iterator
    }
    x.mapPartitions(myfunc).collect
  }


  test("test of cartesian"){
    val x = sc.parallelize(List(1,2,3,4,5))
    val y = sc.parallelize(List(6,7,8,9,10))
    val res: Array[(Int, Int)] = x.cartesian(y).collect
    val str: String = res.map{ x =>
          "(" + x._1.toString() + "," + x._2.toString() + "), "
    }.reduce(_+_)
    println(str)
    //res0: Array[(Int, Int)] = Array((1,6), (1,7), (1,8), (1,9), (1,10), (2,6), (2,7), (2,8), (2,9), (2,10), (3,6), (3,7), (3,8), (3,9), (3,10), (4,6), (5,6), (4,7), (5,7), (4,8), (5,8), (4,9), (4,10), (5,9), (5,10))
  }


}