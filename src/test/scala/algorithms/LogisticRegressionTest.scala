package algorithms

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.scalatest._
/**
  * Created by tony on 16. 4. 17.
  */
class LogisticRegressionTest extends FunSuite{
  test("logistic regression test"){
    val sc=new SparkContext("local", "LogisticRegressionTest")
    val rdd: RDD[Seq[Int]] = sc.parallelize(Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8, 9)))

    val byColumnAndRow: RDD[(Int, (Long, Int))] = rdd.zipWithIndex.flatMap {
      case (row: Seq[Int], rowIndex) => row.zipWithIndex.map {
        case (number, columnIndex) => columnIndex -> (rowIndex, number)
      }
    }
    // Build up the transposed matrix. Group and sort by column index first.
    val byColumn = byColumnAndRow.groupByKey.sortByKey().values
    // Then sort by row index.
    val transposed = byColumn.map {
      indexedRow => indexedRow.toSeq.sortBy(_._1).map(_._2)
    }

    val data= MLUtils.loadLibSVMFile(sc,"/Users/tonypark/rcv1_train.binary.bz2")

    println("///////////////////////////////////////")
    val lb=data.map(_.label).take(10)
    lb.foreach(println)
    val ft=data.map(_.features).take(10)
    ft.foreach(println)

    println("///////////////////////////////////////")
    //println(data.count())

    //println(data.map(_.label).collect().toSet)

    // convert to 1 and 0
    val data2= data.map(p=>new LabeledPoint(if (p.label == 1.0)1.0 else 0.0,p.features))

    //println(data2.map(_.label).take(10)
    data2.cache()

    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(data2)

    println(model.numFeatures)
    println(model.predict(data2.first().features))

    val labels = data2.map(p=>(model.predict(p.features),p.label))

    val error = labels.map(yy=>if( yy._1==yy._2) 0.0 else 1.0).reduce(_+_)/labels.count()
    println(error)

    val metrics = new BinaryClassificationMetrics(labels)


    println(metrics.areaUnderROC())


    val trainingTestSplit = data2.randomSplit(Array(0.8,0.2))
    val train = trainingTestSplit(0)
    val test = trainingTestSplit(1)




    def logreg(reg:Double) = {
      val l= new LogisticRegressionWithLBFGS();
      l.optimizer.setRegParam(reg)
      l
    }

    def evaluate(model: LogisticRegressionModel,data: RDD[LabeledPoint]): BinaryClassificationMetrics= {
      val labels = data.map(p => (model.predict(p.features), p.label))
      new BinaryClassificationMetrics(labels)
    }



    train.cache()

    val result = {for (r<- Seq(0.01,0.1, 1.0,10.0,100.0)) yield (evaluate(logreg(r).setNumClasses(2).run(train),test).areaUnderROC)}

    result.foreach(println)


  }

}
