package algorithms

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, GenericMutableRow}
import org.apache.spark.sql.catalyst.plans.logical.LocalRelation
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, _}
import org.apache.spark.unsafe.types.UTF8String


/**
  * Created by tonyp on 2016-04-26.
  */
class CrossTabulate {


  /** Generate a table of frequencies for the elements of two columns. */
  def crossTabulate(data: RDD[LabeledPoint], col1: String, col2: String): Seq[GenericMutableRow] = {
    val sqlContext = SQLContext.getOrCreate(data.sparkContext)
    import sqlContext.implicits._
    val df = data.toDF()
    val tableName = s"${col1}_$col2"
    val counts = df.groupBy(col1, col2).agg(count("*")).take(1e6.toInt)
    if (counts.length == 1e6.toInt) {
      //logWarning("The maximum limit of 1e6 pairs have been collected, which may not be all of " +
      // "the pairs. Please try reducing the amount of distinct items in your columns.")
    }
    def cleanElement(element: Any): String = {
      if (element == null) "null" else element.toString
    }
    // get the distinct values of column 2, so that we can make them the column names
    val distinctCol2: Map[Any, Int] =
      counts.map(e => cleanElement(e.get(1))).distinct.zipWithIndex.toMap
    val columnSize = distinctCol2.size
    require(columnSize < 1e4, s"The number of distinct values for $col2, can't " +
      s"exceed 1e4. Currently $columnSize")
    val table: Seq[GenericMutableRow] = counts.groupBy(_.get(0)).map { case (col1Item, rows) =>
      val countsRow: GenericMutableRow = new GenericMutableRow(columnSize + 1)
      rows.foreach { (row: Row) =>
        // row.get(0) is column 1
        // row.get(1) is column 2
        // row.get(2) is the frequency
        val columnIndex: Int = distinctCol2.get(cleanElement(row.get(1))).get
        countsRow.setLong(columnIndex + 1, row.getLong(2))
      }
      // the value of col1 is the first value, the rest are the counts
      countsRow.update(0, UTF8String.fromString(cleanElement(col1Item)))
      countsRow
    }.toSeq
    // Back ticks can't exist in DataFrame column names, therefore drop them. To be able to accept
    // special keywords and `.`, wrap the column names in ``.
    def cleanColumnName(name: String): String = {
      name.replace("`", "")
    }
    // In the map, the column names (._1) are not ordered by the index (._2). This was the bug in
    // SPARK-8681. We need to explicitly sort by the column index and assign the column names.
    val headerNames: Seq[StructField] = distinctCol2.toSeq.sortBy(_._2).map { r =>
      StructField(cleanColumnName(r._1.toString), LongType)
    }


    //val schema: StructType = StructType(StructField(tableName, StringType) +: headerNames)
    //val logicalPlan: LocalRelation = LocalRelation(schema.map(f => AttributeReference(f.name, f.dataType, f.nullable, f.metadata)()), table)
    //val qe = sqlContext.executePlan(logicalPlan)
    //qe.assertAnalyzed()
    //val res = new Dataset[Row](sqlContext, logicalPlan, RowEncoder(qe.analyzed.schema))
    //res.toDF()

    table
  }
}

