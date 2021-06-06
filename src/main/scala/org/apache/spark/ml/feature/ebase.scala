package org.apache.spark.ml.feature
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators, Params}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.{ShortType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * the base class of the ensemble learning
 */
trait ebase extends base {

  //region ------- Parameters ----------------------------------------------

  val numPartitions:Param[Int] = new Param[Int](this, "numPartition", "the number of partitions for balancing label distribution", ParamValidators.gtEq(0))

  setDefault(numPartitions, 1)

  def getNumPartitions: Int = $(numPartitions)

  def setNumPartitions(num: Int): this.type = set(numPartitions, num)

  //endregion


  /**
   * @param data a data frame whose class label distribution is not uniform.
   * @return the data frame which its class labels have been distributed uniformly.
   */
  def BalanceLabelDistribution(data: => DataFrame): Dataset[Row] = {

    val np = if (this.getNumPartitions == 1) data.rdd.getNumPartitions else this.getNumPartitions
    val encoder = RowEncoder(StructType(StructField("pIndex", ShortType, nullable = false) +: data.schema))

    logInfo(s"The number of partitions for balancing the distribution of class label is set to $np")

    val ds = data.mapPartitions {
      rows =>
        val labels = collection.mutable.HashMap[Short, Array[Int]]().withDefaultValue(Array.fill[Int](np)(0))
        rows.map {
          row =>
            val cIndex = row.fieldIndex(this.getLabelCol)
            val cLabel = row.get(cIndex) match {
              case v: Double => v.toShort
              case v: Int => v.toShort
              case v: Short => v
              case v: Byte => v.toShort
            }

            val frequencies = labels.getOrElseUpdate(cLabel, Array.fill[Int](np)(0))

            val (minValue, minIndex) = frequencies.zipWithIndex.minBy(_._1)
            labels(cLabel)(minIndex) = minValue + 1

            Row.fromSeq(minIndex.toShort +: row.toSeq)
        }
    }(encoder)

    val pIndex = ds.col("pIndex")
    ds.repartitionByRange(pIndex)
  }

}
