package org.apache.spark.ml.feature.Utilities

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.Row

/**
 * Local Scaling methods
 */
trait Scaling extends Logging {
  def minMaxScaling(data: => Iterator[Row], maxVec: DenseVector, minVec: DenseVector, features: String, label: String): Iterator[Array[Double]] = {

    val scale = 1.0
    val numFeatures = minVec.size
    val constantOutput = 0.5
    val scaleArray = Array.tabulate(numFeatures) { i =>
      val range = maxVec(i) - minVec(i)
      // scaleArray(i) == 0 iff i-th col is constant (range == 0)
      if (range != 0) scale / range else 0.0
    }
    data.map {
      row =>
        val classLabel = row.getAs[Any](label) match {
          case v: Short => v.toDouble
          case v: Double => v
          case v: Int => v.toDouble
          case v: Byte => v.toDouble
        } // class label of the current instance

        val values = row.getAs[Any](features) match {
          case v: DenseVector => v.toArray :+ classLabel
          case v: SparseVector => v.toArray :+ classLabel
        }

        var i = 0
        while (i < numFeatures) {
          if (!values(i).isNaN) {
            if (scaleArray(i) != 0) {
              values(i) = (values(i) - minVec(i)) * scaleArray(i)
            } else {
              // scaleArray(i) == 0 means i-th col is constant
              values(i) = constantOutput
            }
          }
          i += 1
        }
        values
    }
  }

  def minMaxScaling(data: => Array[Array[Double]], maxVec: Array[Double], minVec: Array[Double]): Array[Array[Double]] = {

    val scale = 1.0
    val numFeatures = minVec.length
    val constantOutput = 0.5
    val scaleArray = Array.tabulate(numFeatures) { i =>
      val range = maxVec(i) - minVec(i)
      // scaleArray(i) == 0 iff i-th col is constant (range == 0)
      if (range != 0) scale / range else 0.0
    }

    data.map {
      row =>
        var i = 0
        while (i < numFeatures) {
          if (!row(i).isNaN) {
            if (scaleArray(i) != 0) {
              row(i) = (row(i) - minVec(i)) * scaleArray(i)
            } else {
              // scaleArray(i) == 0 means i-th col is constant
              row(i) = constantOutput
            }
          }
          i += 1
        }
        row
    }
  }
}
