package org.apache.spark.ml.feature.Utilities

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.Row

/**
 * Local Scaling methods
 */
trait Scaling extends Logging {

  def minMaxScaling(data: Iterator[Row], maxVec: DenseVector, minVec: DenseVector, features: String, label: String): Iterator[Array[Double]] = {

    val start = System.currentTimeMillis()
    val scale = 1.0
    val numFeatures = minVec.size
    val constantOutput = 0.5
    val scaleArray = Array.tabulate(numFeatures) { i =>
      val range = maxVec(i) - minVec(i)
      // scaleArray(i) == 0 iff i-th col is constant (range == 0)
      if (range != 0) scale / range else 0.0
    }

    def scaledValue(idx:Int, v:Double):Double = {
      if (v.isNaN) 0
      else if (scaleArray(idx) != 0)
        (v - minVec(idx)) * scaleArray(idx)
      else
        constantOutput
    }

    val result = data.map {
      row =>
        val values = Array.ofDim[Double](numFeatures + 1)
        values(numFeatures) = row.getAs[Double](label)
        row.getAs[Any](features) match {
          case vv: DenseVector =>
            vv.foreachActive { case (i, v) => values(i) = scaledValue(i, v) }
          case vv: SparseVector =>
            vv.foreachActive { case (i, v) => values(i) = scaledValue(i, v) }
        }
        values
    }
    logInfo(s"MinMax scaling takes ${System.currentTimeMillis() - start} milliseconds")

    result
  }

  def minMaxScaling(data: Array[Array[Double]], maxVec: Array[Double], minVec: Array[Double]): Array[Array[Double]] = {

    val start = System.currentTimeMillis()
    val scale = 1.0
    val numFeatures = minVec.length
    val constantOutput = 0.5
    val scaleArray = Array.tabulate(numFeatures) { i =>
      val range = maxVec(i) - minVec(i)
      // scaleArray(i) == 0 iff i-th col is constant (range == 0)
      if (range != 0) scale / range else 0.0
    }

    data.foreach{
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

    logInfo(s"MinMax scaling takes ${System.currentTimeMillis() - start} milliseconds")
    data
  }
}
