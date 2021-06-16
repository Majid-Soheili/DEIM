package org.apache.spark.ml.feature.Utilities

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.Row

import scala.collection.mutable

/**
 * Local Discretion methods
 */
trait Discretizing extends Logging {
  def discEW(data: Iterator[Row], maxVec: DenseVector, minVec: DenseVector, features: String, label: String, nb: Int): Iterator[Array[Byte]] = {

    val t1 = System.currentTimeMillis()
    val numFeatures = minVec.size
    val numColumns = numFeatures + 1
    val binSize = Array.tabulate(numFeatures)(i => (maxVec(i) - minVec(i)) / nb.toDouble)

    def getBin(v: Double, idx: Int): Byte = {
      var b = 0
      while (binSize(idx) != 0.0 && !((minVec(idx) + b * binSize(idx)) <= v && v < (minVec(idx) + (b + 1) * binSize(idx)))) b = b + 1
      if (b == nb)
        b = b - 1
      b.toByte
    }

    val descData = data.map {
      vector =>
        val descRow: Array[Byte] = Array.fill[Byte](numColumns)(0)
        vector.getAs[Any](features) match {
          case v: DenseVector => v.foreachActive((j, v) => descRow(j) = getBin(v, j))
          case v: SparseVector => v.foreachActive((j, v) => descRow(j) = getBin(v, j))
          case v: mutable.WrappedArray[_] =>
            for (j <- v.indices)
              v(j) match {
                case vv: Double => descRow(j) = getBin(vv, j)
                case vv: Byte => descRow(j) = getBin(vv.toDouble, j)
                case vv: String => descRow(j) = getBin(vv.toDouble, j)
              }
        }
        descRow(numColumns - 1) = vector.getAs[Any](label) match {
          case st: String => st.toByte
          case b: Byte => b
          case sh: Short => sh.toByte
          case d: Double => d.toByte
        }
        descRow
    }
    logInfo(s"Discretizing takes ${System.currentTimeMillis() - t1} milliseconds")
    descData
  }

  def discEW(data: Array[Array[Double]], maxVec: Array[Double], minVec: Array[Double], nb: Int): Array[Array[Byte]] = {

    val start = System.currentTimeMillis()
    val numFeatures = minVec.length
    val numColumns = numFeatures + 1
    val binSize = Array.tabulate(numFeatures)(i => (maxVec(i) - minVec(i)) / nb.toDouble)

    def getBin(v: Double, idx: Int): Byte = {

      require(v >= minVec(idx), s"the value must be greater than minVec, $v is not greater than ${minVec(idx)}.")

      var b = 0
      while (binSize(idx) != 0.0 && !((minVec(idx) + b * binSize(idx)) <= v && v < (minVec(idx) + (b + 1) * binSize(idx)))) b = b + 1
      if (b == nb) b = b - 1

      b.toByte
    }

    val descData = data.map {
      vector =>
        val descRow: Array[Byte] = Array.fill[Byte](numColumns)(0)
        for (j <- 0 until numFeatures) descRow(j) = getBin(vector(j), j)
        descRow(numColumns - 1) = vector.last.toByte
        descRow
    }
    logInfo(s"Discretizing takes ${System.currentTimeMillis() - start} milliseconds")
    descData
  }
}
