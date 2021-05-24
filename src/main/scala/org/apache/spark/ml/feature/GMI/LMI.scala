package org.apache.spark.ml.feature.GMI

import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Utilities.Discretizing

import scala.collection.mutable

class LMI(method: String, data: => Array[Array[Double]], maxVec: Array[Double], minVec: Array[Double], b:Int) extends Serializable with Discretizing with Logging {

  private val numFeatures = maxVec.length
  private val numColumns = numFeatures + 1
  private val Dsc = super.discEW(data, maxVec, minVec, b).toVector
  private val Dst = TransformColumnar(Dsc)

  def getSimilarity: Array[Array[Double]] = method.toUpperCase() match {
    case "QP" => MI(Dst, b, true)
    case "SR" | "TP" => CMI(Dst, b)
  }

  def MI(data: => Array[Array[Byte]], maxBin: Int, normal: Boolean): Array[Array[Double]] = {

    val start: Long = System.currentTimeMillis()
    val nColumns = data.length
    val nInstances = data.head.length.toDouble
    val frqMatrix = Array.ofDim[Int](maxBin, maxBin)
    val entropies = mutable.Map[(Int, Int), Double]()
    var fx = Array.emptyIntArray
    var fy = Array.emptyIntArray

    for (x <- 0 until nColumns; y <- (x + 1) until nColumns) {

      for (i <- 0 until maxBin; j <- 0 until maxBin) frqMatrix(i)(j) = 0
      for (i <- 0 until nInstances.toInt) frqMatrix(data(x)(i))(data(y)(i)) += 1

      entropies.put((x, y), Entropy(frqMatrix.flatten, nInstances))
      if (y - x == 1) {
        fx = frqMatrix.map(_.sum)
        entropies.put((x, x), Entropy(fx, nInstances))
        fy = frqMatrix.transpose.map(_.sum)
        entropies.put((y, y), Entropy(fy, nInstances))
      }
    }

    val miMatrix = Array.fill[Double](nColumns, nColumns)(0.0)
    var v: Double = 0.0
    for (x <- 0 until nColumns; y <- x until nColumns) {
      v = if (normal && x == y) 1.0
      else if (!normal && x == y) entropies(x, x)
      else if (normal && x != y) 2.0 * (entropies(x, x) + entropies(y, y) - entropies(x, y)) / (entropies(x, x) + entropies(y, y))
      else entropies(x, x) + entropies(y, y) - entropies(x, y)

      miMatrix(x)(y) = v
      miMatrix(y)(x) = v
    }

    logInfo(s"Computing local mutual information (LMI) takes ${System.currentTimeMillis() - start} milliseconds")
    miMatrix
  }

  def CMI(data: => Array[Array[Byte]], maxBin: Int): Array[Array[Double]] = {

    val start: Long = System.currentTimeMillis()
    val nColumns = data.length
    val nFeatures: Int = nColumns - 1
    val z: Int = nFeatures
    val nInstances = data.head.length.toDouble

    val entropies = mutable.Map[(Int, Int, Int), Double]()
    val frqCube = Array.ofDim[Int](maxBin, maxBin, maxBin)
    var fx = Array.emptyIntArray
    var fy = Array.emptyIntArray
    var fz = Array.emptyIntArray
    var fxy = Array.emptyIntArray
    var fyz = Array.emptyIntArray
    var fxz = Array.emptyIntArray

    logInfo(s"CMI -> nInstances: $nInstances")
    logInfo(s"CMI -> nFeatures: $nFeatures")

    for (x <- 0 until nFeatures; y <- (x + 1) until nFeatures) {

      //val t1 = System.currentTimeMillis()
      for (i <- 0 until maxBin; j <- 0 until maxBin; k <- 0 until maxBin) frqCube(i)(j)(k) = 0
      for (i <- 0 until nInstances.toInt) {
        frqCube(data(x)(i))(data(y)(i))(data(z)(i)) += 1
      }

      entropies.put((x, y, z), Entropy(frqCube.flatten.flatten, nInstances))

      fxy = frqCube.flatMap(m => m.map(v => v.sum))
      entropies.put((x, y, y), Entropy(fxy, nInstances))

      if (y - x == 1) {

        fx = frqCube.map(m => m.flatten.sum)
        fxz = frqCube.flatMap(m => m.transpose.map(v => v.sum))
        entropies.put((x, x, x), Entropy(fx, nInstances))
        entropies.put((x, z, z), Entropy(fxz, nInstances))
      }

      if (y - x == 1 && nFeatures - y == 1) {

        fy = frqCube.transpose.map(_.flatten.sum)
        entropies.put((y, y, y), Entropy(fy, nInstances))
        fz = frqCube.map(_.transpose).transpose.map(_.flatten.sum)
        entropies.put((z, z, z), Entropy(fz, nInstances))
        fyz = frqCube.map(_.flatten).transpose.map(_.sum)
        entropies.put((y, z, z), Entropy(fyz, nInstances))
      }
      //logDebug(s"CMI($x, $y) takes ${System.currentTimeMillis() - t1} milliseconds")
    }

    val cmiMatrix = Array.fill[Double](nFeatures, nFeatures)(0.0)
    for (x <- 0 until nFeatures; y <- x until nFeatures) {

      val v = if (x == y)
        entropies((x, x, x)) + entropies((z, z, z)) - entropies((x, z, z)) //(Hx + Hz - Hxz)
      else
        0.5 * (2 * entropies((x, y, y)) + entropies((y, z, z)) + entropies((x, z, z)) - 2 * entropies((x, y, z)) - entropies((y, y, y)) - entropies((x, x, x)))

      cmiMatrix(x)(y) = v
      cmiMatrix(y)(x) = v
    }

    logInfo(s"Computing local conditional mutual information (LCMI) takes ${System.currentTimeMillis() - start} milliseconds")
    cmiMatrix
  }

  private def Entropy(frequencies: => Array[Int], nInstances: Double): Double = {
    val entropy = frequencies.foldLeft(0.0) {
      case (acc, v) =>
        if (v == 0) acc
        else {
          val p = v / nInstances
          acc + (-1 * p * math.log(p))
        }
    } / math.log(2)
    entropy
  }

  private def TransformColumnar(data: => Vector[Array[Byte]]): Array[Array[Byte]] = {

    val start = System.currentTimeMillis()
    val numRow = data.size
    val DSc = Array.ofDim[Byte](numColumns, numRow)
    for (i <- 0 until numRow) {
      for (j <- 0 until numColumns) {
        DSc(j)(i) = data(i)(j)
      }
    }
    logInfo(s"Transform to Columnar takes ${System.currentTimeMillis() - start} milliseconds")
    DSc
  }
}
