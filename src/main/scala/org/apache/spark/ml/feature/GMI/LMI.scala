package org.apache.spark.ml.feature.GMI

import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Utilities.Discretizing

import scala.collection.mutable

class LMI(method: String, data:Array[Array[Double]], indexes:Array[Int], b:Int) extends Serializable with Discretizing with Logging {

  private final val numFeatures = data.head.length - 1
  private final val numColumns = numFeatures + 1
  private final val Dsct = super.discAndColumnarEW(data, indexes, nb = b)

  def getSimilarity: Array[Array[Double]] = method.toUpperCase() match {
    case "QP" => MI(Dsct, b, true)
    case "SR" | "TP" => CMI(Dsct, b)
  }

  def MI(data: Array[Array[Byte]], maxBin: Int, normal: Boolean): Array[Array[Double]] = {

    val start: Long = System.currentTimeMillis()
    val nColumns = data.length
    val nInstances = data.head.length.toDouble
    val frqMatrix = Array.ofDim[Int](maxBin, maxBin)
    val entropies = mutable.Map[(Int, Int), Double]()

    var counter = 0
    val vFrequencies = Array.fill[Int](maxBin)(0)
    val mFrequencies = Array.fill[Int](maxBin* maxBin)(0)

    for (x <- 0 until nColumns; y <- (x + 1) until nColumns) {

      for (i <- 0 until maxBin; j <- 0 until maxBin) frqMatrix(i)(j) = 0
      for (i <- 0 until nInstances.toInt) frqMatrix(data(x)(i))(data(y)(i)) += 1

      counter = 0
      for (i <- 0 until maxBin; j <- 0 until maxBin) {
        mFrequencies(counter) = frqMatrix(i)(j)
        counter = counter + 1
      }
      entropies += (x, y) -> Entropy(mFrequencies, nInstances)

      if (y - x == 1) {

        counter = 0
        for (i <- 0 until maxBin) {
          var s = 0
          for (j <- 0 until maxBin) s = s + frqMatrix(i)(j)
          vFrequencies(counter) = s;
          counter = counter + 1
        }
        entropies += (x, x) -> Entropy(vFrequencies, nInstances)

        counter = 0
        for (j <- 0 until maxBin) {
          var s = 0
          for (i <- 0 until maxBin) s = s + frqMatrix(i)(j)
          vFrequencies(counter) = s;
          counter = counter + 1
        }
        entropies += (y, y) -> Entropy(vFrequencies, nInstances)
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

  def CMI(data: Array[Array[Byte]], maxBin: Int): Array[Array[Double]] = {

    val start: Long = System.currentTimeMillis()
    val nColumns = data.length
    val nFeatures: Int = nColumns - 1
    val z: Int = nFeatures
    var counter = 0
    val nInstances = data.head.length.toDouble

    val entropies = mutable.Map[(Int, Int, Int), Double]()
    val frqCube = Array.ofDim[Int](maxBin, maxBin, maxBin)

    val vFrequencies = Array.fill[Int](maxBin)(0)
    val mFrequencies = Array.fill[Int](maxBin* maxBin)(0)
    val cFrequencies = Array.fill[Int](maxBin * maxBin * maxBin)(0)

    logInfo(s"CMI -> nInstances: $nInstances")
    logInfo(s"CMI -> nFeatures: $nFeatures")

    for (x <- 0 until nFeatures; y <- (x + 1) until nFeatures) {

      for (i <- 0 until maxBin; j <- 0 until maxBin; k <- 0 until maxBin) frqCube(i)(j)(k) = 0
      for (i <- 0 until nInstances.toInt) frqCube(data(x)(i))(data(y)(i))(data(z)(i)) += 1

      counter = 0
      for (i <- 0 until maxBin; j <- 0 until maxBin; k <- 0 until maxBin) {
        cFrequencies(counter) = frqCube(i)(j)(k);
        counter = counter + 1;
      }
      entropies += (x, y, z) -> Entropy(cFrequencies, nInstances)

      counter = 0
      for (i <- 0 until maxBin; j <- 0 until maxBin) {
        var s = 0
        for (k <- 0 until maxBin) s = s + frqCube(i)(j)(k)
        mFrequencies(counter) = s
        counter = counter + 1
      }
      entropies += (x, y, y) -> Entropy(mFrequencies, nInstances)

      if (y - x == 1) {

        counter = 0
        for (i <- 0 until maxBin) {
          var s = 0
          for (j <- 0 until maxBin; k <- 0 until maxBin) s = s + frqCube(i)(j)(k)
          vFrequencies(counter) = s
          counter = counter + 1
        }
        entropies += (x, x, x) -> Entropy(vFrequencies, nInstances)

        counter = 0
        for (i <- 0 until maxBin; k <- 0 until maxBin) {
          var s = 0
          for (j <- 0 until maxBin) s = s + frqCube(i)(j)(k)
          mFrequencies(counter) = s; counter = counter + 1;
        }
        entropies += (x, z, z) -> Entropy(mFrequencies, nInstances)
      }

      if (y - x == 1 && nFeatures - y == 1) {

        counter = 0
        for (j <- 0 until maxBin) {
          var s = 0
          for (i <- 0 until maxBin; k <- 0 until maxBin) s = s + frqCube(i)(j)(k)
          vFrequencies(counter) = s; counter = counter + 1;
        }
        entropies += (y, y, y) -> Entropy(vFrequencies, nInstances)

        counter = 0
        for (k <- 0 until maxBin) {
          var s = 0
          for (i <- 0 until maxBin; j <- 0 until maxBin) s = s + frqCube(i)(j)(k)
          vFrequencies(counter) = s; counter = counter + 1;
        }
        entropies += (z, z, z) -> Entropy(vFrequencies, nInstances)


        counter = 0
        for (j <- 0 until maxBin; k <- 0 until maxBin) {
          var s = 0;
          for (i <- 0 until maxBin) s = s + frqCube(i)(j)(k)
          mFrequencies(counter) = s; counter = counter + 1;
        }
        entropies += (y, z, z) -> Entropy(mFrequencies, nInstances)
      }

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

  private def Entropy(frequencies: Array[Int], nInstances: Double): Double = {
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
}
