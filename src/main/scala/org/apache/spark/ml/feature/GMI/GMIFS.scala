package org.apache.spark.ml.feature.GMI

import org.apache.spark.ml.feature.OptimizationMethods.{QP, SR, TP}

/**
 * Global Mutual Information based Feature Selector
 */
class GMIFS {

  def apply(name: String): Array[Array[Double]] => Array[Double] =
    name.toUpperCase() match {
      case "QPFS" => QPFS
      case "SRFS" => SRFS
      case "TPFS" => TPFS
    }

  private def QPFS(similarity: Array[Array[Double]]): Array[Double] = {

    val nColumns = similarity.length
    val nze = similarity.zipWithIndex.filterNot(_._1.sum.isNaN).map(_._2)
    val nzeSU = similarity.filterNot(_.sum.isNaN).map(a => a.filterNot(_ == 0))
    val ffSU = nzeSU.dropRight(1).map(row => row.dropRight(1))
    val fcSU = nzeSU.last.dropRight(1)

    val nzeWeights = QP(ffSU, fcSU).getWeights()
    val weights = Array.fill(nColumns - 1)(nzeWeights.min)
    for (i <- nze.dropRight(1).indices) weights(nze(i)) = nzeWeights(i)

    weights
  }

  private def SRFS(similarity: Array[Array[Double]]): Array[Double] = {
    SR(similarity, isSymmetric = true).getWeights
  }

  private def TPFS(similarity: Array[Array[Double]]): Array[Double] = {
    val nFeatures = similarity.length
    TP(similarity, nFeatures, isSymmetric = true).getWeights
  }
}

