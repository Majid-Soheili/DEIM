package org.apache.spark.ml.feature.Neighbours

import org.apache.spark.internal.Logging

import scala.util.Random

/**
 * Neighbours Manager: Find nearest or furthest neighbours
 * data: It is assumed that the data is scaled formerly. It is necessary for computing euclidean distance.
 * threshold: It is a threshold for detecting nominal features.
 */
trait NeiManager extends Serializable with Logging {

  final def getKnn(data: Array[Array[Double]], nominalFeatures: Array[Boolean], idx: Int, nn: Int): Array[Int] = {

    val neighbours = new NerNeiHeap(nn)
    for (j <- data.indices; if idx != j) {
      val dist = this.getDistanceInstances(data(idx), data(j), nominalFeatures)
      neighbours += (j, dist)
    }
    if (neighbours.size < nn)
      throw new Exception("There is not enough neighbours")

    neighbours.neighbourIndexes
  }

  final def getKnnHit(data: Array[Array[Double]], nominalFeatures: Array[Boolean], idx: Int, nn: Int): Array[Int] = {
    val neighbours = new NerNeiHeap(nn)
    for (j <- data.indices; if idx != j && data(idx).last == data(j).last) {
      val dist = this.getDistanceInstances(data(idx), data(j), nominalFeatures)
      neighbours += (j, dist)
    }
    if (neighbours.size < nn)
      throw new Exception("There is not enough neighbours")

    neighbours.neighbourIndexes
  }

  def getKnnMiss(data: Array[Array[Double]], nominalFeatures: Array[Boolean], idx: Int, nn: Int): Array[Int] = {
    val neighbours = new NerNeiHeap(nn)
    for (j <- data.indices; if idx != j && data(idx).last != data(j).last) {
      val dist = this.getDistanceInstances(data(idx), data(j), nominalFeatures)
      neighbours += (j, dist)
    }
    if (neighbours.size < nn)
      throw new Exception("There is not enough neighbours")

    neighbours.neighbourIndexes
  }

  def getKnnByClass(data: Array[Array[Double]], nominalFeatures: Array[Boolean], idx: Int, neiNum: Int, clsNum: Int): Array[Array[Int]] = {

    val cIndex = data.head.length - 1
    val neighbours = Array.fill[NerNeiHeap](clsNum)(new NerNeiHeap(neiNum))
    for (j <- data.indices; if idx != j) {
      val cClass = data(j)(cIndex).toInt
      val dist = this.getDistanceInstances(data(idx), data(j), nominalFeatures)
      neighbours(cClass) += (j, dist)
    }

    for (k <- 0 until clsNum; if neighbours(k).size < neiNum)
      throw new Exception("There is not enough")

    neighbours.map(_.neighbourIndexes)
  }

  def generateSynthetic(first: Array[Double], second: Array[Double], nominalFeatures: Array[Boolean], seed: Int): Array[Double] = {

    val nFeatures = first.length - 1
    val diff = getDifferentFeatures(second, first, nominalFeatures)
    val r = new Random(seed)
    val gap = r.nextDouble()
    val syntheticValues = Array.tabulate(nFeatures) {
      i =>
        if (nominalFeatures(i)) {
          if (r.nextBoolean()) first(i) else second(i)
          //math.round(data(first)(i) + gap * diff(i))
        } else
          first(i) + gap * diff(i)
    }
    syntheticValues :+ first.last
  }

  def getNominalFeatures(data: Array[Array[Double]], threshold: Int = 1): Array[Boolean] = {

    val start = System.currentTimeMillis()
    val nFeatures: Int = data.head.length - 1
    val distinctValues = Array.fill[Set[Double]](nFeatures)(Set.empty)
    for (i <- data.indices) {
      for (j <- 0 until nFeatures) {
        if (distinctValues(j).size < threshold)
          distinctValues(j) += data(i)(j)
      }
    }
    val result = distinctValues.map(_.size < threshold)
    logInfo(s"Finding nominal features takes ${System.currentTimeMillis() - start} milliseconds")
    logInfo("The number of nominal feature:" + result.count(_ == true))
    result
  }

  def getDistanceInstances(first: Array[Double], second: Array[Double], nominalFeatures: Array[Boolean]): Double = {

    val diff = getDifferentFeatures(first, second, nominalFeatures)
    var squaredDistance = 0.0
    for (i <- diff.indices) squaredDistance += diff(i) * diff(i)
    val result = math.sqrt(squaredDistance)
    result
  }

  def getDifferentFeatures(first: Array[Double], second: Array[Double], nominalFeatures: Array[Boolean]): Array[Double] = {

    require(first.length == second.length, s"Vector dimensions do not match, : Dim(v1)=${first.length} and Dim(v2)=${second.length}.")

    val cIndex = first.length - 1
    val nFeatures: Int = first.length - 1
    val diff = Array.fill[Double](nFeatures)(0.0)
    for (i <- first.indices; if i != cIndex) {
      if (nominalFeatures(i)) {
        diff(i) = if (first(i) == second(i)) 0 else 1
      }
      // numerical features
      else
        diff(i) = first(i) - second(i)
    }
    diff
  }
}
