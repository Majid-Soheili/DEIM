package org.apache.spark.ml.feature.Neighbours

import org.apache.spark.internal.Logging

import scala.util.Random

/**
 * Neighbours Manager: Find nearest or furthest neighbours
 * //@param data: It is assumed that the data is scaled formerly. It is necessary for computing euclidean distance.
 * //@param threshold: It is a threshold for detecting nominal features.
 */
trait NeiManager extends Serializable with Logging {

  def data: Array[Array[Double]]
  def threshold:Int = 1
  def seed:Int = 5341
  private val r = new Random(seed)
  private final val cIndex = data.head.length - 1
  private final val nFeatures: Int = data.head.length - 1
  private final val nominalFeatures = this.getNominalFeatures

  final def getKnn(idx: Int, nn: Int): Array[Int] = {

    val neighbours = new NerNeiHeap(nn)
    for (j <- data.indices; if idx != j) {
      val dist = this.distanceInstances(data(idx), data(j))
      neighbours += (j, dist)
    }
    if (neighbours.size < nn)
      throw new Exception("There is not enough neighbours")

    neighbours.neighbourIndexes
  }

  final def getKnnHit(idx: Int, nn: Int): Array[Int] = {
    val neighbours = new NerNeiHeap(nn)
    for (j <- data.indices; if idx != j && data(idx).last == data(j).last) {
      val dist = this.distanceInstances(data(idx), data(j))
      neighbours += (j, dist)
    }
    if (neighbours.size < nn)
      throw new Exception("There is not enough neighbours")

    neighbours.neighbourIndexes
  }

  def getKnnMiss(idx: Int, nn: Int): Array[Int] = {
    val neighbours = new NerNeiHeap(nn)
    for (j <- data.indices; if idx != j && data(idx).last != data(j).last) {
      val dist = this.distanceInstances(data(idx), data(j))
      neighbours += (j, dist)
    }
    if (neighbours.size < nn)
      throw new Exception("There is not enough neighbours")

    neighbours.neighbourIndexes
  }

  def getKnnByClass(idx: Int, neiNum: Int, clsNum: Int): Array[Array[Int]] = {

    val neighbours = Array.fill[NerNeiHeap](clsNum)(new NerNeiHeap(neiNum))
    for (j <- data.indices; if idx != j) {
      val cClass = data(j)(cIndex).toInt
      val dist = this.distanceInstances(data(idx), data(j))
      neighbours(cClass) += (j, dist)
    }

    for (k <- 0 until clsNum; if neighbours(k).size < neiNum)
      throw new Exception("There is not enough")

    neighbours.map(_.neighbourIndexes)
  }

  def generateSynthetic(first: Int, second: Int): Array[Double] = {
    val diff = differentFeatures(data(second), data(first))
    val gap = r.nextDouble()
    val syntheticValues = Array.tabulate(nFeatures) {
      i =>
        if (this.nominalFeatures(i))
          math.round(data(first)(i) + gap * diff(i))
        else
          data(first)(i) + gap * diff(i)
    }
    syntheticValues :+ data(first)(cIndex)

  }

  def getNearestDistanceMatrix(first: Array[Int], second: Array[Int], nerNei: Int): Array[Array[Double]] = getDistanceMatrix(first, second, nerNei, "nearest")

  def getFurthestDistanceMatrix(first: Array[Int], second: Array[Int], furNei: Int): Array[Array[Double]] = getDistanceMatrix(first, second, furNei, "furthest")

  def getDifferentFeatures(first:Int, second:Int): Array[Double] =  this.differentFeatures( this.data(first), this.data(second))

  private def getDistanceMatrix(first: Array[Int], second: Array[Int], NeiNumber: Int, kind: String): Array[Array[Double]] = {

    val nr = first.length
    val nc = second.length
    val neighbours =
      if (kind == "nearest")
        Array.fill[NerNeiHeap](nr)(new NerNeiHeap(NeiNumber))
      else
        Array.fill[FurNeiHeap](nr)(new FurNeiHeap(NeiNumber))

    for (i <- 0 until nr; j <- 0 until nc) {
      val dist = this.distanceInstances(data(first(i)), data(second(j)))
      neighbours(i) += (j, dist)
    }
    neighbours.map(nn => nn.toArray.map(_._2))
  }

  private def getNominalFeatures: Array[Boolean] = {

    val start = System.currentTimeMillis()
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

  private def distanceInstances(first: => Array[Double], second: => Array[Double]): Double = {

    val diff = differentFeatures(first, second)
    var squaredDistance = 0.0
    for (i <- diff.indices) squaredDistance += math.pow(diff(i), 2)
    math.sqrt(squaredDistance)
  }

  private def differentFeatures(first: => Array[Double], second: => Array[Double]): Array[Double] = {

    require(first.length == second.length, s"Vector dimensions do not match: Dim(v1)=${first.length} and Dim(v2)=${second.length}.")
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
