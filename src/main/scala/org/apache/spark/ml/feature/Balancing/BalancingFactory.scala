package org.apache.spark.ml.feature.Balancing

import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Neighbours.{FurNeiHeap, NeiManager, NerNeiHeap}

import scala.util.Random

trait BalancingFactory extends NeiManager with Logging {

  def getBalanced(method: String, data: Array[Array[Double]], NeiNumber: Int = 5, percentage: Int = 500, threshold: Int = 1, seed: Int = 12345): Array[Array[Double]] = {

    if (method.toLowerCase() == "smote") this.getSmote(data, NeiNumber, percentage, threshold, seed)
    else if (method.toLowerCase() == "nearmiss1") this.getNearMiss(data, NeiNumber, percentage, threshold, version = 1)
    else if (method.toLowerCase() == "nearmiss2") this.getNearMiss(data, NeiNumber, percentage, threshold, version = 2)
    else if (method.toLowerCase() == "baggingundersampling") this.getRandomUnderSampling(data, percentage, seed)
    else if (method.toLowerCase() == "none") data
    else Array.empty[Array[Double]]
  }

  def getNearMiss(data: Array[Array[Double]], NeiNumber: Int = 5, percentage: Int = 500, threshold: Int = 5, version: Int = 1): Array[Array[Double]] = {

    val start = System.currentTimeMillis()
    val (minIndex, othIndex) = this.getIndexes(data.map(_.last))
    val ns = minIndex.length * percentage / 100
    val nf: Int = data.head.length - 1
    val nominalFeatures = this.getNominalFeatures(data, threshold)

    val neighbours = if (version == 1) new NerNeiHeap(NeiNumber) else new FurNeiHeap(NeiNumber)
    var counter = 0
    val distances =
      for (i <- othIndex) yield {
        neighbours.clear
        val first = data(i)
        for (j <- minIndex) {
          val second = data(j)
          val dist = super.getDistanceInstances(first, second, nominalFeatures)
          neighbours += (j, dist)
        }
        counter = counter + 1
        if (counter % 1000 == 0) logInfo(s"NearMiss method counter $counter / ${othIndex.length}")
        neighbours.toArray.map(_._2).sum
      }

    val selected = (distances zip othIndex).sortBy(_._1).take(ns).map(_._2)
    val indexes = (minIndex ++ selected).sorted
    val result = indexes.map(data)
    logInfo(s"NearMiss1 method takes ${System.currentTimeMillis() - start} milliseconds")

    result
  }

  def getSmote(data: Array[Array[Double]], NeiNumber: Int = 5, percentage: Int = 500, threshold: Int = 5, seed: Int = 12345): Array[Array[Double]] = {

    val r = new Random(seed)
    val N = percentage / 100
    val start = System.currentTimeMillis()
    val (minIndex, othIndex) = this.getIndexes(data.map(_.last))
    val neighbours = new NerNeiHeap(NeiNumber)
    val nominalFeatures = this.getNominalFeatures(data, threshold)
    val nf: Int = data.head.length - 1

    val synthetic =
      for (i <- minIndex) yield {
        val first = data(i)
        for (j <- minIndex; if i != j) {
          val second = data(j)
          val dist = super.getDistanceInstances(first, second, nominalFeatures)
          neighbours += (j, dist)
        }

        val nearest = neighbours.neighbourIndexes
        val populate = Array.tabulate(N) { _ =>
          val selected = nearest(r.nextInt(nearest.length))
          val second = data(selected)
          super.generateSynthetic(first, second, nominalFeatures, seed)
        }
        populate
      }

    val xx = synthetic.flatten.toArray
    val result = data ++ xx
    logInfo(s"SMOTE method takes ${System.currentTimeMillis() - start} milliseconds")
    result
  }

  def getRandomUnderSampling(data: Array[Array[Double]], percentage: Int = 500, seed: Int = 12345): Array[Array[Double]] = {

    val r = new Random(seed)
    val (minIndex, othIndex) = this.getIndexes(data.map(_.last))
    val ns = percentage / 100 * minIndex.length
    val selected = Array.tabulate(ns)(_ => othIndex(r.nextInt(othIndex.length)))
    val indexes = (minIndex ++ selected).sorted
    indexes.map(data)
  }

  private def getIndexes(labels: Array[Double]): (Array[Int], Array[Int]) = {
    val grouped = labels.groupBy(identity)
    val mcValue = grouped.mapValues(_.length).minBy(_._2)._1
    val minIndex = labels.zipWithIndex.filter(_._1 == mcValue).map(_._2)
    val othIndex = labels.zipWithIndex.filterNot(_._1 == mcValue).map(_._2)

    logInfo(s"The number of instances belonging to the minority class: ${minIndex.length}")
    logInfo(s"The number of instances belonging to the other class: ${othIndex.length}")

    (minIndex, othIndex)
  }
}
