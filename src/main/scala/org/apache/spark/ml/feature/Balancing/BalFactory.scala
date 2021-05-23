package org.apache.spark.ml.feature.Balancing

import org.apache.spark.ml.feature.Neighbours.NeiManager
import scala.util.Random

/**
 * the factory of balancing method, includes over-sampling and under-sampling
 * @param data, it is assumed that the data is scaled formerly
 * @param NeiNumber, the number of neighbour would be considered in methods
 * @param percentage, it is factor of 100, for example 200, 300, or 500
 * @param threshold, a threshold for detecting nominal features
 * @param seed. the seed for generating random numbers
 */

class BalFactory(data: => Array[Array[Double]], NeiNumber:Int = 5, percentage:Int = 500, threshold:Int = 1, seed:Int=12345) extends Serializable {

  private val r = new Random(seed)
  private val grouped = data.map(_.last).groupBy(identity)
  private val mcValue = grouped.mapValues(_.length).minBy(_._2)._1
  private val minIndex = data.zipWithIndex.filter(_._1.last == mcValue).map(_._2)
  private val othIndex = data.zipWithIndex.filterNot(_._1.last == mcValue).map(_._2)
  private val knn = new NeiManager(data, threshold)
  private val N = percentage / 100

  def getNearMiss(version:Int):Array[Array[Double]] = {

    val ns = N * minIndex.length
    val selected = if (version == 1) {
      val distances = knn.getNearestDistanceMatrix(othIndex, minIndex, NeiNumber)
      val dists = distances.map(a => a.sum)
      (dists zip othIndex).sortBy(_._1).take(ns).map(_._2)
    }
    else if (version == 2) {
      val distances = knn.getFurthestDistanceMatrix(othIndex, minIndex, NeiNumber)
      val dists = distances.map(a => a.sum)
      (dists zip othIndex).sortBy(_._1).take(ns).map(_._2)
    }
    else
      Array.emptyIntArray

    val indexes = (minIndex ++ selected).sorted
    indexes.map(data)
  }

  def getSmote:Array[Array[Double]] = {

    val T = minIndex.length
    val synthetic = minIndex.flatMap {
      i =>
        val nnarray = knn.getKnnHit(i, NeiNumber)
        val populate = Array.tabulate(N) { j =>
          val nn = r.nextInt(NeiNumber)
          val k = nnarray(nn)
          knn.generateSynthetic(i, k)
        }
        populate
    }
    data ++ synthetic
  }

  def getRandomUnderSampling:Array[Array[Double]] = {
    val ns = N * minIndex.length
    val noth = othIndex.length

    val selected = Array.tabulate(ns)(_ => othIndex(r.nextInt(noth)))
    val indexes = (minIndex ++ selected).sorted
    indexes.map(data)
  }
}
