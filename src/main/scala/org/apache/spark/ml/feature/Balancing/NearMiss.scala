package org.apache.spark.ml.feature.Balancing

import org.apache.spark.ml.feature.Neighbours.NeiManager

import scala.util.Random

/**
 * NearMiss Algorithm:  Under-Sampling
 * Useful link
 *   https://www.jeremyjordan.me/imbalanced-data/
 */

class NearMiss(data:Array[ Array[Double]], NeiNumber:Int = 5, percentage:Int = 500) {

  val r = new Random(12345)

  def getNearMiss(version:Int):Array[Array[Double]] = {

    val grouped = data.map(_.last).groupBy(identity)
    val mcValue = grouped.mapValues(_.length).minBy(_._2)._1

    val minIndex = data.zipWithIndex.filter(_._1.last == mcValue).map(_._2)
    val othIndex = data.zipWithIndex.filterNot(_._1.last == mcValue).map(_._2)


    val ns = percentage / 100 * minIndex.length
    val knn = new NeiManager(data)
    val selected = if (version == 1) {
      val distances =knn.getNearestDistanceMatrix(othIndex, minIndex, NeiNumber)
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

}
