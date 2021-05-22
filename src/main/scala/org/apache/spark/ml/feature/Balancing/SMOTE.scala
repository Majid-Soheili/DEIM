package org.apache.spark.ml.feature.Balancing

import org.apache.spark.ml.feature.Neighbours.NeiManager

import scala.util.Random

/**
 * Synthetic Minority Oversampling Technique – Oversampling
 * Useful links:
 *  https://github.com/Stardust1225/SmoteColt/blob/master/src/SmoteColt.java
 *  https://github.com/Lif3line/MATLAB_SMOTE/blob/master/SMOTE.m
 *  https://arxiv.org/pdf/1106.1813.pdf
 */
class SMOTE(data:Array[ Array[Double]], nearNei:Int = 5, percentage:Int = 500) {

  val r = new Random(12345)

  def getSmoteData:Array[Array[Double]] = {

    val grouped = data.map(_.last).groupBy(identity)
    val mcValue = grouped.mapValues(_.length).minBy(_._2)._1

    val needSMOTE = data.filter(a => a.last == mcValue)
    val others = data.filterNot(a => a.last == mcValue)
    val newData = doSMOTE(needSMOTE)
    others ++ newData
  }
  private def doSMOTE(data: => Array[Array[Double]]):Array[Array[Double]] = {

    val T = data.length
    val N = percentage / 100
    val knn = new NeiManager(data, 3)

    val synthetic = Array.tabulate(T) {
      i =>

        val nnarray = knn.getKnn(i, nearNei) //  Compute k nearest neighbors for i, and save the indices in the nnarray
        val populate = Array.tabulate(N) { j =>
          val nn = r.nextInt(nearNei)
          val k = nnarray(nn)
          knn.generateSynthetic(i, k)
        }
        populate
    }.flatten
    data ++ synthetic
  }
}
