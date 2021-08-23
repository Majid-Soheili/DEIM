package org.apache.spark.ml.feature.Relief
import org.apache.spark.ml.feature.Neighbours.{NeiManager, NerNeiHeap}

import scala.util.Random

/**
 * the ReliefF, traditional feature ranking algorithm
 * @param data, the scaled data, it is assumed that the data scaled before.
 * @param nSamples, the number of samples
 * @param nNeighbours, the number of neighbours
 * @param threshold, the threshold value to detecting nominal feature.
 * @param seed, the seed value for random number generator
 */
class ReliefF(data: Array[Array[Double]], indexes:Array[Int], nSamples:Int = 0, nNeighbours:Int = 10, val threshold:Int = 1, val seed:Int = 5341) extends NeiManager{

  private final val rand = new Random(seed)
  private final val cIndex = data.head.length - 1
  private final val nInstances: Double = indexes.length
  private final val numSamples = if (nSamples <= 0 || nSamples > nInstances) nInstances.toInt else nSamples
  private final val nFeatures: Int = data.head.length - 1
  private final val priorClass = this.getPriorProbabilityClass
  private final val nClass = priorClass.length
  private final val samples = this.getSamples
  private final val nominalFeatures = super.getNominalFeatures(data, indexes, threshold)
  private final val weights: Array[Double] = this.computeWeights

  def getWeights: Array[Double] = this.weights

  def getRanks: Array[Int] = this.getWeights.zipWithIndex.sortBy(_._1).reverse.map(_._2)

  private def computeWeights: Array[Double] = {
    val start = System.currentTimeMillis()
    val weights = Array.fill(nFeatures)(0.0)
    samples.foreach {
      idx =>

        val neighbours = Array.fill[NerNeiHeap](nClass)(new NerNeiHeap(this.nNeighbours))
        for (j <- indexes; if idx != j) {
          val cClass = data(j)(cIndex).toInt
          val dist = this.getDistanceInstances(data(idx), data(j), nominalFeatures)
          neighbours(cClass) += (j, dist)
        }

        val neighbourIndexes = neighbours.map(_.neighbourIndexes)
        val smClass = data(idx)(cIndex).toInt
        for (j <- 0 until nClass) {
          val hit = smClass == j
          val factor = if (hit) -1.0 else priorClass(j) / (1 - priorClass(smClass))
          for (k <- neighbourIndexes(j)) {
            val diff = super.getDifferentFeatures(data(idx), data(k), nominalFeatures).map(math.abs)
            (0 until nFeatures).foreach(idx => weights(idx) = weights(idx) + factor * diff(idx))
          }
        }
    }
    logInfo(s"Computing weights takes ${System.currentTimeMillis() - start} milliseconds")
    weights
  }

  private def getSamples: Array[Int] = {
    rand.shuffle(indexes.toIndexedSeq).take(this.numSamples).toArray.sorted
  }

  private def getPriorProbabilityClass: Array[Double] = {
    val prior = Array.fill[Double](255)(0.0)
    for (i <- indexes) {
      val idx = data(i)(cIndex).toInt
      prior(idx) += 1.0
    }
    val nz = prior.indexOf(0)
    for (i <- 0 until nz) prior(i) /= nInstances.toDouble
    prior.take(nz)
  }

}
