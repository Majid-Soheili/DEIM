package org.apache.spark.ml.feature.Fusion

import org.apache.commons.math3.distribution.{BetaDistribution, NormalDistribution}

object RankFusion {

  def perform[A](input: Array[Seq[A]], N: Int = 0, method: String = "RRA", full: Boolean = false, exact: Boolean = false, topCutoff: Array[Double] = Array.emptyDoubleArray, risk: Double = 0.95): (Seq[A], Seq[Double]) = {

    val validateMethods = Array("mean", "min", "median", "geom.mean", "RRA", "stuart", "owa")

    if (!validateMethods.contains(method)) throw new Exception("method should be one of: 'mean', 'min', 'median', 'geom.mean', 'RRA', 'stuart' or  'OWA'")

    val common = input.flatMap(_.toSeq).distinct
    val space = common.sortWith((x, y) => x.toString < y.toString)

    val defValue = if (!full) 1.0 else Double.NaN
    val rmat = Array.fill[Double](common.length, input.length)(defValue)

    for (i <- input.indices) {
      for (j <- input(i).indices) {
        val v = input(i)(j)
        val idx = space.indexOf(v)
        val n = if (N != 0) N else if (!full) common.length else input(i).length
        rmat(idx)(i) = (j + 1) / n.toDouble
      }
    }

    val tradeOff = 4 * risk * (1 - risk)
    val owg = if (method == "owa") OWG.getWeights(input.length, risk, tradeOff) else Array.emptyDoubleArray

    val result =
      if (method == "min")
        rmat.map(arr => arr.min)
      else if (method == "median")
        rmat.map(arr => arr.sorted.drop(arr.length / 2).head)
      else if (method == "geom.mean")
        rmat.map(arr => math.exp(arr.map(x => math.log(x)).sum / arr.length))
      else if (method == "RRA")
        rmat.map(arr => rhoScores(arr, topCutoff = topCutoff, exact = exact))
      else if (method == "stuart")
        stuart(rmat)
      else if (method == "owa")
        rmat.map(arr => owg.zip(arr.sorted.reverse).map { case (item, weight) => item * weight }.sum)
      else if (method == "mean")
        rmat.map {
          arr =>
            val n = arr.length
            val a = arr.sum / arr.length
            val sd = math.sqrt(1.0 / 12.0 / n)
            new NormalDistribution(0.5, sd).cumulativeProbability(a)
        }
      else Array.emptyDoubleArray


    val score = result.sorted
    val order = result.zipWithIndex.sortBy(x => x._1).map(_._2)
    val ranks = order.map(x => space(x))

    (ranks, score)
  }

  //# Stuart-Aerts method helper functions
  private def sumStuart(v: Array[Double], r: Double) = {

    val k = v.length
    val ones = Array.tabulate(k) { p => math.pow(-1, p + 2) }

    var f = 1
    val fact = Array.tabulate(k) { i => f = f * (i + 1); f }
    //val p = Array.tabulate(k) { i => math.pow(r(i), i + 1) }
    val p = Array.tabulate(k) { i => math.pow(r, i + 1) }
    val r1 = (v.reverse, p, fact).zipped.map((v1, v2, v3) => v1 * v2 / v3)

    r1.zip(ones).map { case (v1, v2) => v1 * v2 }.sum
    //(r1 zip ones)
  }

  private def qStuart(r: Array[Double]) = {

    val N = r.count(_.isNaN == false)
    val v = Array.fill[Double](N + 1)(1.0)
    for (k <- 0 until N)
      v(k + 1) = sumStuart(v.take(k + 1), r(N - k - 1))

    (1 to N).product * v.last
  }

  private def stuart(rmat: Array[Array[Double]]): Array[Double] = rmat.map(a => a.sorted).map(r => qStuart(r))

  private def betaScores(r: Array[Double]) = {
    val n = r.count(_.isNaN == false)
    val p = Array.fill[Double](n)(1.0)
    val r2 = r.sorted
    Array.tabulate[Double](n) { i => new BetaDistribution(i + 1, n - i).cumulativeProbability(r2(i)) }

  }

  private def thresholdBetaScore(r: Array[Double], kk: Array[Int] = Array.emptyIntArray, nn: Int = 0, sig: Array[Double] = Array.emptyDoubleArray): Array[Double] = {

    val k = if (kk.isEmpty) Array.tabulate(r.length) { i => i } else kk
    val n = if (nn == 0) r.length else nn
    var sigma = if (sig.isEmpty) Array.fill[Double](n)(1.0) else sig

    if (sigma.length != n) throw new Exception("The length of sigma does not match n")
    if (r.length != n) throw new Exception("The length of pvalues does not match n")
    if (sigma.min < 0 || sigma.max > 1) throw new Exception("Elements of sigma are not in the range [0,1]")
    if (r.zip(sigma).map(vv => !vv._1.isNaN && vv._2 > vv._1).count(_ == true) > 0) throw new Exception("Elements of r must be smaller than elements of sigma")
    val x = r.filter(_.isNaN == false).sorted
    sigma = sigma.sorted.reverse
    val betha = Array.fill[Double](k.length)(Double.NaN)

    for (i <- k.indices) {

      val v = k(i)
      if (v > n - 1) betha(i) = 0
      else if (v > x.length - 1) betha(i) = 1
      else if (sigma(n - 1) >= x(v)) {
        betha(i) = new BetaDistribution(v + 1, n - v).cumulativeProbability(x(v))
      }
      else {

        // Non-trivial cases
        // Find the last element such that sigma[n0] <= x[k[i]]


        val B = Array.fill[Double](v + 2)(0.0)
        val n0 = sigma.zipWithIndex.filter(a => a._1 < x(v)).head._2
        B(0) = 1.0
        val n3 = math.min(n0, v + 1)
        for (j <- 1 to n3) B(j) = new BetaDistribution(j, n0 - j + 1).cumulativeProbability(x(v))

        // In the following update steps sigma < x[k[i]]
        val z = sigma.slice(n0, n)
        for (j <- 0 until n - n0) {
          val bb = for (y <- 0 to v) yield {
            val z1 = (1 - z(j)) * B(y + 1)
            val z2 = z(j) * B(y)
            val z3 = z1 + z2
            z3
          }
          for (y <- 0 to v) B(y + 1) = bb(y)
        }

        betha(i) = B(v + 1)
      }
    }
    betha
  }

  private def correctBetaPvalues(p: Double, k: Double): Double = math.min(p * k, 1)

  private def correctBetaPvaluesExact(p: Double, k: Int): Double = {
    val rm = Array.tabulate[Double](k) { i => 1 - new BetaDistribution(i + 1, k - i).inverseCumulativeProbability(p) }
    1 - stuart(Array(rm)).head
  }

  private def rhoScores(r: Array[Double], topCutoff: Array[Double] = Array.emptyDoubleArray, exact: Boolean = false): Double = {

    val x = if (topCutoff.isEmpty) {
      betaScores(r)
    }
    else {
      val r2 = r.filter(_.isNaN == false).map(v => if (v == 1) Double.NaN else v)
      thresholdBetaScore(r2, sig = topCutoff)
    }

    val rho = if (exact)
      correctBetaPvaluesExact(x.min, k = x.count(_.isNaN == false))
    else
      correctBetaPvalues(x.min, k = x.count(_.isNaN == false))

    rho
  }
}
