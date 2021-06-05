package org.apache.spark.ml.feature.Fusion

import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.optim.{InitialGuess, MaxEval}
import org.apache.commons.math3.optim.nonlinear.scalar.{GoalType, ObjectiveFunction}
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.{NelderMeadSimplex, SimplexOptimizer}

/**
 * the weight generator of Ordered weight averaging (OWA)
 *
 * Order Weight Averaging operator needs to specific weights vector. One of the solutions to determine the weights proposed by Maxime Lenormand in 2018.
 * We converted his proposed method from R language to Scala, according to the code published in his repository
 * Lenormand’s repository: https://github.com/maximelenormand/OWA-weights-generator
 *
 * it is worth mentioning that to compute truncate normal distribution we used the definition proposed in //http://promethee.irsn.org/R/library/msm/html/tnorm.html
 */

object OWG {

  private def dnorm(x: Double, mean: Double = 0, sd: Double = 1): Double = new NormalDistribution(mean, sd).density(x)

  private def pnorm(x: Double, mean: Double = 0, sd: Double = 1): Double = new NormalDistribution(mean, sd).cumulativeProbability(x)

  private def dtnorm(x: Double, lb: Double = Double.NegativeInfinity, ub: Double = Double.PositiveInfinity, mean: Double = 0, sd: Double = 1): Double = {
    dnorm(x, mean, sd) / (pnorm(ub, mean, sd) - pnorm(lb, mean, sd))
  }

  private def moment_tnorm(mu: Double, sd: Double): (Double, Double) = {

    val alpha = (0 - mu) / sd
    val beta = (1 - mu) / sd
    val muw = mu + sd * (dnorm(alpha) - dnorm(beta)) / (pnorm(beta) - pnorm(alpha))
    var sdw = 1 + (alpha * dnorm(alpha) - beta * dnorm(beta)) / (pnorm(beta) - pnorm(alpha)) - math.pow((dnorm(alpha) - dnorm(beta)) / (pnorm(beta) - pnorm(alpha)), 2)

    sdw = math.sqrt(math.pow(sd, 2) * sdw)

    (muw, sdw)
  }

  private def inv_moment_tnorm(muw: Double, sdw: Double): Array[Double] = {

    def L2(muw: Double, sdw: Double) = new MultivariateFunction() {

      override def value(doubles: Array[Double]): Double = {

        val par_1 = doubles(0)
        val par_2 = doubles(1)
        val mom = moment_tnorm(par_1, par_2)
        val mom_muw = mom._1
        val mom_sdw = mom._2
        math.sqrt(math.pow(muw - mom_muw, 2) + math.pow(sdw - mom_sdw, 2))
      }
    }

    try {

      val optimizer = new SimplexOptimizer(1e-10, 1e-10)
      val opt = optimizer.optimize(
        new InitialGuess(Array(muw, sdw)),
        new ObjectiveFunction(L2(muw, sdw)),
        new MaxEval(500),
        GoalType.MINIMIZE,
        new NelderMeadSimplex(Array(muw, sdw))).getPoint

      val mom = moment_tnorm(opt(0), opt(1))

      Array(opt(0), opt(1), mom._1, mom._2)
    }
    catch {
      case e: Exception => {
        println("exception caught: " + e)
        Array()
      }
    }
  }

  /**
   *
   * @param n the number of items
   * @param risk the risk value
   * @param tradeOff the tradeoff value
   * @param warn if warn true, when there is not proper distribution function a warning message will be print
   * @return an array includes weights
   */
  def getWeights(n:Int, risk:Double, tradeOff:Double,warn:Boolean=true):Array[Double] = {

    if( n == 1) {
      val w = Array.fill[Double](n)(1.0)
      w
    }
    else if(tradeOff == 0) {

      val d = Range(0, n, 1).map(v => math.abs(risk - v / (1.0 * n - 1)))
      val minIndex = d.zipWithIndex.minBy(_._1)._2
      val w = Array.fill[Double](n)(0.0)
      w(minIndex) = 1
      w
    }
    else {
      val maxsdw = 1.0 / (2 * math.sqrt(3))
      val muw = risk
      val sdw = tradeOff * maxsdw
      val res = inv_moment_tnorm(muw, sdw)

      if ((tradeOff > (4 * risk * (1 - risk))) && warn) {
        print("No suitable PDF found for these values of risk and trade-off")
      }

      val mu = res(0)
      val sd = res(1)

      //#Discretization
      val w = for (i <- 0 until n) yield dtnorm(1.0 * i / (n - 1), 0, 1, mu, sd)
      val sw = w.sum

      w.map(_ / sw).toArray

    }
  }

  def Orness(weights:Array[Double]):Double =    {
    1 - Andness(weights)
  }
  def Andness(weights:Array[Double]):Double = {

    val n = weights.length - 1
    weights.zipWithIndex.map{case(w,i) => (n-i)* w}.sum / n
  }
  def Dispersion(weights:Array[Double]):Double = {
    val n: Double = weights.length
    -1 * weights.map(w => w * math.log(w)).sum / math.log(n)
  }
  def TradeOff(weights:Array[Double]):Double = {
    val n: Double = weights.length
    1 - math.sqrt(n / (n - 1) * weights.map(w => math.pow(w - 1 / n, 2)).sum)
  }
}
