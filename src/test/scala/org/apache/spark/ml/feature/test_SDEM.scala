package org.apache.spark.ml.feature

import scala.util.Random

/**
 *  Test Stacking Distributed Ensemble Model
 */
class test_SDEM extends BaseTest {

  val r = new Random(4356)
  val weights: Array[Array[Array[Double]]] = Array.fill[Double](10, 10, 10)(0.0)
  for (i <- 0 until 10; j <- 0 until 10; k <- 0 until 10) weights(i)(j)(k) = r.nextDouble()

  test("SDEM- 95-95") {

    val model = new SDEM("id", weights)

    val vv = model.setFirstFusionMethod("min")
      .setSecondFusionMethod("min")
      .getRank

    println(vv.mkString(","))
    //0,1,2,3,4,5,6,7,8,9


    val actual = model.setFirstFusionMethod("owa")
      .setSecondFusionMethod("owa")
      .setFirstRiskFusion(0.95)
      .setSecondRiskFusion(0.95)
      .getRank

    val expected = Array(4, 3, 8, 0, 1, 2, 6, 7, 5, 9)
    assert(actual === expected)
  }

  test("SDEM - 95-05") {

    val model = new SDEM("id", weights)

    val actual = model.setFirstFusionMethod("owa")
      .setSecondFusionMethod("owa")
      .setFirstRiskFusion(0.95)
      .setSecondRiskFusion(0.05)
      .getRank

    val expected = Array(5, 0, 2, 7, 4, 8, 1, 9, 3, 6)
    assert(actual === expected)
  }

  test("SDEM - 05-95") {

    val model = new SDEM("id", weights)

    val actual = model.setFirstFusionMethod("owa")
      .setSecondFusionMethod("owa")
      .setFirstRiskFusion(0.05)
      .setSecondRiskFusion(0.95)
      .getRank

    val expected = Array(3, 0, 9, 7, 4, 8, 2, 1, 6, 5)
    assert(actual === expected)
  }
}
