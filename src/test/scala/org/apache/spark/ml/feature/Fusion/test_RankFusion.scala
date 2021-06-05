package org.apache.spark.ml.feature.Fusion

import org.apache.spark.ml.feature.BaseTest

class test_RankFusion extends BaseTest {

  private val r1 = Seq("x", "o", "m", "w")
  private val r2 = Seq("d", "x", "h", "i", "r", "t", "a", "g", "v", "k")
  private val r3 = Seq("e", "y", "t", "q", "m", "c", "x", "u", "z", "i", "v", "h")
  private val ranks: Array[Seq[String]] = Array(r1, r2, r3)
  private val N = ranks.flatten.length //is equal to number of items in ranks

  test("stuart with N") {
    val actual = RankFusion.perform(ranks, N, method = "stuart")._2.take(5).toArray
    val expected = Array(0.00295858, 0.08192990, 0.10292444, 0.11100364, 0.11100364)
    assert(actual === expected)
  }

  test("stuart without N") {
    val actual = RankFusion.perform(ranks, method = "stuart")._2.take(5).toArray
    val expected = Array(0.00758128, 0.14564805, 0.14973028, 0.14973028, 0.18107596)
    assert(actual === expected)
  }
  test("RRA with N") {
    val actual = RankFusion.perform(ranks, N, method = "RRA")._2.take(5).toArray
    val expected = Array(0.05052344, 0.29016841, 0.33301092, 0.33301092, 0.40555303)
    assert(actual === expected)
  }
  test("RRA without N") {
    val actual = RankFusion.perform(ranks, method = "RRA")._2.take(5).toArray
    val expected = Array(0.09272489, 0.44919084, 0.44919084, 0.51392331, 0.70855810)
    assert(actual === expected)
  }

  test("min") {
    val actual = RankFusion.perform(ranks, method = "min")._2.take(5).toArray
    val expected = Array(0.05263158, 0.05263158, 0.05263158, 0.10526316, 0.10526316)
    assert(actual === expected)
  }

  test("median"){
    val actual = RankFusion.perform(ranks, method = "median")._2.take(5).toArray
    val expected = Array(0.1052632, 0.2631579, 0.3157895, 0.5263158, 0.5789474)
    assert(actual === expected)
  }

  test("geom.mean") {
    val actual = RankFusion.perform(ranks, method = "geom.mean")._2.take(5).toArray
    val expected = Array(0.1268496, 0.3463602, 0.3680627, 0.3747562, 0.3747562)
    assert(actual === expected)
  }

  test("mean") {
    val actual = RankFusion.perform(ranks, method = "mean")._2.take(5).toArray
    val expected = Array(0.02574529, 0.43726987, 0.47901273, 0.68213744, 0.71868766)
    assert(actual === expected)
  }

  test("OWA-0.95") {
    val actual = RankFusion.perform(ranks, method = "owa", risk = 0.95)._1.take(5).toArray
    val expected = Array("x", "d", "e", "o", "y")
    assert(actual === expected)
  }
  test("OWA-0.50") {
    val (actualRank, actualScores) = RankFusion.perform(ranks, method = "owa", risk = 0.50)

    //println(actualRank.zip(actualScores).map{ case (r,v) => r + " : " + v}.mkString("\n"))
    val expected = Array("x", "m", "t", "i", "h", "v", "d", "e", "o", "y", "q", "w", "r", "c", "a", "g", "u", "z", "k")
    assert(actualRank === expected)
  }
  test("OWA-0.05") {
    val (actualRank, actualScores) = RankFusion.perform(ranks, method = "owa", risk = 0.05)
    //println(actualRank.zip(actualScores).map { case (r, v) => r + " : " + v }.mkString("\n"))
    val expected = Array("x", "m", "t", "i", "v", "h", "d", "e", "o", "y", "q", "w", "r", "c", "a", "g", "u", "z", "k")
    assert(actualRank === expected)
  }
}
