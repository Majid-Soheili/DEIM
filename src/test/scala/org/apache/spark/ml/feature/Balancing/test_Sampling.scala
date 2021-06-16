package org.apache.spark.ml.feature.Balancing

import org.apache.spark.ml.feature.BaseTest

import scala.collection.mutable.ListBuffer

class test_Sampling extends BaseTest{

  case class UT() extends BalancingFactory
  test("SMOTE") {

    val data = new ListBuffer[Array[Double]]()
    data += Array(0, 1, 0, 1)
    data += Array(0, 2, 1, 1)
    data += Array(3, 1, 0, 1)
    data += Array(4, 2, 0, 1)
    data += Array(3, 2, 1, 1)
    data += Array(4, 1, 0, 0)
    data += Array(6, 1, 1, 0)
    data += Array(5, 2, 0, 0)
    data += Array(0, 1, 1, 1)
    data += Array(0, 3, 0, 1)
    data += Array(7, 1, 1, 1)
    data += Array(4, 4, 0, 0)
    data += Array(6, 3, 1, 0)

    val smoted = new UT().getSmote(data.toArray, 2, 100, threshold = 1).takeRight(5)
    println(smoted.map(a => a.mkString("\t")).mkString("\n"))

    val actual = smoted.map(_.head)
    val expected = Array(4.7236, 5.6381, 5.0, 4.3618, 5.6381)

    assert(actual === expected)
  }

  test("NearMiss") {

    val data = new ListBuffer[Array[Double]]()
    data += Array(0, 1, 0, 1)
    data += Array(0, 2, 1, 1)
    data += Array(3, 1, 0, 1)
    data += Array(4, 2, 0, 1)
    data += Array(3, 2, 1, 1)
    data += Array(4, 1, 0, 0)
    data += Array(6, 1, 1, 0)
    data += Array(5, 2, 0, 0)
    data += Array(0, 1, 1, 1)
    data += Array(0, 3, 0, 1)
    data += Array(7, 1, 1, 1)
    data += Array(4, 4, 0, 0)
    data += Array(6, 3, 1, 0)

    val results = new UT().getNearMiss(data.toArray, 2, 100, version = 2)

    val pos = results.count(a => a.last == 0)
    val neg = results.count(a => a.last == 1)

    println(s"Positive instances: $pos")
    println(s"Negative instances: $neg")
    println(results.map(a => a.mkString("\t")).mkString("\n"))

    assert(pos === neg)

  }
}
