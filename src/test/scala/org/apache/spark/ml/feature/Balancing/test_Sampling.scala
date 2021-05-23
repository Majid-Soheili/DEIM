package org.apache.spark.ml.feature.Balancing

import org.apache.spark.ml.feature.BaseTest

import scala.collection.mutable.ListBuffer

class test_Sampling extends BaseTest{

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

   val smoted=  new BalFactory(data.toArray, 2, 100).getSmote.takeRight(5)

    println(smoted.map(a => a.mkString("\t")).mkString("\n"))

    val actual = smoted.map(_.head)
    val expected = Array(4.72360, 6.0, 5.83309, 4.65295, 5.76447)

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

    val results = new BalFactory(data.toArray, 2, 100).getNearMiss(2)

    val pos = results.count(a => a.last == 0)
    val neg = results.count(a => a.last == 1)

    println(s"Positive instances: $pos")
    println(s"Negative instances: $neg")
    println(results.map(a => a.mkString("\t")).mkString("\n"))

    assert(pos === neg)

  }
}
