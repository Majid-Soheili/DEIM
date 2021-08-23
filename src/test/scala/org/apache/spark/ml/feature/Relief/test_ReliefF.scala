package org.apache.spark.ml.feature.Relief

import org.apache.spark.ml.feature.Utilities.Scaling
import org.apache.spark.ml.feature.{BaseSparkTest, DatasetSchema}

class test_ReliefF extends BaseSparkTest with Scaling {

  test("ReliefF-Musk") {

    val schema = DatasetSchema.Musk
    val data = readFile(schema.name)
    val (minVec, maxVec) = MinMax(data)
    val scaled = super.minMaxScaling(data, maxVec, minVec)
    val nInstances = scaled.length

    val rel = new ReliefF(data, data.indices.toArray, nInstances, 10, 1)
    val actualRank = rel.getRanks.take(10)
    val expectedRank = Array(35, 162, 83, 96, 32, 159, 135, 58, 36, 95)
    assert(actualRank === expectedRank)

  }

  test("ReliefF-Musk-threshold-10") {

    val schema = DatasetSchema.Musk
    val data = readFile(schema.name)
    val (minVec, maxVec) = MinMax(data)
    val scaled = super.minMaxScaling(data, maxVec, minVec)
    val nInstances = scaled.length

    val rel = new ReliefF(data, data.indices.toArray, nInstances, 10, 10)
    val actualRank = rel.getRanks.take(10)
    val expectedRank = Array(94, 83, 145, 36, 3, 96, 87, 120, 89, 158)
    assert(actualRank === expectedRank)
  }

  private def MinMax(data: => Array[Array[Double]]):(Array[Double], Array[Double]) = {

    val nf = data.head.length - 1
    val minVec = data.head.clone().dropRight(1)
    val maxVec = data.head.clone().dropRight(1)

    data.foreach(a =>
      for (i <- 0 until nf) {
        if (a(i) < minVec(i)) minVec(i) = a(i)
        if (a(i) > maxVec(i)) maxVec(i) = a(i)
      })

    (minVec, maxVec)
  }
}
