package org.apache.spark.ml.feature.GMI

import org.apache.spark.ml.feature.{BaseSparkTest, DatasetSchema}

class test_GMI extends BaseSparkTest {

  /**
   *  the Test of Global Mutual Information based Feature Selection (GMIFS) methods
   */

  test("GMIFS-QP-Musk") {

    val schema = DatasetSchema.Musk
    val data = readFile(schema.name)
    val (minVec, maxVec) = MinMax(data)

    val similarity = new LMI("QP", data, maxVec, minVec, 10).getSimilarity
    val weights = new GMIFS().apply("QPFS")(similarity)
    val actualRank = weights.zipWithIndex.sortBy(_._1).reverse.map(_._2).take(10)

    //val expectedRank = Array(125, 144, 109, 94, 66, 137, 150, 35, 162, 74)
    val expectedRank = Array(125, 144, 94, 108, 137, 66, 162, 150, 74, 165)

    assert(actualRank === expectedRank)
  }

    test(testName = "GMIFS-SR-Synthetic") {

      val schema = DatasetSchema.Synthetic
      val data = readFile(schema.name)
      val (minVec, maxVec) = MinMax(data)

      val similarity = new LMI("SR", data, maxVec, minVec, 10).getSimilarity
      val weights = new GMIFS().apply("SRFS")(similarity)
      val actualRank = weights.zipWithIndex.sortBy(_._1).reverse.map(_._2)
      val expectedRank = Array(10, 11, 9, 8, 14, 15, 5, 16, 12, 4, 6, 13, 7, 3, 17, 18, 2, 19, 0, 1, 20)
      assert(actualRank === expectedRank)
    }

      test(testName = "GMIFS-SR-Musk") {

        val schema = DatasetSchema.Musk
        val data = readFile(schema.name)
        val (minVec, maxVec) = MinMax(data)
        val similarity = new LMI("SR", data, maxVec, minVec, 30).getSimilarity
        val weights = new GMIFS().apply("SRFS")(similarity)
        val actualRank = weights.zipWithIndex.sortBy(_._1).reverse.map(_._2).take(10)
        val expectedRank = Array(161, 101, 164, 162, 35, 41, 91, 160, 165, 130)
        //val expectedRank = Array(125, 144, 109, 94, 66, 137, 150, 35, 162, 74)

        assert(actualRank === expectedRank)
      }

      test(testName = "GMIFS-SR-Alpha") {

        val schema = DatasetSchema.Alpha
        val data = readFile(schema.name)
        val (minVec, maxVec) = MinMax(data)
        val similarity = new LMI("SR", data, maxVec, minVec, 30).getSimilarity
        val weights = new GMIFS().apply("SRFS")(similarity)
        val actualRank = weights.zipWithIndex.sortBy(_._1).reverse.map(_._2).take(10)
        val expectedRank = Array(493, 285, 470, 498, 457, 297, 87, 114, 6, 427)

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
