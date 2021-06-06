package org.apache.spark.ml.feature

import org.apache.spark.ml.Pipeline

class test_DEIM extends BaseSparkTest {

  test("DEIM-QPFS-Musk-None-50") {

    val spark = createSession("DEIM-QPFS-Musk")
    val schema = DatasetSchema.Musk
    val data = readDataFrame(spark, schema)

    val featureAssembler = new VectorAssembler()
      .setInputCols(schema.fNames.toArray)
      .setOutputCol("features")

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
    val newNames = Seq("features", "label")
    val pipe = new Pipeline().setStages(Array(featureAssembler, indexer))
    val processedData = pipe.fit(data).transform(data).select("features", "labelIndex").toDF(newNames: _*)

    val deim = new DEIM()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setBaggingNum(1)
      .setBalancingMethod("none")
      .setRankingMethod("QPFS")
      .setMaxBin(50)

    val model = deim.fit(processedData)

    val ranking = model.setFirstFusionMethod("min")
      .setSecondFusionMethod("min")
      .getRank

    val actualRank = ranking.take(10)
    val expectedRank = Array(125, 144, 109, 94, 66, 137, 150, 35, 162, 74)
    assert(actualRank === expectedRank)
  }

  test("DEIM-SRFS-Musk-None-50") {

    val spark = createSession("DEIM-SRFS-Musk")
    val schema = DatasetSchema.Musk
    val data = readDataFrame(spark, schema)
    val featureAssembler = new VectorAssembler()
      .setInputCols(schema.fNames.toArray)
      .setOutputCol("features")

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
    val newNames = Seq("features", "label")
    val pipe = new Pipeline().setStages(Array(featureAssembler, indexer))
    val processedData = pipe.fit(data).transform(data).select("features", "labelIndex").toDF(newNames: _*)


    val model = new DEIM()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setBaggingNum(1)
      .setBalancingMethod("none")
      .setRankingMethod("SRFS")
      .setMaxBin(50)
      .setNumPartitions(1)
      .fit(processedData)

    val actualRank = model.getRank.take(10)
    val expectedRank = Array(161, 101, 164, 162, 35, 41, 91, 160, 165, 130)

    assert(actualRank === expectedRank)

  }

  test("DEIM-ReliefF-Musk") {

    val spark = createSession("tsRelief-Musk")
    val schema = DatasetSchema.Musk
    val data = readDataFrame(spark, schema)
    val nInstances = data.count()

    val featureAssembler = new VectorAssembler()
      .setInputCols(schema.fNames.toArray)
      .setOutputCol("features")

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
    val newNames = Seq("features", "label")
    val pipe = new Pipeline().setStages(Array(featureAssembler, indexer))
    val processedData = pipe.fit(data).transform(data).select("features", "labelIndex").toDF(newNames: _*)

    val start = System.currentTimeMillis()
    val model = new DEIM()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setBaggingNum(1)
      .setBalancingMethod("none")
      .setRankingMethod("ReliefF")
      .setNumSamples(nInstances.toInt)
      .setNumNeighbours(10)
      .setNumPartitions(1)
      .setThresholdNominal(1)
      .fit(processedData)

    logInfo(s"DEIM-Relief takes ${{System.currentTimeMillis() - start}} ms")
    val actualRank = model.getRank.take(10)

    val expectedRank = Array(35, 162, 83, 96, 32, 159, 135, 58, 36, 95)

    assert(actualRank === expectedRank)
  }
}
