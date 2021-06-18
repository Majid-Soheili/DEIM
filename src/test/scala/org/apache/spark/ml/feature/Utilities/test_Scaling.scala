package org.apache.spark.ml.feature.Utilities

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{BaseSparkTest, DatasetSchema, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.col

class test_Scaling extends BaseSparkTest {

  test("MinMax Scaling - Alpha") {

    case class UT() extends Scaling
    val spark = createSession("test scaling Alpha")
    val schema = DatasetSchema.Alpha
    val data = readDataFrame(spark, schema)

    val featureAssembler = new VectorAssembler()
      .setInputCols(schema.fNames.toArray)
      .setOutputCol("features")

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
    val newNames = Seq("features", "label")
    val pipe = new Pipeline().setStages(Array(featureAssembler, indexer))
    val assembled = pipe.fit(data).transform(data).select("features", "labelIndex").toDF(newNames: _*)

    val Row(maxVec: DenseVector, minVec: DenseVector) = assembled
      .select(Summarizer.metrics("max", "min").summary(col("features")).as("summary"))
      .select("summary.max", "summary.min")
      .first()

    val utl = new UT()
    val scaled = utl.minMaxScaling(assembled.collect().toIterator, maxVec, minVec, "features", "label").toArray
    val actual = scaled.head.take(20)
    val expected = Array(0.3333, 1.0, 0.5, 0.2, 0.0, 0.3333, 0.7777, 0.5555, 0.0, 0.625, 1.0, 0.5, 0.0, 0.600, 0.5, 0.4285, 0.7142, 0.7142, 1.0, 0.8)
    assert(actual === expected)
  }
}
