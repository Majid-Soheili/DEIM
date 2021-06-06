package org.apache.spark.ml.feature.Utilities

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{BaseSparkTest, BaseTest, DatasetSchema, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.col

class test_Discretizing extends BaseSparkTest {

  test("Equal Width - Alpha") {

    case class UT() extends Discretizing
    val spark = createSession("test discretizing Alpha")
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

    val asData = assembled.collect().map { row => row.getAs[DenseVector](0).toArray :+ row.getDouble(1) }

    val utl = new UT()
    val desc = utl.discEW(asData, maxVec.toArray, minVec.toArray, 10).toArray
    val actual = desc.head.take(20)
    val expected = Array(3, 9, 5, 2, 0, 3, 7, 5, 0, 6, 9, 5, 0, 6, 0, 4, 7, 7, 9, 8)
    assert(actual === expected)
  }

  test("Equal Width - Musk") {

    case class UT() extends Discretizing
    val spark = createSession("test discretizing Musk")
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
    val assembled = pipe.fit(data).transform(data).select("features", "labelIndex").toDF(newNames: _*)

    val Row(maxVec: DenseVector, minVec: DenseVector) = assembled
      .select(Summarizer.metrics("max", "min").summary(col("features")).as("summary"))
      .select("summary.max", "summary.min")
      .first()

    val asData = assembled.collect().map { row => row.getAs[DenseVector](0).toArray :+ row.getDouble(1) }

    val utl = new UT()
    val desc = utl.discEW(asData, maxVec.toArray, minVec.toArray, 10)
    val actual = desc.head.take(20)
    val expected = Array(3, 6, 9, 5, 0, 9, 0, 0, 0, 9, 0, 0, 3, 0, 1, 0, 2, 0, 0, 0)
    assert(actual === expected)
  }
}
