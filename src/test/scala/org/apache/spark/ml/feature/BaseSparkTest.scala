package org.apache.spark.ml.feature

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

class BaseSparkTest extends BaseTest {

  def createSession(name:String): SparkSession = {
    def spark: SparkSession = SparkSession
      .builder()
      .appName(name = name)
      .config("spark.master", "local[5]")
      .config("spark.executor.heartbeatInterval", "1000s")
      .config("spark.network.timeout", "1200s")
      .getOrCreate()

    spark
  }

  def readRDD(spark: SparkSession, s: DatasetSchema): RDD[Row] = {

    val path = "src/test/scala/resources/data/" + s.name + ".csv"
    val context = spark.sparkContext
    val rdd = context.textFile(path)

    val cIndex = s.cIndex
    rdd.map(line => line.split(",")).map(line => Row.fromSeq(for (i <- 0 to cIndex) yield line(i).trim.toByte))
  }

  def readDataFrame(spark: SparkSession, s: DatasetSchema): DataFrame = {

    val path = "src/test/scala/resources/data/" + s.name + ".csv"
    spark.read.format("csv")
      .option("delimiter", ",").option("quote", "")
      .option("header", "false")
      .schema(s.Schema).load(path)
  }
}
