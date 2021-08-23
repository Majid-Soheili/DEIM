package ir.soheili

import org.apache.spark.ml.feature.DEIM
import org.joda.time.DateTime

/**
 * Hello world!
 *
 */
object App extends BaseApp {
  def main(args: Array[String]): Unit = {

    //"SMOTE", "NearMiss2", "BaggingUnderSampling"
    localExecution = args.length == 0
    val (method, dsFile, schema, balMethod, useCatch) = if (localExecution) {
      ("QPFS", "", DatasetSchema.Wiretap, "BaggingUnderSampling", false)
    }
    else {
      val mtdName = args(0)
      val dsFile = args(1)
      val batch: Double = if (args.length >= 3 && args(2).toDouble <= 1.0 && args(2).toDouble > 0.0) args(2).toDouble else 1.0
      val useCatch: Boolean = if (args.length >= 4 && args(3).toShort >= 1) true else false
      val schema = if (dsFile.toLowerCase.contains("ocr")) DatasetSchema.OCR
      else if (dsFile.toLowerCase.contains("epsilon")) DatasetSchema.Epsilon
      else if (dsFile.toLowerCase.contains("fd")) DatasetSchema.FD
      else if (dsFile.toLowerCase.contains("ecbdl")) DatasetSchema.ECBDL
      else DatasetSchema.Alpha
      (mtdName, dsFile, schema, "", useCatch)
    }

    val start: Long = System.currentTimeMillis()
    val appName = s"DEIM-$method-$balMethod-${schema.name}"
    val spark = super.createSession(appName)

    try {

      // super.getBalancedProcessingPartitionNumber(spark) would be set to NumPartitions
      val train = super.readData(spark, schema, dsFile, Balancing = true)
      val numberPartitions = super.getBalancedProcessingPartitionNumber(spark, train.rdd.getNumPartitions)
      val wdTrain = super.WithoutDiscretizing(train, schema)

      val model = new DEIM()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setRankingMethod(method)
        .setBaggingNum(10)
        .setBalancingMethod(balMethod)
        .setUseCatch(useCatch)
        .setNumPartitions(numberPartitions)
        .fit(wdTrain)

      val fusions =
        Array(
          ("min", 1.0, 1.0),
          ("median", 1.0, 1.0),
          ("geom.mean", 1.0, 1.0),
          ("RRA", 1.0, 1.0),
          ("stuart", 1.0, 1.0),
          ("mean", 1.0, 1.0),
          ("owa", 0.95, 0.95),
          ("owa", 0.95, 0.75),
          ("owa", 0.95, 0.50),
          ("owa", 0.95, 0.25),
          ("owa", 0.95, 0.05),
          ("owa", 0.75, 0.95),
          ("owa", 0.75, 0.75),
          ("owa", 0.75, 0.50),
          ("owa", 0.75, 0.25),
          ("owa", 0.50, 0.05),
          ("owa", 0.50, 0.95),
          ("owa", 0.50, 0.75),
          ("owa", 0.50, 0.50),
          ("owa", 0.50, 0.25),
          ("owa", 0.50, 0.05),
          ("owa", 0.25, 0.95),
          ("owa", 0.25, 0.75),
          ("owa", 0.25, 0.50),
          ("owa", 0.25, 0.25),
          ("owa", 0.25, 0.05),
          ("owa", 0.05, 0.95),
          ("owa", 0.05, 0.75),
          ("owa", 0.05, 0.50),
          ("owa", 0.05, 0.25),
          ("owa", 0.05, 0.05)
        )
      val nr = if (balMethod == "BaggingUnderSampling") 31 else 11
      val ranks = fusions.take(nr).map {
        case (method, risk1, risk2) =>
          model.setFirstFusionMethod(method)
            .setSecondFusionMethod(method)
            .setFirstRiskFusion(risk1)
            .setSecondRiskFusion(risk2)
            .getRank
      }

      val duration = (System.currentTimeMillis() - start) / 60000.0
      logInfo(s"Total computing time: $duration minutes.")
      logInfo(s"Final result: ${ranks.head.take(10).mkString(", ")}")


      val SaveRankResult = true
      val SaveExecutionTime = true

      if (SaveRankResult) {
        val outString = ranks.map(r => r.mkString(",")).mkString("\n")
        super.SaveOutput(spark, appName, outString.getBytes, append = false)
      }

      if (SaveExecutionTime) {
        val numExecutors = spark.sparkContext.getConf.get("spark.executor.instances", "00")
        val outString = s"${DateTime.now()}, $appName, $numExecutors, $duration \n"
        super.SaveOutput(spark, fileName = "ExecutionTime", outString.getBytes, append = true)
      }

    } finally {
      spark.close()
    }
  }
}
