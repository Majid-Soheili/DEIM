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
      ("QPFS", "", DatasetSchema.UCR,"SMOTE", true)
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
      val wdTrain = super.WithoutDiscretizing(train, schema)

      val model = new DEIM()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setRankingMethod(method)
        .setBaggingNum(10)
        .setBalancingMethod(balMethod)
        .setUseCatch(useCatch)
        .setNumPartitions(1)
        .fit(wdTrain)

      val rank = model.setFirstFusionMethod("owa")
        .setSecondFusionMethod("owa")
        .setFirstRiskFusion(0.95)
        .setFirstRiskFusion(0.95)
        .getRank

      val duration = (System.currentTimeMillis() - start) / 60000.0
      logInfo(s"Total computing time: $duration minutes.")
      logInfo(s"Final result: ${rank.take(10).mkString(", ")}")

      val SaveRankResult = true
      val SaveExecutionTime = true

      if (SaveRankResult) {
        val outString = rank.mkString(",")
        super.SaveOutput(spark, appName, outString.getBytes, append = false)
      }

      if(SaveExecutionTime) {
        val numExecutors = spark.sparkContext.getConf.get("spark.executor.instances", "00")
        val outString = s"${DateTime.now()}, $appName, $numExecutors, $duration \n"
        super.SaveOutput(spark, fileName = "ExecutionTime", outString.getBytes, append = true)
      }

    } finally {
      spark.close()
    }

  }
}
