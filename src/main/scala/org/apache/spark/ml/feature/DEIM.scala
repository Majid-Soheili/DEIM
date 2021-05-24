package org.apache.spark.ml.feature

import org.apache.spark.TaskContext
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.feature.Balancing.BalFactory
import org.apache.spark.ml.feature.GMI.{GMIFS, LMI}
import org.apache.spark.ml.feature.Relief.ReliefF
import org.apache.spark.ml.feature.Utilities.Scaling
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, ParamValidators, StringArrayParam}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, Row}
import org.apache.spark.storage.StorageLevel

@Experimental
final class DEIM (override val uid: String) extends Estimator[SDEM] with ebase with Scaling{

  def this() = this(Identifiable.randomUID("DEIM"))

  //region ---- Parameters -------------------------------

  private val validRankingMethods = Array("QPFS", "SRFS", "TPFS", "ReliefF") // Fisher
  private val validBalancingMethod = Array("SMOTE", "NearMiss1", "NearMiss2", "BaggingUnderSampling")

  private val maxBin: Param[Byte] = new Param[Byte](this, "maxBin", "The maximum number of bins applied in information-theory based ranking methods", ParamValidators.inRange(1, 255))
  private val useCatch: Param[Boolean] = new Param[Boolean](this, "useCatch", "Using the catch")
  private val balNumNei: Param[Int] = new Param[Int](this, "neiNumber", "the number of neighbours which is used in SMOTE and NeaMiss balancing methods")
  private val thrNominal: Param[Int] = new Param[Int](this, "thrNominal", "the number of unique values to detecting nominal features")
  private val seed: Param[Int] = new Param[Int](this, "seed", "seed value for sampling", ParamValidators.gt(0))
  private val bagNum: Param[Int] = new Param[Int](this, "bagNum", "the number of bagging used in BaggingUnderSampling methods")

  private val numSamples: Param[Int] = new Param[Int](this, "numSamples", "The number of samples applied in ReliefF method")
  private val numNeighbours: Param[Int] = new Param[Int](this, "numNeighbours", "The number of neighbours applied in ReliefF method")


  private val rankingMethod: Param[String] = new Param[String](this, "rankingMethod", "The feature ranking methods array applied as a based rankers", ParamValidators.inArray[String](validRankingMethods))
  private val balancingMethod: Param[String] = new Param[String](this, "balancingMethod", doc = "The balancing method to fix unbalancing problem", ParamValidators.inArray[String](validBalancingMethod))

  setDefault(maxBin, 10.toByte)
  setDefault(useCatch, value = false)
  setDefault(balNumNei, value = 5)
  setDefault(numSamples, value = 20)
  setDefault(numNeighbours, value = 20)
  setDefault(thrNominal, value = 5)
  setDefault(seed, value = 5341)
  setDefault(bagNum, value = 10)
  setDefault(rankingMethod, value = "QPFS")
  setDefault(balancingMethod, value = "BaggingUnderSampling")

  def getMaxBin: Int = $(maxBin)

  def getBaggingNum: Int = $(bagNum)

  def getBalancingNumNeighbours: Int = $(balNumNei)

  def getNumSamples: Int = $(numSamples)

  def getNumNeighbours: Int = $(numNeighbours)

  def getThresholdNominal: Int = $(thrNominal)

  def getSeed: Int = $(seed)

  def getRankingMethod: String = $(rankingMethod)

  def getBalancingMethod: String = $(balancingMethod)

  def getUseCatch: Boolean = $(useCatch)

  def setMaxBin(num: Byte): this.type = set(maxBin, num)

  def setBaggingNum(num: Int): this.type = set(bagNum, num)

  def setBalancingNumNeighbours(value: Int): this.type = set(balNumNei, value)

  def setNumSamples(num:Int):this.type  = set(numSamples, num)

  def setNumNeighbours(num:Int):this.type  = set(numNeighbours, num)

  def setThresholdNominal(value: Int): this.type = set(thrNominal, value)

  def setSeed(value: Int): this.type = set(seed, value)

  def setRankingMethod(names: String): this.type = set(rankingMethod, names)

  def setBalancingMethod(names: String): this.type = set(balancingMethod, names)

  def setUseCatch(value: Boolean): this.type = set(useCatch, value)

  //endregion

  //region ---- Overridable Methods ----------------------

  override def fit(dataset: Dataset[_]): SDEM = {

    val encoder = Encoders.kryo[Array[Array[Double]]]
    val Row(maxVec: DenseVector, minVec: DenseVector) = dataset
      .select(Summarizer.metrics("max", "min").summary(col($(this.featuresCol))).as("summary"))
      .select("summary.max", "summary.min")
      .first()

    val balancedData: DataFrame =
      if (this.getUseCatch)
        super.BalanceLabelDistribution(dataset.select(this.getLabelCol, this.getFeaturesCol)).persist(StorageLevel.MEMORY_AND_DISK)
      else
        super.BalanceLabelDistribution(dataset.select(this.getLabelCol, this.getFeaturesCol))

    val weightMatrix = balancedData.mapPartitions {
      rows =>

        logInfo(s"Partition Index: ${TaskContext.getPartitionId()} was started")
        if (rows.isEmpty) Iterator.empty
        else {

          val scaled = minMaxScaling(rows, maxVec, minVec, this.getFeaturesCol, this.getLabelCol).toArray
          val balFactory = new BalFactory(scaled, this.getBalancingNumNeighbours, 100, this.getThresholdNominal, this.getSeed)

          val bg = if (this.getBalancingMethod != validBalancingMethod.last) 1 else this.getBaggingNum
          Array.tabulate(bg) {
            _ =>

              val balanced = balFactory.getBalanced(this.getBalancingMethod)
              this.getRankingMethod.toUpperCase() match {
                case "QPFS" | "SRFS" | "TPFS" =>
                  val similarity = new LMI("QP", balanced, maxVec.toArray, minVec.toArray, this.getMaxBin).getSimilarity
                  new GMIFS().apply(this.getRankingMethod)(similarity)
                case "ReliefF" =>
                  new ReliefF(balanced, this.getNumSamples, this.getNumNeighbours, this.getThresholdNominal,this.getSeed)
              }
          }
          Iterator(Array.empty[Array[Double]])
        }
    }(encoder).collect()

    new SDEM(uid, weightMatrix)
  }

  override def copy(extra: ParamMap): Estimator[SDEM] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  //endregion

}
