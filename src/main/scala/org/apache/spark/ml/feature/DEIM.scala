package org.apache.spark.ml.feature

import org.apache.spark.TaskContext
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.feature.Balancing.BalancingFactory
import org.apache.spark.ml.feature.GMI.{GMIFS, LMI}
import org.apache.spark.ml.feature.Relief.ReliefF
import org.apache.spark.ml.feature.Utilities.Scaling
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, ParamValidators, StringArrayParam}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, Row}
import org.apache.spark.storage.StorageLevel

import scala.util.Random

@Experimental
final class DEIM (override val uid: String) extends Estimator[SDEM] with ebase with Scaling with BalancingFactory {

  def this() = this(Identifiable.randomUID("DEIM"))

  //region ---- Parameters -------------------------------

  private val validRankingMethods = Array("QPFS", "SRFS", "TPFS", "ReliefF") // Fisher
  private val validBalancingMethod = Array("SMOTE", "NearMiss1", "NearMiss2", "BaggingUnderSampling", "none")

  val maxBin: Param[Int] = new Param[Int](this, "maxBin", "The maximum number of bins applied in information-theory based ranking methods", ParamValidators.inRange(1, 255))
  val useCatch: Param[Boolean] = new Param[Boolean](this, "useCatch", "Using the catch")
  val balNumNei: Param[Int] = new Param[Int](this, "neiNumber", "the number of neighbours which is used in SMOTE and NeaMiss balancing methods")
  val thrNominal: Param[Int] = new Param[Int](this, "thrNominal", "the number of unique values to detecting nominal features")
  val seed: Param[Int] = new Param[Int](this, "seed", "seed value for sampling", ParamValidators.gt(0))
  val bagNum: Param[Int] = new Param[Int](this, "bagNum", "the number of bagging used in BaggingUnderSampling methods", ParamValidators.gt(0))

  val numSamples: Param[Int] = new Param[Int](this, "numSamples", "The number of samples applied in ReliefF method")
  val numNeighbours: Param[Int] = new Param[Int](this, "numNeighbours", "The number of neighbours applied in ReliefF method")

  val rankingMethod: Param[String] = new Param[String](this, "rankingMethod", "The feature ranking methods array applied as a based rankers", ParamValidators.inArray[String](validRankingMethods))
  val balancingMethod: Param[String] = new Param[String](this, "balancingMethod", doc = "The balancing method to fix unbalancing problem", ParamValidators.inArray[String](validBalancingMethod))


  setDefault(maxBin, 10)
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

  def setMaxBin(num: Int): this.type = set(maxBin, num)

  def setBaggingNum(num: Int): this.type = set(bagNum, num)

  def setBalancingNumNeighbours(value: Int): this.type = set(balNumNei, value)

  def setNumSamples(num: Int): this.type = set(numSamples, num)

  def setNumNeighbours(num: Int): this.type = set(numNeighbours, num)

  def setThresholdNominal(value: Int): this.type = set(thrNominal, value)

  def setSeed(value: Int): this.type = set(seed, value)

  def setRankingMethod(names: String): this.type = set(rankingMethod, names)

  def setBalancingMethod(names: String): this.type = set(balancingMethod, names)

  def setUseCatch(value: Boolean): this.type = set(useCatch, value)

  //endregion

  //region ---- Overridable Methods ----------------------

  override def fit(dataset: Dataset[_]): SDEM = {

    if (validBalancingMethod.take(3).contains(this.getBalancingMethod)) this.setBaggingNum(1)

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
          val rnd = new Random(this.getSeed)
          val weights = Array.tabulate(this.getBaggingNum) {
            counter =>
              logInfo(s"Bagging number ${counter} in data partition ${TaskContext.getPartitionId()} was started")

              val (balanced, indexes) = super.getBalancedDataAndIndex(
                this.getBalancingMethod, scaled,
                this.getBalancingNumNeighbours, percentage = 100,
                this.getThresholdNominal, rnd.nextInt()) //

              this.getRankingMethod.toUpperCase() match {
                case "QPFS" =>
                  val similarity = new LMI("QP", balanced, indexes, this.getMaxBin).getSimilarity
                  new GMIFS().apply(this.getRankingMethod)(similarity)
                case "SRFS" | "TPFS" =>
                  val similarity = new LMI("SR", balanced, indexes, this.getMaxBin).getSimilarity
                  new GMIFS().apply(this.getRankingMethod)(similarity)
                case "RELIEFF" =>
                  new ReliefF(balanced, indexes, this.getNumSamples, this.getNumNeighbours, this.getThresholdNominal, rnd.nextInt()).getWeights
                // Apply indexes, the sub set of dataset would be apply as like as  QPFS
              }
          }
          Iterator(weights)
        }
    }(encoder).collect()

    new SDEM(uid, weightMatrix)
  }

  override def copy(extra: ParamMap): Estimator[SDEM] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  //endregion

}
