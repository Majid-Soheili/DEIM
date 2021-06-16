package org.apache.spark.ml.feature

import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Model
import org.apache.spark.ml.feature.Fusion.RankFusion
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/**
 *  Stacking Distributed Ensemble Model
 */

@Experimental
final class SDEM private [ml](override val uid: String, private val featuresWeights: Array[Array[Array[Double]]])
  extends Model[SDEM] with HasFeaturesCol with HasOutputCol with HasLabelCol{


  logInfo(s"The number of feature weights is equal to ${featuresWeights.length} X ${featuresWeights.head.length}")

  //region -------- Parameters ------------------------------------------

  private val validFusionMethods = Array("mean", "min", "median", "geom.mean", "RRA", "stuart", "owa")

  final val firstFusionMethod = new Param[String](this, "fusionMethod",
    "A fusion method applied in aggregate local feature rankings",
    ParamValidators.inArray(validFusionMethods))

  final val secondFusionMethod = new Param[String](this, "fusionMethod",
    "A fusion method applied in aggregate global feature rankings",
    ParamValidators.inArray(validFusionMethods))

  final val selectionThreshold = new DoubleParam(this, "selectionThreshold",
    "Represents a proportion of features to select, it should be bigger than 0.0 and less than or equal to 1.0. It is by default set to 0.10.",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))

  final val firstRiskFusion = new DoubleParam(this, "firstRiskFusion",
    "The risk value for fusion items, it should be bigger than 0.0 and less than or equal to 1.0. It is by default set to 0.95.",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))

  final val secondRiskFusion = new DoubleParam(this, "secondRiskFusion",
    "The risk value for fusion items, it should be bigger than 0.0 and less than or equal to 1.0. It is by default set to 0.95.",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))


  setDefault(firstFusionMethod, "min")
  setDefault(secondFusionMethod, "min")
  setDefault(selectionThreshold, 0.10)
  setDefault(firstRiskFusion, 0.05)
  setDefault(secondRiskFusion, 0.95)

  def getFirstFusionMethod: String = $(firstFusionMethod)

  def getSecondFusionMethod: String = $(secondFusionMethod)

  def getSelectionThreshold: Double = $(selectionThreshold)

  def getFirstRiskFusion: Double = $(firstRiskFusion)

  def getSecondRiskFusion: Double = $(secondRiskFusion)

  def setFirstFusionMethod(value: String): this.type = set(firstFusionMethod, value)

  def setSecondFusionMethod(value: String): this.type = set(secondFusionMethod, value)

  def setSelectionThreshold(value: Double): this.type = set(selectionThreshold, value)

  def setFirstRiskFusion(value: Double): this.type = set(firstRiskFusion, value)

  def setSecondRiskFusion(value: Double): this.type = set(secondRiskFusion, value)

  //endregion

  def getRank: Array[Int] = {

    val ranks = featuresWeights.map(w => w.map(ww => ww.zipWithIndex.sortBy(_._1).reverse.map(_._2)))
    val firstFusion = ranks.map {
      first =>
        val input = first.map(_.toSeq)
        val (rank, _) = RankFusion.perform[Int](input, method = this.getFirstFusionMethod, risk = this.getFirstRiskFusion)
        rank
    }

    val (result, _) = RankFusion.perform[Int](firstFusion, method = this.getSecondFusionMethod, risk = this.getSecondRiskFusion)
    result.toArray
  }


  //region -------- Overridable methods ---------------------------------

  override def copy(extra: ParamMap): SDEM = ???

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???

  //endregion
}
