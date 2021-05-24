package org.apache.spark.ml.feature

import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators}
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

  //region -------- Parameters ------------------------------------------

  final val selectionThreshold = new DoubleParam(this, "selectionThreshold",
    "Represents a proportion of features to select, it should be bigger than 0.0 and less than or equal to 1.0. It is by default set to 0.10.",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))
  setDefault(selectionThreshold -> 0.10)

  final val firstRiskFusion = new DoubleParam(this, "firstRiskFusion",
    "The risk value for fusion items, it should be bigger than 0.0 and less than or equal to 1.0. It is by default set to 0.95.",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))
  setDefault(firstRiskFusion -> 0.05)

  final val secondRiskFusion = new DoubleParam(this, "secondRiskFusion",
    "The risk value for fusion items, it should be bigger than 0.0 and less than or equal to 1.0. It is by default set to 0.95.",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))
  setDefault(secondRiskFusion -> 0.95)

  def getSelectionThreshold: Double = $(selectionThreshold)

  def getFirstRiskFusion: Double = $(firstRiskFusion)

  def getSecondRiskFusion: Double = $(secondRiskFusion)

  def setSelectionThreshold(value: Double): this.type = set(selectionThreshold, value)

  def setFirstRiskFusion(value: Double): this.type = set(firstRiskFusion, value)

  def setSecondRiskFusion(value: Double): this.type = set(secondRiskFusion, value)

  //endregion

  //region -------- Overridable methods ---------------------------------

  override def copy(extra: ParamMap): SDEM = ???

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???

  //endregion
}
