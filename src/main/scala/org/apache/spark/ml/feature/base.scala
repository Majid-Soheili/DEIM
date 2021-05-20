package org.apache.spark.ml.feature

import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}

trait base extends HasFeaturesCol with HasOutputCol with HasLabelCol  with Logging {

  def setFeaturesCol(name: String): this.type = set(featuresCol, name)

  def setLabelCol(name: String): this.type = set(labelCol, name)

}