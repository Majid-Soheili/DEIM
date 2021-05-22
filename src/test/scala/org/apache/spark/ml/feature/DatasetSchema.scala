package org.apache.spark.ml.feature

import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

sealed trait DatasetSchema extends Serializable {
  def nColumns: Int

  def fIndexes: Range // feature indexes

  def cIndex: Integer // class index

  def name: String // name of data set

  def fileName: String // name of data set file

  def fNamePrefix: String = "col"

  // feature names
  def fNames: IndexedSeq[String] = fIndexes.map(id => fNamePrefix + id)

  def Schema: StructType = {
    val nullable = true
    val structures = for (i <- fIndexes) yield StructField(fNamePrefix + i, DoubleType, nullable)
    StructType(structures :+ StructField("label", DoubleType, nullable))
  }
}

object DatasetSchema {

  case object Musk extends DatasetSchema {

    override def nColumns: Int = 167

    override def fIndexes: Range = 0 to 165

    override def cIndex: Integer = 166

    override def name: String = "Musk"

    override def fileName: String = "Musk.csv"
  }

  case object Alpha extends DatasetSchema {

    override def nColumns: Int = 501

    override def fIndexes: Range = 0 to 499

    override def cIndex: Integer = 500

    override def name: String = "Alpha"

    override def fileName: String = "Alpha.csv"
  }

  case object Synthetic extends DatasetSchema {

    override def nColumns: Int = 22

    override def fIndexes: Range = 0 to 20

    override def cIndex: Integer = 21

    override def name: String = "Synthetic"

    override def fileName: String = "Synthetic.csv"
  }
}
