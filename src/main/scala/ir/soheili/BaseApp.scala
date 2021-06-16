package ir.soheili

import java.io.FileOutputStream

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Bucketizer, MinMaxScaler, QuantileDiscretizer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
 * The base class of spark applications
 * new update
 */

trait BaseApp extends Logging {

  final val LOCAL_FILE_PREFIX = "D://DataSets/"
  final val CLUSTER_FILE_PREFIX = "hdfs:///data/"
  final val SEED = 31
  var localExecution: Boolean = true

  //region --------- Create Sessions ----------------------------------------------------------------------

  def createSession(appName: String): SparkSession = if (localExecution) createStandaloneSession(appName) else createClusterSession(appName)

  def createStandaloneSession(appName: String, numberCores: Int = 4): SparkSession = {
    System.setProperty("hadoop.home.dir", "C:///Hadoop")
    val session = SparkSession
      .builder()
      .appName(name = appName)
      .config("spark.master", s"local[$numberCores]")
      .config("spark.eventLog.enabled", value = true)
      .config("spark.driver.maxResultSize", "4g")
      .config("spark.executor.memory", "4g")
      .config("spark.executor.heartbeatInterval", "40000s")
      .config("spark.network.timeout", "45000s")
      .config("spark.storage.blockManagerSlaveTimeoutMs", "45000s")
      .config("spark.eventLog.dir", "file:///D:/eventLogging/")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.sql.warehouse-dir", "D:///spark-warehouse")
      .getOrCreate()

    session.sparkContext.hadoopConfiguration.set("dfs.block.size", "128m")
    session
  }

  def createClusterSession(appName: String): SparkSession = {
    SparkSession
      .builder()
      .appName(name = appName)
      .getOrCreate()
  }

  //endregion

  //region --------- Read Data ----------------------------------------------------------------------------

  def readData(spark: SparkSession, s: DatasetSchema, dsName:String, Balancing: Boolean): DataFrame = {

    val data = if (s.isBinary) readBinaryData(spark, s, dsName) else readCsvData(spark, s, dsName)

    val nDefaultPartitions = data.rdd.getNumPartitions
    val np = getBalancedProcessingPartitionNumber(spark, nDefaultPartitions)

    if (Balancing && np != nDefaultPartitions) {
      logInfo(s"The number of data partitions for a balanced computing is $np")
      data.repartition(np)
    }
    else
      data
  }

  def readCsvData(spark: SparkSession, s:DatasetSchema, fileName: String = ""): DataFrame = {

    val name = if (fileName.isEmpty) s.fileName else fileName
    val path = if (localExecution) LOCAL_FILE_PREFIX + name
    else CLUSTER_FILE_PREFIX + name

    spark.read.format(source = "csv")
      .option("delimiter", ",").option("quote", "")
      .option("header", "false")
      .schema(s.Schema).load(path)
  }

  def readBinaryData(spark: SparkSession, s:DatasetSchema, fileName: String = ""): DataFrame = {

    val name = if(fileName.isEmpty) s.fileName else fileName
    val path = if (localExecution) LOCAL_FILE_PREFIX + name
    else CLUSTER_FILE_PREFIX + name

    /*
      conf.set("mapred.min.split.size", "536870912"); 512MB
      conf.set("mapred.max.split.size", "536870912");
      67108864 64MB
      134217728 128MB
    */
    val conf = spark.sparkContext.hadoopConfiguration
    conf.set("mapred.min.split.size", "67108864")

    val rdd = spark.sparkContext.binaryRecords(path, s.binaryRecordLength, conf)
      .map(Row.fromSeq(_))
    spark.createDataFrame(rdd, s.Schema)
  }

  def getBalancedProcessingPartitionNumber(spark: SparkSession, default: Int): Int = {

    val localExecution = spark.sparkContext.getConf.get("spark.master", "").contains("local")
    if (localExecution) default
    else {
      val nExecutors = spark.sparkContext.getConf.get("spark.executor.instances").toDouble
      val nExecutorCores = spark.sparkContext.getConf.get("spark.executor.cores").toInt
      val nDefaultPartitions = default
      val np = (math.round(nDefaultPartitions / (nExecutors * nExecutorCores)) * nExecutors * nExecutorCores).toInt
      logInfo(s"The number of data partitions for a balanced computing is $np")
      np
    }
  }

  def scaling(data: => DataFrame, schema: DatasetSchema): DataFrame = {

    val featureAssembler = new VectorAssembler()
      .setInputCols(schema.fNames.toArray)
      .setOutputCol("features")
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")

    val newNames = Seq("features", "label")
    val pipe = new Pipeline().setStages(Array(featureAssembler, scaler, indexer))
    pipe.fit(data).transform(data).select("scaledFeatures", "labelIndex").toDF(newNames: _*)
  }
  //endregion

  //region --------- Discretization -----------------------------------------------------------------------

  def SimpleDiscretizer(data: DataFrame, nBuckets: Int, s: DatasetSchema): DataFrame = {

    val continuous = s.cofNames.toArray
    val continuousDisc = continuous.map(c => s"${c}_disc")
    val label = s.cName

    val discretizer = new QuantileDiscretizer()
      .setInputCols(continuous)
      .setOutputCols(continuousDisc)
      .setNumBuckets(nBuckets)
      .setHandleInvalid("keep")

    val assembler = new VectorAssembler()
      .setInputCols(continuousDisc)
      .setOutputCol("FeaturesVector")
    val indexer = new StringIndexer()
      .setInputCol(label)
      .setOutputCol("labelIndex")
    val pipeline = new Pipeline()
      .setStages(Array(discretizer, assembler, indexer))
    val model = pipeline.fit(data)

    val newNames = Seq("features", label)
    model.transform(data).select("FeaturesVector", "labelIndex").toDF(newNames: _*)
  }

  def WriteSimpleDiscretizer(data: DataFrame, nBuckets: Int, s: DatasetSchema): Unit = {

    val path = if (localExecution)
      LOCAL_FILE_PREFIX + "parquet/" + s.name + ".parquet"
    else
      CLUSTER_FILE_PREFIX + "parquet/" + s.name + ".parquet"

    val continuous = s.cofNames.toArray
    val continuousDisc = continuous.map(c => s"${c}_disc")

    val discretizer = new QuantileDiscretizer()
      .setInputCols(continuous)
      .setOutputCols(continuousDisc)
      .setNumBuckets(nBuckets)

    discretizer.fit(data).write.save(path)
  }

  def LoadSimpleDiscretizer(data: DataFrame, s: DatasetSchema): DataFrame = {

    val path = if (localExecution)
      LOCAL_FILE_PREFIX + "parquet/" + s.name + ".parquet"
    else
      CLUSTER_FILE_PREFIX + "parquet/" + s.name + ".parquet"

    val continuous = s.cofNames.toArray
    val continuousDisc = continuous.map(c => s"${c}_disc")
    val label = s.cName

    val bucketizer = Bucketizer.read.load(path)

    val assembler = new VectorAssembler()
      .setInputCols(continuousDisc)
      .setOutputCol("FeaturesVector")
    val indexer = new StringIndexer()
      .setInputCol(label)
      .setOutputCol("labelIndex")
    val pipeline = new Pipeline()
      .setStages(Array(bucketizer, assembler, indexer))
    val model = pipeline.fit(data)

    val newNames = Seq("features", label)
    model.transform(data).select("FeaturesVector", "labelIndex").toDF(newNames: _*)
  }

  def WithoutDiscretizing(data: => DataFrame, s: DatasetSchema): DataFrame = {

    val continuous = s.cofNames.toArray
    val label = s.cName
    val assembler = new VectorAssembler()
      .setInputCols(continuous)
      .setOutputCol("FeaturesVector")
    val indexer = new StringIndexer()
      .setInputCol(label)
      .setOutputCol("labelIndex")
    val pipeline = new Pipeline()
      .setStages(Array(assembler, indexer))
    val model = pipeline.fit(data)
    val newNames = Seq("features", label)
    model.transform(data).select("FeaturesVector", "labelIndex").toDF(newNames: _*)
  }

  //endregion

  //region --------- Saving ---------------------------------------------------

  def SaveOutput(spark: SparkSession, fileName: String, data: Array[Byte], append: Boolean): Unit = {

    val master = spark.sparkContext.getConf.get("spark.master", "")
    val deploy = spark.sparkContext.getConf.get("spark.submit.deployMode", "")

    if (master.contains("local"))
      WriteInFS(LOCAL_FILE_PREFIX + s"Output/$fileName.csv", data, append)
    else if (master.contains("yarn") && deploy.contains("client"))
      WriteInFS(fileName = s"/usr/local/spark/bin/spark-warehouse/$fileName.csv", data, append)
    else
      WriteInHDFS(spark, fileName, data)
  }

  def WriteInHDFS(spark: SparkSession, fileName: String, data: Array[Byte]): Unit = {

    val path = new Path(s"hdfs:///output/$fileName.csv")
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val os = fs.create(path)

    os.write(data)
    os.close()
    fs.close()
  }

  def WriteInFS(fileName: String, data: Array[Byte], append: Boolean): Unit = {
    val out = new FileOutputStream(fileName, append)
    out.write(data)
    out.close()
  }

  //endregion

}
