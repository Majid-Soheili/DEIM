Distributed Ensemble Imbalanced Feature Ranking Framework
===========================================================
Distributed Ensemble Imbalanced feature selection framework, called DEIM, is presented to cope with big imbalanced datasets. In high skewed big
imbalanced datasets, It is most likely that some data partitions do not have any instances belonging to the minority class.
In this situation, the data partitions are not representative.
Consequently, the informative features of data partitions to recognizing instances of the minority class would be lost. To
this matter, DEIM utilizes a novel and approximated method to make representative data partitions only in a single pass.
Next, DEIM applies a feature ranking method on a bag of random under-sampling datasets in each data partition. Finally, the intermediate feature rankings are fused in a stacking approach in two separated levels.

The DEIM
    * feature ranking method: QPFS, ReliefF
    * balancing method: "SMOTE", "NearMiss2", "BaggingUnderSampling"
    * rank fusion meethod: "mean", "min", "median", "geom.mean", "RRA", "stuart", "owa"
    * risk factors: a double value between 0, and 1 
## Example (ml):
    import org.apache.spark.ml.feature._
    val rkMethod = "QPFS"
    val blMethod = "BaggingUnderSampling"
    val fsMethod = "owa"
    val numberPartitions = data.rdd.getNumPartitions
    val model = new DEIM()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setRankingMethod(rkMethod)
      .setBaggingNum(10)
      .setBalancingMethod(blMethod)
      .setUseCatch(false)
      .setNumPartitions(numberPartitions)
      .fit(data)
      
    val result = model.setFirstFusionMethod(fsMethod)
                   .setSecondFusionMethod(fsMethod)
                   .setFirstRiskFusion(0.95)
                   .setSecondRiskFusion(0.95)
                   .getRank  
