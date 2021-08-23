package org.apache.spark.ml.feature.OptimizationMethods

import breeze.linalg.{DenseMatrix, diag, eig, min, svd}
import com.joptimizer.functions.{ConvexMultivariateRealFunction, LinearMultivariateRealFunction, PDQuadraticMultivariateRealFunction}
import com.joptimizer.optimizers.{JOptimizer, OptimizationRequest}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.OptimizationMethods.Utilities.Cholesky

import scala.annotation.tailrec

/**
  * Quadratic Programming
  * @param ff similarity between paired features
  * @param fc relevancy between feature and class label
  */
class QP (ff: Array[Array[Double]], fc: Array[Double], doCorrection:Boolean = false) extends Logging {


  if (doCorrection && !ff.isEmpty && !Cholesky.isPositiveDefinite(ff)) {

    logWarning("there is not positive definite matrix; please correct it")
    val start = System.currentTimeMillis()
    val nearestMatrix = NearestMatrix(ff)
    logInfo(s"Finding nearest PD takes ${System.currentTimeMillis() - start}")

    ff.indices.foreach(idx => ff.update(idx, nearestMatrix(idx)))
  }



  /**
    *
    * @param a, alpha parameter which
    * @return weights of features
    */
  def getWeights(a:Double = -1): Array[Double] = {

    /*
      For calculating features weight ( or important rate), we used 'Quadratic Programming'. There are
      a few methods to minimize quadratic problem, and in this code, we used 'interior-point' method.
      Address of reference method
     */

    /*
      ffMI, is a Symmetric semi positive matrix which represents similarity between features
      fcMI, is a semi positive vector which represents similarity between feature and target class
     */

    /*
      the ffMI matrix should be a symmetric positive definite
      If a matrix is a symmetric positive definite, its the determinant values of all its sub-matrix are positive

                     1.0,  0.3,  0.9
      for example    0.3,  1.0,  0.5    All determinant values are positive (1.0,0.91 0.12) > 0
                     0.9,  0.5,  1.0    so that this matrix is a positive definite

                          1.0, 0.1, 0.8
      As another example  0.1, 1.0, 0.9   All determinant values are not positive (1.0, 0.99, -0.316)
                          0.8, 0.9, 1.0   so that this matrix is not a positive definite

      catch more information in https://www.math.utah.edu/~zwick/Classes/Fall2012_2270/Lectures/Lecture33_with_Examples.pdf

      Any way,  computing determinant values in order to check being a positive definite matrix
      is time-consuming in a large matrix, so, an alternative way is applying cholesky decomposition

      If 'A' matrix is a symmetric positive definite, the result of the cholesky decomposition is
      a unique lower triangular matrix L and its transpose: A = LL' (L' is transpose of L matrix)
      so that the values of L are grater than zero

      We applied the implementation of cholesky decomposition from below address
        https://introcs.cs.princeton.edu/java/95linear/Cholesky.java.html
     */

    /*
      If the ffMI matrix is not a symmetric positive definite, we should find the nearest positive definite matrix
      there is some useful address to find a nearest matrix

      https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite?noredirect=1&lq=1

     */

    logInfo("---------- Quadratic Programming Method ---------------")

    val start: Long = System.currentTimeMillis()
    if (ff.isEmpty || fc.isEmpty) return Array.emptyDoubleArray

    val nf = fc.length

    // if number of features equals 1, its weight is 1
    if (nf == 1) return Array(1.0)

    /* Alpha is a parameter that should balance between ffMI and fcMI */
    val mean_ffMI = 1.0 * ff.map(r => r.sum).sum / (nf * nf)
    val mean_fcMI = 1.0 * fc.sum / nf
    val alpha = if (a == -1) mean_ffMI / (mean_ffMI + mean_fcMI) else a
    if( alpha < 0 || alpha > 1)
      throw new Exception("The alpha factor is not valid.")

    logInfo("Alpha factor is: " + alpha + " | number of features: " + nf)

    if (1.0 - alpha < 1e-10)
      logWarning("Alpha factor is not valid")

    val hMat = ff.map(r => r.map(x => x * (1 - alpha)))
    val fVec = fc.map(x => -1 * x * alpha)
    val objectiveFunction = new PDQuadraticMultivariateRealFunction(hMat, fVec, 0)

    /* Equalities => sum of all weight should be equal one */
    /* [1, 1, 1] = [1] */

    val A = Array.ofDim[Double](1, nf)
    A(0) = Array.fill[Double](nf)(1.0D)
    val b = Array[Double](1)

    /* Inequality => all of the features should be bigger than zero */
    /* -1, 0 , 0
       0 ,-1 , 0
       0 , 0 , -1
     */

    val inequalities = new Array[ConvexMultivariateRealFunction](nf)
    for (i <- 0 until nf) {
      val inEqualVector = Array.fill[Double](nf)(0)
      inEqualVector(i) = -1
      inequalities(i) = new LinearMultivariateRealFunction(inEqualVector, 0)
    }

    //optimization problem
    val or = new OptimizationRequest()
    or.setA(A)
    or.setB(b)
    or.setF0(objectiveFunction)
    or.setFi(inequalities)
    or.setRescalingDisabled(false)

    //optimization
    val opt = new JOptimizer()
    opt.setOptimizationRequest(or)
    val returnCode = opt.optimize()
    val sol = opt.getOptimizationResponse.getSolution

    logInfo(s"Quadratic Programming ${System.currentTimeMillis() - start} ms")

    sol
  }

  /**
    * Find nearest symmetric positive matrix semi definite (PSD) of a given square matrix
    * @param mat a square matrix
    * @return nearest PSD
    */

  private def NearestMatrix(mat: Array[Array[Double]]): Array[Array[Double]] = {

    /*
      We converted the Python and Matlab code from below address to scala
      https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite?noredirect=1&lq=1
      https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

      Also to compute epsilon for Scala.Double we applied the source code are suggested
      https://stackoverflow.com/questions/24540349/scala-double-epsilon-calculation-in-a-functional-style

     */

    val n = mat.length
    val A = DenseMatrix(mat: _*)
    val B = (A + A.t).map(v => v / 2)
    val svd.SVD(_, s, v) = svd(B)

    val H = v.t * (diag(s) * v)
    val A2 = (B + H).map(v => v / 2)
    var A3 = (A2 + A2.t).map(v => v / 2)

    var res: DenseMatrix[Double] = null

    if (Cholesky.isPositiveDefinite(A3))
      res = A3

    val eps = calculateMachineEpsilonFloat

    val minEigen = min(eig(A3).eigenvalues)

    var k = 1
    while (!Cholesky.isPositiveDefinite(A3)) {
      val I = DenseMatrix.eye[Double](A.rows)

      A3 += I * (math.pow(-minEigen * k, 2) + eps)
      k += 1
    }
    A3.toArray.sliding(n, n).toArray
  }

  private def calculateMachineEpsilonFloat = {
    @tailrec
    def calc(machEps: Float): Float = {
      if ((1.0 + (machEps / 2.0)).toFloat != 1.0)
        calc(machEps / 2f)
      else
        machEps
    }

    calc(1f)
  }
}

object QP {
  def apply(ff: Array[Array[Double]], fc: Array[Double]): QP = new QP(ff, fc)
}

