package org.apache.spark.ml.feature.OptimizationMethods.Utilities

import breeze.linalg.{DenseMatrix, diag, eig, min, svd}

import scala.annotation.tailrec

/**
  * Find nearest positive definite matrix
  * We converted the Python and Matlab code from below address to scala
  * https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite?noredirect=1&lq=1
  * https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
  *
  * Also to compute epsilon for Scala.Double we applied the source code are suggested
  * https://stackoverflow.com/questions/24540349/scala-double-epsilon-calculation-in-a-functional-style
  */
object NPD {

  def NearestMatrix(mat: Array[Array[Double]]): Array[Array[Double]] = {

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
