package org.apache.spark.ml.feature.OptimizationMethods.Utilities

import breeze.linalg.DenseMatrix

/**
  *  this code is taken from https://introcs.cs.princeton.edu/java/95linear/Cholesky.java.html
  *
  */
object Cholesky {

  private val EPSILON = 1e-10

  def isSymmetric(A: Array[Array[Double]]): Boolean = {

    val N = A.length
    var i = 0
    while (i < N) {
      var j = 0
      while (j < i) {
        if (A(i)(j) != A(j)(i)) return false
        j += 1
      }
      i += 1
    }
    true
  }

  def isSquare(A: Array[Array[Double]]): Boolean = {

    val N = A.length
    var i = 0
    while (i < N) {
      if (A(i).length != N) return false
      i += 1
    }
    true
  }

  def isPositiveDefinite(A: Array[Array[Double]]): Boolean = {

    if (!isSquare(A)) return false

    if (!isSymmetric(A)) return false

    val N = A.length
    val L = Array.ofDim[Double](N, N)
    var i = 0
    while (i < N) {
      var j = 0
      while (j <= i) {
        var sum = 0.0
        var k = 0
        while (k < j) {
          sum += L(i)(k) * L(j)(k)
          k += 1
        }
        if (i == j) L(i)(i) = Math.sqrt(A(i)(i) - sum)
        else L(i)(j) = 1.0 / L(j)(j) * (A(i)(j) - sum)

        j += 1
      }
      if (L(i)(i).isNaN || L(i)(i) <= 0) return false
      i += 1
    }
    true
  }

  def isPositiveDefinite(A: DenseMatrix[Double]): Boolean = {
    val n = A.rows
    isPositiveDefinite(A.toArray.sliding(n, n).toArray)
  }
}

