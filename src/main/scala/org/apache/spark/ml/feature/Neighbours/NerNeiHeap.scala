package org.apache.spark.ml.feature.Neighbours

/**
 * Nearest Neighbours Heap
 * @param capacity, the number of the nearest neighbors that should be found
 */
class NerNeiHeap(capacity: Int) extends BinaryHeap[(Int, Double)](capacity)(Ordering.by[(Int, Double), Double](_._2).reverse, reflect.classTag[(Int, Double)]) {
  def neighbourIndexes: Array[Int] = this.toArray.map(_._1).sorted
  def neighbourDistances:Array[Double] = this.toArray.sortBy(_._1).map(_._2)
}