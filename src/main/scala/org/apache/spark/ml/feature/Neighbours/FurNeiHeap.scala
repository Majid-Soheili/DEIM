package org.apache.spark.ml.feature.Neighbours

/**
 * Furthest Neighbours Heap
 * @param capacity, the number of the furthest neighbors that should be found
 */
class FurNeiHeap(capacity: Int) extends BinaryHeap[(Int, Double)](capacity)(Ordering.by[(Int, Double), Double](_._2), reflect.classTag[(Int, Double)]) {
  def neighbourIndexes: Array[Int] = this.toArray.sortBy(_._2).map(_._1).sorted
}

