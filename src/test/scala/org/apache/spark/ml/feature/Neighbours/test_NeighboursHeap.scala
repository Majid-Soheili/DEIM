package org.apache.spark.ml.feature.Neighbours

import org.apache.spark.ml.feature.BaseTest

class test_NeighboursHeap extends BaseTest{

  test("BinaryHeap-Simple") {
    val bheap = new BinaryHeap[Int](5)(Ordering.Int.reverse, reflect.classTag[Int])
    bheap.+=(15)
    bheap.+=(40)
    bheap.+=(50)
    bheap.+=(10)
    bheap.+=(41)
    bheap.+=(100)
    bheap.+=(30)

    val actual = bheap.toArray.sorted
    val expected = Array(10, 15, 30, 40, 41)
    assert( actual === expected)
  }

  test("Nearest Neighbour Heap") {
    val nHeap = new NerNeiHeap(5)
    nHeap.+=(1, 15.0)
    nHeap.+=(2, 40.0)
    nHeap.+=(3, 50.0)
    nHeap.+=(4, 10.0)
    nHeap.+=(5, 41.0)
    nHeap.+=(6, 100.0)
    nHeap.+=(7, 30.0)

    val actual = nHeap.toArray.map(_._2).sorted
    val expected = Array(10.0, 15.0, 30.0, 40.0, 41.0)
    assert(actual === expected)
  }

  test("Furthest Neighbour Heap") {
    val fHeap = new FurNeiHeap(5)
    fHeap.+=(1, 15.0)
    fHeap.+=(2, 40.0)
    fHeap.+=(3, 50.0)
    fHeap.+=(4, 10.0)
    fHeap.+=(5, 41.0)
    fHeap.+=(6, 100.0)
    fHeap.+=(7, 30.0)

    val actual = fHeap.toArray.map(_._2).sorted.reverse
    val expected = Array(100.0, 50.0, 41.0, 40.0, 30.0)
    assert(actual === expected)
  }

}
