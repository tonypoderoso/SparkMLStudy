package algorithms.common

/**
  * Created by tonypark on 2016. 6. 7..
  */

import java.nio.ByteBuffer
import java.util.{Random => JavaRandom}

import scala.util.hashing.MurmurHash3


/**
  * This class implements a XORShift random number generator algorithm
  * Source:
  * Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, Vol. 8, Issue 14.
  * @see <a href="http://www.jstatsoft.org/v08/i14/paper">Paper</a>
  * This implementation is approximately 3.5 times faster than
  * {@link java.util.Random java.util.Random}, partly because of the algorithm, but also due
  * to renouncing thread safety. JDK's implementation uses an AtomicLong seed, this class
  * uses a regular Long. We can forgo thread safety since we use a new instance of the RNG
  * for each thread.
  */
class XORShiftRandom(init: Long) extends JavaRandom(init) {



  def this() = this(System.nanoTime)

  private var seed = XORShiftRandom.hashSeed(init)

  // we need to just override next - this will be called by nextInt, nextDouble,
  // nextGaussian, nextLong, etc.
  override protected def next(bits: Int): Int = {
    var nextSeed = seed ^ (seed << 21)
    nextSeed ^= (nextSeed >>> 35)
    nextSeed ^= (nextSeed << 4)
    seed = nextSeed
    (nextSeed & ((1L << bits) -1)).asInstanceOf[Int]
  }

  override def setSeed(s: Long) {
    seed = XORShiftRandom.hashSeed(s)
  }
}

/* * Contains benchmark method and main method to run benchmark of the RNG */
object XORShiftRandom {

  /**
    * Method executed for repeating a task for side effects.
    * Unlike a for comprehension, it permits JVM JIT optimization
    */
  def times(numIters: Int)(f: => Unit): Unit = {
    var i = 0
    while (i < numIters) {
      f
      i += 1
    }
  }

  /**
    * Timing method based on iterations that permit JVM JIT optimization.
    *
    * @param numIters number of iterations
    * @param f function to be executed. If prepare is not None, the running time of each call to f
    *          must be an order of magnitude longer than one millisecond for accurate timing.
    * @param prepare function to be executed before each call to f. Its running time doesn't count.
    * @return the total time across all iterations (not counting preparation time)
    */
  def timeIt(numIters: Int)(f: => Unit, prepare: Option[() => Unit] = None): Long = {
    if (prepare.isEmpty) {
      val start = System.currentTimeMillis
      times(numIters)(f)
      System.currentTimeMillis - start
    } else {
      var i = 0
      var sum = 0L
      while (i < numIters) {
        prepare.get.apply()
        val start = System.currentTimeMillis
        f
        sum += System.currentTimeMillis - start
        i += 1
      }
      sum
    }
  }


  /** Hash seeds to have 0/1 bits throughout. */
  def hashSeed(seed: Long): Long = {
    val bytes = ByteBuffer.allocate(java.lang.Long.SIZE).putLong(seed).array()
    val lowBits = MurmurHash3.bytesHash(bytes)
    val highBits = MurmurHash3.bytesHash(bytes, lowBits)
    (highBits.toLong << 32) | (lowBits.toLong & 0xFFFFFFFFL)
  }

  /**
    * Main method for running benchmark
    * @param args takes one argument - the number of random numbers to generate
    */
  def main(args: Array[String]): Unit = {
    // scalastyle:off println
    if (args.length != 1) {
      println("Benchmark of XORShiftRandom vis-a-vis java.util.Random")
      println("Usage: XORShiftRandom number_of_random_numbers_to_generate")
      System.exit(1)
    }
    println(benchmark(args(0).toInt))
    // scalastyle:on println
  }

  /**
    * @param numIters Number of random numbers to generate while running the benchmark
    * @return Map of execution times for {@link java.util.Random java.util.Random}
    * and XORShift
    */
  def benchmark(numIters: Int): Map[String, Long] = {

    val seed = 1L
    val million = 1e6.toInt
    val javaRand = new JavaRandom(seed)
    val xorRand = new XORShiftRandom(seed)

    // this is just to warm up the JIT - we're not timing anything
    timeIt(million) {
      javaRand.nextInt()
      xorRand.nextInt()
    }

    /* Return results as a map instead of just printing to screen
    in case the user wants to do something with them */
    Map("javaTime" -> timeIt(numIters) { javaRand.nextInt() },
      "xorTime" -> timeIt(numIters) { xorRand.nextInt() })
  }
}

