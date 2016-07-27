package edu.uic.cs.purposeful.mpg.optimizer.numerical.lbfgs.stanford;


/**
 * Class ArrayMath
 *
 * @author Teg Grenager
 */
class ArrayMath {

  private ArrayMath() {} // not instantiable

  /**
   * Scales the values in this array by b. Does it in place.
   */
  static void multiplyInPlace(double[] a, double b) {
    for (int i = 0; i < a.length; i++) {
      a[i] = a[i] * b;
    }
  }

  /**
   * Computes 1-norm of vector.
   *
   * @param a A vector of double
   * @return 1-norm of a
   */
  static double norm_1(double[] a) {
    double sum = 0;
    for (double anA : a) {
      sum += (anA < 0 ? -anA : anA);
    }
    return sum;
  }

  /**
   * Computes 2-norm of vector.
   *
   * @param a A vector of double
   * @return Euclidean norm of a
   */
  static double norm(double[] a) {
    double squaredSum = 0;
    for (double anA : a) {
      squaredSum += anA * anA;
    }
    return Math.sqrt(squaredSum);
  }

  /**
   * @return the index of the max value; if max is a tie, returns the first one.
   */
  private static int argmax(double[] a) {
    double max = Double.NEGATIVE_INFINITY;
    int argmax = 0;
    for (int i = 0; i < a.length; i++) {
      if (a[i] > max) {
        max = a[i];
        argmax = i;
      }
    }
    return argmax;
  }

  static double max(double[] a) {
    return a[argmax(a)];
  }

  /**
   * @return the index of the min value; if min is a tie, returns the first one.
   */
  private static int argmin(double[] a) {
    double min = Double.POSITIVE_INFINITY;
    int argmin = 0;
    for (int i = 0; i < a.length; i++) {
      if (a[i] < min) {
        min = a[i];
        argmin = i;
      }
    }
    return argmin;
  }

  /**
   * @return The minimum value in an array.
   */
  static double min(double[] a) {
    return a[argmin(a)];
  }

  // LINEAR ALGEBRAIC FUNCTIONS

  static double innerProduct(double[] a, double[] b) {
    double result = 0.0;
    int len = Math.min(a.length, b.length);
    for (int i = 0; i < len; i++) {
      result += a[i] * b[i];
    }
    return result;
  }
}
