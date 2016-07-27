package edu.uic.cs.purposeful.mpg.optimizer.numerical;

public interface IterationCallback {
  void call(int iterationIndex, double[] thetas);
}
