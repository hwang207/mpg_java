package edu.uic.cs.purposeful.mpg.optimizer.numerical.objective;

import org.apache.commons.lang3.tuple.Pair;

import edu.uic.cs.purposeful.mpg.common.FeatureWiseRegularization;
import edu.uic.cs.purposeful.mpg.common.Regularization;

/**
 * Find thetas that gives min{#getValue(thetas)}
 */
public interface MinimizationObjectiveFunction {

  Pair<Double, double[]> getValueAndGradients(double[] thetas);

  Pair<Double, double[]> getValueAndGradients(double[] thetas, int[] indicesInBatch);

  double getValue(double[] thetas);

  double getValue(double[] thetas, int[] indicesInBatch);

  double[] getGradients(double[] thetas);

  double[] getGradients(double[] thetas, int[] indicesInBatch);

  void setRegularization(Regularization regularization);

  void setRegularization(FeatureWiseRegularization featureWiseRegularization);

  int getNumberOfInstances();
}
