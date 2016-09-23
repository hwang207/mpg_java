package edu.uic.cs.purposeful.mpg.optimizer.numerical;

import edu.uic.cs.purposeful.mpg.common.FeatureWiseRegularization;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.objective.MinimizationObjectiveFunction;

public interface NumericalOptimizer {

  void setMinimizationObjectiveFunction(MinimizationObjectiveFunction objectiveFunction);

  boolean optimize(double[] thetas, Regularization regularization);

  boolean optimize(double[] thetas, Regularization regularization,
      IterationCallback iterationCallback);

  boolean optimize(double[] thetas, FeatureWiseRegularization featureWiseRegularization);

  boolean optimize(double[] thetas, FeatureWiseRegularization featureWiseRegularization,
      IterationCallback iterationCallback);
}
