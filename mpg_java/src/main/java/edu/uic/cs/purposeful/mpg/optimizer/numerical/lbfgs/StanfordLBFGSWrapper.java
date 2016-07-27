package edu.uic.cs.purposeful.mpg.optimizer.numerical.lbfgs;

import org.apache.log4j.Logger;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.IterationCallback;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.NumericalOptimizer;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.lbfgs.stanford.StanfordCoreNLPQNMinimizerLite;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.objective.MinimizationObjectiveFunction;

public class StanfordLBFGSWrapper implements NumericalOptimizer {
  private static final Logger LOGGER = Logger.getLogger(StanfordLBFGSWrapper.class);

  private static final int NUMBER_OF_PREVIOUS_ESTIMATIONS = 15;
  private static final boolean USE_ROBUST_OPTIONS = true;

  private MinimizationObjectiveFunction objectiveFunction;

  @Override
  public void setMinimizationObjectiveFunction(MinimizationObjectiveFunction objectiveFunction) {
    this.objectiveFunction = objectiveFunction;
  }

  @Override
  public boolean optimize(double[] thetas, Regularization regularization) {
    return optimize(thetas, regularization, null);
  }

  @Override
  public boolean optimize(double[] thetas, Regularization regularization,
      IterationCallback iterationCallback) {
    objectiveFunction.setRegularization(regularization);

    StanfordCoreNLPQNMinimizerLite lbfgs =
        new StanfordCoreNLPQNMinimizerLite(NUMBER_OF_PREVIOUS_ESTIMATIONS, USE_ROBUST_OPTIONS);
    lbfgs.shutUp();
    double[] optimalThetas = lbfgs.minimize(objectiveFunction, thetas,
        MPGConfig.LBFGS_MAX_NUMBER_OF_ITERATIONS, iterationCallback);

    Assert.isTrue(optimalThetas.length == thetas.length);
    System.arraycopy(optimalThetas, 0, thetas, 0, optimalThetas.length);

    LOGGER.info(lbfgs.getState());
    return lbfgs.wasSuccessful();
  }

}
