package edu.uic.cs.purposeful.mpg.optimizer.numerical.lbfgs;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;

import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.IterationCallback;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.NumericalOptimizer;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.lbfgs.riso.LBFGSOptimizer;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.lbfgs.riso.LBFGSOptimizer.ExceptionWithIflag;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.objective.MinimizationObjectiveFunction;

public class RosiLBFGSWrapper implements NumericalOptimizer {
  private static final Logger LOGGER = Logger.getLogger(RosiLBFGSWrapper.class);

  private static final int[] OUTPUT_OPTIONS = specifyOutputOptions();

  // The number of corrections used in the BFGS update. Values of m less than 3 are not recommended;
  // large values of m will result in excessive computing time. 3 <= m <= 7 is recommended.
  // Restriction: m > 0.
  private static final int NUMBER_OF_CORRECTIONS = 5;

  // // An estimate of the machine precision. The line search routine will terminate if the relative
  // // width of the interval of uncertainty is less than this value.
  // private static final double MACHINE_PRECISION = 1.0e-16;

  private static int[] specifyOutputOptions() {
    int[] iprint = new int[2];
    /**
     * iprint[0] specifies the frequency of the output: <br>
     * iprint[0] < 0: no output is generated, <br>
     * iprint[0] = 0: output only at first and last iteration, <br>
     * iprint[0] > 0: output every iprint[0] iterations.
     */
    iprint[0] = -1; // no output
    /**
     * iprint[1] = 0: iteration count, number of function evaluations, function value, norm of the
     * gradient, and steplength, <br>
     * iprint[1] = 1: same as iprint[1]=0, plus vector of variables and gradient vector at the
     * initial point, <br>
     * iprint[1] = 2: same as iprint[1]=1, plus vector of variables, <br>
     * iprint[1] = 3: same as iprint[1]=2, plus gradient vector.
     */
    iprint[1] = 0;
    return iprint;
  }

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

    // stores the values of the diagonal matrix Hk0
    double[] diag = new double[thetas.length];

    LBFGSOptimizer lbfgs = new LBFGSOptimizer();
    int[] terminationFlag = new int[1];

    int iterationIndex = 0;
    for (; iterationIndex < MPGConfig.LBFGS_MAX_NUMBER_OF_ITERATIONS
        && (iterationIndex == 0 || terminationFlag[0] != 0); iterationIndex++) {
      Pair<Double, double[]> objectiveValueAndGradients =
          objectiveFunction.getValueAndGradients(thetas);
      double objectiveValue = objectiveValueAndGradients.getLeft();
      double[] objectiveGradients = objectiveValueAndGradients.getRight();
      try {
        lbfgs.lbfgs(thetas.length, NUMBER_OF_CORRECTIONS, thetas, objectiveValue,
            objectiveGradients, /* diagco, provide the diagonal matrix Hk0 */false, diag,
            OUTPUT_OPTIONS, MPGConfig.LBFGS_TERMINATE_GRADIENT_TOLERANCE,
            MPGConfig.LBFGS_TERMINATE_VALUE_TOLERANCE, terminationFlag);
      } catch (ExceptionWithIflag e) {
        LOGGER.warn(e);
        return false;
      }

      if (iterationCallback != null) {
        try {
          iterationCallback.call(iterationIndex, thetas);
        } catch (Exception e) {
          LOGGER.error("", e);
        }
      }
    }
    // not converged if >= maxNumIterations
    return iterationIndex < MPGConfig.LBFGS_MAX_NUMBER_OF_ITERATIONS;
  }
}
