package edu.uic.cs.purposeful.mpg.optimizer.numerical.adadelta;

import org.apache.commons.lang3.tuple.Pair;

import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;
import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.IterationCallback;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.NumericalOptimizer;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.adadelta.jhu.AdaDeltaSGDLite;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.objective.MinimizationObjectiveFunction;

public class JhuAdaDeltaWrapper implements NumericalOptimizer {
  private static class DifferentiableBatchFunctionAdapter implements DifferentiableBatchFunction {
    private final MinimizationObjectiveFunction objectiveFunction;
    private final int numDimensions;

    private DifferentiableBatchFunctionAdapter(MinimizationObjectiveFunction objectiveFunction,
        int numDimensions) {
      this.objectiveFunction = objectiveFunction;
      this.numDimensions = numDimensions;
    }

    @Override
    public int getNumExamples() {
      return objectiveFunction.getNumberOfInstances();
    }

    @Override
    public int getNumDimensions() {
      return numDimensions;
    }

    @Override
    public double getValue(IntDoubleVector point) {
      Assert.isTrue(MPGConfig.ADADELTA_USE_TERMINATE_VALUE,
          "Config 'adadelta_use_terminate_value=false', no value should be computed.");

      return objectiveFunction.getValue(point.toNativeArray());
    }

    @Override
    public IntDoubleVector getGradient(IntDoubleVector point) {
      return new IntDoubleDenseVector(objectiveFunction.getGradients(point.toNativeArray()));
    }

    @Override
    public ValueGradient getValueGradient(IntDoubleVector point) {
      Assert.isTrue(MPGConfig.ADADELTA_USE_TERMINATE_VALUE,
          "Config 'adadelta_use_terminate_value=false', no value should be computed.");

      Pair<Double, double[]> valueAndGradients =
          objectiveFunction.getValueAndGradients(point.toNativeArray());
      return new ValueGradient(valueAndGradients.getLeft(),
          new IntDoubleDenseVector(valueAndGradients.getRight()));
    }

    @Override
    public double getValue(IntDoubleVector point, int[] batch) {
      Assert.isTrue(MPGConfig.ADADELTA_USE_TERMINATE_VALUE,
          "Config 'adadelta_use_terminate_value=false', no value should be computed.");

      return objectiveFunction.getValue(point.toNativeArray(), batch);
    }

    @Override
    public IntDoubleVector getGradient(IntDoubleVector point, int[] batch) {
      return new IntDoubleDenseVector(objectiveFunction.getGradients(point.toNativeArray(), batch));
    }

    @Override
    public ValueGradient getValueGradient(IntDoubleVector point, int[] batch) {
      Assert.isTrue(MPGConfig.ADADELTA_USE_TERMINATE_VALUE,
          "Config 'adadelta_use_terminate_value=false', no value should be computed.");

      Pair<Double, double[]> valueAndGradients =
          objectiveFunction.getValueAndGradients(point.toNativeArray(), batch);
      return new ValueGradient(valueAndGradients.getLeft(),
          new IntDoubleDenseVector(valueAndGradients.getRight()));
    }
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

    AdaDeltaSGDLite adaDeltaSGDLite = new AdaDeltaSGDLite();

    return adaDeltaSGDLite.minimize(
        new DifferentiableBatchFunctionAdapter(objectiveFunction, thetas.length),
        new IntDoubleDenseVector(thetas), iterationCallback);
  }
}
