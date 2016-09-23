package edu.uic.cs.purposeful.mpg.optimizer.numerical.objective.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.common.assertion.PurposefulBaseException;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.FeatureWiseRegularization;
import edu.uic.cs.purposeful.mpg.common.Norm;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.optimizer.game.ZeroSumGameSolver;
import edu.uic.cs.purposeful.mpg.optimizer.game.impl.DoubleOracleGameSolver;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.objective.MinimizationObjectiveFunction;
import edu.uic.cs.purposeful.mpg.target.OptimizationTarget;
import net.mintern.primitive.pair.ObjDoublePair;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;
import no.uib.cipr.matrix.sparse.SparseVector;

public class MinimizationObjectiveFunctionImpl<Permutation, InitialData>
    implements MinimizationObjectiveFunction {
  private static final Logger LOGGER = Logger.getLogger(MinimizationObjectiveFunctionImpl.class);

  private static enum ObjectiveTask {
    VALUE, GRADIENTS, BOTH;
  }

  private static class ValueAndGradients {
    private final Double rawObjectiveValue;
    private final Double valueRegularization;
    private final Double objectiveValue;
    private final double[] rawObjectiveGradients;
    private final double[] gradientRegularizations;
    private final double[] objectiveGradients;

    private ValueAndGradients(Double rawObjectiveValue, Double valueRegularization,
        Double objectiveValue, double[] rawObjectiveGradients, double[] gradientRegularizations,
        double[] objectiveGradients) {
      this.rawObjectiveValue = rawObjectiveValue;
      this.valueRegularization = valueRegularization;
      this.objectiveValue = objectiveValue;
      this.rawObjectiveGradients = rawObjectiveGradients;
      this.gradientRegularizations = gradientRegularizations;
      this.objectiveGradients = objectiveGradients;
    }
  }

  private static class InitializeOptimizationTarget<Permutation, InitialData>
      implements Callable<OptimizationTarget<Permutation, InitialData>> {
    private final int index;
    private final OptimizationTarget<Permutation, InitialData> optimizationTarget;
    private final InitialData initialData;

    private InitializeOptimizationTarget(
        Class<? extends OptimizationTarget<Permutation, InitialData>> optimizationTargetClass,
        InitialData initialData, int index) {
      try {
        this.optimizationTarget = optimizationTargetClass.newInstance();
      } catch (Exception e) {
        throw new PurposefulBaseException(e);
      }
      this.initialData = initialData;
      this.index = index;
    }

    @Override
    public OptimizationTarget<Permutation, InitialData> call() throws Exception {
      if (MPGConfig.SHOW_RUNNING_TRACING) {
        System.err
            .println("[" + Thread.currentThread().getId() + "] initializing instance data...");
      }
      optimizationTarget.initialize(initialData, true);
      if (MPGConfig.SHOW_RUNNING_TRACING) {
        System.err.println("[" + Thread.currentThread().getId() + "] #" + index + " is done.");
      }
      return optimizationTarget;
    }
  }

  private static class ComputeInstanceRawObjectiveGradientsAndValue<Permutation, InitialData>
      implements Callable<ObjDoublePair<Vector>> {
    private final OptimizationTarget<Permutation, InitialData> optimizationTarget;
    private final double[] thetas;
    private final ObjectiveTask objectiveTask;

    private ComputeInstanceRawObjectiveGradientsAndValue(
        OptimizationTarget<Permutation, InitialData> optimizationTarget, double[] thetas,
        ObjectiveTask objectiveTask) {
      this.optimizationTarget = optimizationTarget;
      this.thetas = thetas;
      this.objectiveTask = objectiveTask;
    }

    @Override
    public ObjDoublePair<Vector> call() throws Exception {
      ZeroSumGameSolver<Permutation> gameSolver = new DoubleOracleGameSolver<>(optimizationTarget);
      if (MPGConfig.SHOW_RUNNING_TRACING) {
        System.err.print("[" + Thread.currentThread().getId() + " " + objectiveTask + "]");
      }
      gameSolver.solve(thetas, optimizationTarget.getGoldenPermutation());

      Vector goldenFeatureValues = optimizationTarget.getGoldenFeatureValues();

      double rawObjectiveValue = 0.0;
      if (objectiveTask == ObjectiveTask.BOTH || objectiveTask == ObjectiveTask.VALUE) {
        double maximizerValue = gameSolver.getMaximizerValue();
        rawObjectiveValue = computeRawObjectiveValue(goldenFeatureValues, thetas, maximizerValue);
      }

      Vector rawObjectiveGradients = null;
      if (objectiveTask == ObjectiveTask.BOTH || objectiveTask == ObjectiveTask.GRADIENTS) {
        double[] minimizerProbabilities = gameSolver.getMinimizerProbabilities();
        LinkedHashSet<Permutation> minimizerPermutations = gameSolver.getMinimizerPermutations();
        Assert.isTrue(minimizerProbabilities.length == minimizerPermutations.size());
        Vector minimizerFeatureValueExpectations = optimizationTarget
            .computeExpectedFeatureValues(minimizerProbabilities, minimizerPermutations);
        rawObjectiveGradients =
            computeRawGradients(goldenFeatureValues, minimizerFeatureValueExpectations);
      }

      return ObjDoublePair.of(rawObjectiveGradients, rawObjectiveValue);
    }

    private double computeRawObjectiveValue(Vector goldenFeatureValues, double[] thetas,
        double maximizerValue) {
      Assert.isTrue(goldenFeatureValues.size() == thetas.length);
      double empiricalPotential = goldenFeatureValues.dot(new DenseVector(thetas, false));
      return -empiricalPotential - maximizerValue;
    }

    private Vector computeRawGradients(Vector goldenFeatureValues,
        Vector minimizerFeatureValueExpectations) {
      Assert.isTrue(goldenFeatureValues.size() == minimizerFeatureValueExpectations.size());

      Vector rawGradients;

      if (goldenFeatureValues instanceof SparseVector
          && minimizerFeatureValueExpectations instanceof SparseVector) {
        int usedSizeGoldenFeatureValues = ((SparseVector) goldenFeatureValues).getUsed();
        int usedSizeMinimizerFeatureValueExpectations =
            ((SparseVector) minimizerFeatureValueExpectations).getUsed();

        rawGradients = new SparseVector(goldenFeatureValues.size());
        if (usedSizeGoldenFeatureValues < usedSizeMinimizerFeatureValueExpectations) {
          for (VectorEntry e : goldenFeatureValues) {
            rawGradients.set(e.index(), minimizerFeatureValueExpectations.get(e.index()) - e.get());
          }
        } else {
          for (VectorEntry e : minimizerFeatureValueExpectations) {
            rawGradients.set(e.index(), e.get() - goldenFeatureValues.get(e.index()));
          }
        }
      } else {
        double[] rawGradientsArray = new double[goldenFeatureValues.size()];
        for (int index = 0; index < goldenFeatureValues.size(); index++) {
          rawGradientsArray[index] =
              minimizerFeatureValueExpectations.get(index) - goldenFeatureValues.get(index);
        }
        rawGradients = new DenseVector(rawGradientsArray, false);
      }

      return rawGradients;
    }
  }

  private Regularization regularization;
  private FeatureWiseRegularization featureWiseRegularization;
  private final List<OptimizationTarget<Permutation, InitialData>> optimizationTargets;
  private final int[] allInstanceIndices;

  public MinimizationObjectiveFunctionImpl(
      Class<? extends OptimizationTarget<Permutation, InitialData>> optimizationTargetClass,
      InitialData initialData) {
    this(optimizationTargetClass, Collections.singletonList(initialData));
  }

  public MinimizationObjectiveFunctionImpl(
      Class<? extends OptimizationTarget<Permutation, InitialData>> optimizationTargetClass,
      List<InitialData> initialDataList) {
    ExecutorService threadPool = Executors.newFixedThreadPool(MPGConfig.THREAD_POOL_SIZE);
    CompletionService<OptimizationTarget<Permutation, InitialData>> completionService =
        new ExecutorCompletionService<>(threadPool);

    try {
      int index = 0;
      for (InitialData initialData : initialDataList) {
        completionService.submit(
            new InitializeOptimizationTarget<>(optimizationTargetClass, initialData, index++));
      }

      optimizationTargets = new ArrayList<>(initialDataList.size());
      allInstanceIndices = new int[initialDataList.size()];
      for (index = 0; index < initialDataList.size(); index++) {
        optimizationTargets.add(completionService.take().get());
        allInstanceIndices[index] = index;
      }
    } catch (Exception e) {
      throw new PurposefulBaseException(e);
    } finally {
      threadPool.shutdown();
    }
  }

  private ValueAndGradients computeValueAndGradientsInParallel(double[] thetas,
      int[] indicesInBatch, ObjectiveTask objectiveTask) {
    ExecutorService threadPool = Executors.newFixedThreadPool(MPGConfig.THREAD_POOL_SIZE);
    CompletionService<ObjDoublePair<Vector>> completionService =
        new ExecutorCompletionService<>(threadPool);

    double[] rawObjectiveGradientsSum = new double[thetas.length];
    double rawObjectiveValueSum = 0.0;
    try {
      for (int index : indicesInBatch) {
        OptimizationTarget<Permutation, InitialData> optimizationTarget =
            optimizationTargets.get(index);
        completionService.submit(new ComputeInstanceRawObjectiveGradientsAndValue<>(
            optimizationTarget, thetas, objectiveTask));
      }

      for (int targetIndex = 0; targetIndex < indicesInBatch.length; targetIndex++) {
        ObjDoublePair<Vector> rawObjectiveGradientsAndValue = completionService.take().get();

        if (objectiveTask == ObjectiveTask.BOTH || objectiveTask == ObjectiveTask.VALUE) {
          double rawObjectiveValue = rawObjectiveGradientsAndValue.getRight();
          rawObjectiveValueSum += rawObjectiveValue;
        }

        if (objectiveTask == ObjectiveTask.BOTH || objectiveTask == ObjectiveTask.GRADIENTS) {
          Vector rawObjectiveGradients = rawObjectiveGradientsAndValue.getLeft();
          for (VectorEntry entry : rawObjectiveGradients) {
            rawObjectiveGradientsSum[entry.index()] += entry.get();
          }
        }
      }
    } catch (Exception e) {
      throw new PurposefulBaseException(e);
    } finally {
      threadPool.shutdown();
    }

    // /////////////////////////////////////////////////////////////////////////
    Double rawObjectiveValue = null;
    Double valueRegularization = null;
    Double objectiveValue = null;
    // compute the average among all the training instances
    if (objectiveTask == ObjectiveTask.BOTH || objectiveTask == ObjectiveTask.VALUE) {
      rawObjectiveValue = rawObjectiveValueSum;
      valueRegularization = computeValueRegularization(thetas);
      // since we are minimize objectiveValue, add valueRegularization to penalize large thetas
      objectiveValue = rawObjectiveValue + valueRegularization;
    }

    double[] rawObjectiveGradients = null;
    double[] gradientRegularizations = null;
    double[] objectiveGradients = null;
    if (objectiveTask == ObjectiveTask.BOTH || objectiveTask == ObjectiveTask.GRADIENTS) {
      rawObjectiveGradients = rawObjectiveGradientsSum;
      gradientRegularizations = computeGradientRegularizations(thetas);
      objectiveGradients = new double[thetas.length];
      for (int index = 0; index < thetas.length; index++) {
        objectiveGradients[index] = rawObjectiveGradients[index] + gradientRegularizations[index];
      }
    }

    if (MPGConfig.SHOW_RUNNING_TRACING) {
      System.err.println("[>" + objectiveTask + "<]");
    }
    return new ValueAndGradients(rawObjectiveValue, valueRegularization, objectiveValue,
        rawObjectiveGradients, gradientRegularizations, objectiveGradients);
  }

  @Override
  public int getNumberOfInstances() {
    return allInstanceIndices.length;
  }

  @Override
  public Pair<Double, double[]> getValueAndGradients(double[] thetas, int[] indicesInBatch) {
    ValueAndGradients valueAndGradients =
        computeValueAndGradientsInParallel(thetas, indicesInBatch, ObjectiveTask.BOTH);

    if (LOGGER.isInfoEnabled()) {
      double rawGradientsNorm2 = norm2(valueAndGradients.rawObjectiveGradients);
      double gradientRegNorm2 = norm2(valueAndGradients.gradientRegularizations);
      double gradientNorm2 = norm2(valueAndGradients.objectiveGradients);
      double thetasNorm2 = norm2(thetas);
      double gradientThetasRatio = gradientNorm2 / Math.max(1.0, thetasNorm2);
      LOGGER.info("ValueAndGradients | (gradient/thetas)=" + gradientThetasRatio + ", rawValue="
          + valueAndGradients.rawObjectiveValue + ", rawGradients=" + rawGradientsNorm2
          + ", valueReg=" + valueAndGradients.valueRegularization + ", gradientReg="
          + gradientRegNorm2 + ", value=" + valueAndGradients.objectiveValue + ", gradient="
          + gradientNorm2);
    }

    return Pair.of(valueAndGradients.objectiveValue, valueAndGradients.objectiveGradients);
  }

  @Override
  public Pair<Double, double[]> getValueAndGradients(double[] thetas) {
    return getValueAndGradients(thetas, allInstanceIndices);
  }

  @Override
  public double getValue(double[] thetas, int[] indicesInBatch) {
    ValueAndGradients valueAndGradients =
        computeValueAndGradientsInParallel(thetas, indicesInBatch, ObjectiveTask.VALUE);

    if (LOGGER.isInfoEnabled()) {
      LOGGER.info("Value | rawValue=" + valueAndGradients.rawObjectiveValue + ", valueReg="
          + valueAndGradients.valueRegularization + ", value=" + valueAndGradients.objectiveValue);
    }

    return valueAndGradients.objectiveValue;
  }

  @Override
  public double getValue(double[] thetas) {
    return getValue(thetas, allInstanceIndices);
  }

  @Override
  public double[] getGradients(double[] thetas, int[] indicesInBatch) {
    ValueAndGradients valueAndGradients =
        computeValueAndGradientsInParallel(thetas, indicesInBatch, ObjectiveTask.GRADIENTS);

    if (LOGGER.isInfoEnabled()) {
      double rawGradientsNorm2 = norm2(valueAndGradients.rawObjectiveGradients);
      double gradientRegNorm2 = norm2(valueAndGradients.gradientRegularizations);
      double gradientNorm2 = norm2(valueAndGradients.objectiveGradients);
      double thetasNorm2 = norm2(thetas);
      double gradientThetasRatio = gradientNorm2 / Math.max(1.0, thetasNorm2);
      LOGGER.info("Gradients | (gradient/thetas)=" + gradientThetasRatio + ", rawGradients="
          + rawGradientsNorm2 + ", gradientReg=" + gradientRegNorm2 + ", gradient="
          + gradientNorm2);
    }

    return valueAndGradients.objectiveGradients;
  }

  @Override
  public double[] getGradients(double[] thetas) {
    return getGradients(thetas, allInstanceIndices);
  }

  @Override
  public void setRegularization(Regularization regularization) {
    Assert.isNull(featureWiseRegularization,
        "Feature-wise regularization has been setted, can't use the (uniform) regularization.");
    this.regularization = regularization;
  }

  @Override
  public void setRegularization(FeatureWiseRegularization featureWiseRegularization) {
    Assert.isNull(regularization,
        "(Uniform) regularization has been setted, can't use the feature-wise regularization.");
    this.featureWiseRegularization = featureWiseRegularization;
  }

  private double computeValueRegularization(double[] thetas) {
    if (regularization == null && featureWiseRegularization == null) {
      return 0.0;
    }

    // do not regularize bias feature's weight
    if (!MPGConfig.REGULARIZE_BIAS_FEATURE && MPGConfig.BIAS_FEATURE_VALUE >= 0) {
      // the last one is the bias feature
      thetas = thetas.clone();
      thetas[thetas.length - 1] = 0; // give zero since we don't regularize it
    }

    double regularizationScore = 0.0;

    if (regularization != null) { // uniform regularization
      if (regularization.getNorm() == Norm.L1) {
        regularizationScore = regularization.getParameter() * norm1(thetas);
      } else if (regularization.getNorm() == Norm.L2) {
        regularizationScore = regularization.getParameter() * norm2Square(thetas) / 2;
      } else {
        Assert.canNeverHappen();
      }
    } else { // feature-wise regularization
      double[] regParameters = featureWiseRegularization.getParameters();
      Assert.isTrue(regParameters.length == thetas.length, "regParameters.length["
          + regParameters.length + "] != thetas.length[" + thetas.length + "].");

      if (featureWiseRegularization.getNorm() == Norm.L1) {
        regularizationScore = elementWiseRegularizedNorm1(thetas, regParameters);
      } else if (featureWiseRegularization.getNorm() == Norm.L2) {
        regularizationScore = elementWiseRegularizedNorm2Square(thetas, regParameters) / 2;
      } else {
        Assert.canNeverHappen();
      }
    }
    return regularizationScore;
  }

  private double[] computeGradientRegularizations(double[] thetas) {
    double[] regularizationScores = new double[thetas.length];
    if (regularization == null && featureWiseRegularization == null) {
      return regularizationScores;
    }

    int numOfRegularizedWeights = thetas.length;
    // do not regularize bias feature's weight
    if (!MPGConfig.REGULARIZE_BIAS_FEATURE && MPGConfig.BIAS_FEATURE_VALUE >= 0) {
      // the last one is the bias feature
      numOfRegularizedWeights--;
    }

    if (regularization != null) { // uniform regularization
      if (regularization.getNorm() == Norm.L1) {
        for (int index = 0; index < numOfRegularizedWeights; index++) {
          regularizationScores[index] = regularization.getParameter() * Math.signum(thetas[index]);
        }
      } else if (regularization.getNorm() == Norm.L2) {
        for (int index = 0; index < numOfRegularizedWeights; index++) {
          regularizationScores[index] = regularization.getParameter() * thetas[index];
        }
      } else {
        Assert.canNeverHappen();
      }
    } else { // feature-wise regularization
      double[] regParameters = featureWiseRegularization.getParameters();
      Assert.isTrue(regParameters.length == thetas.length, "regParameters.length["
          + regParameters.length + "] != thetas.length[" + thetas.length + "].");

      if (featureWiseRegularization.getNorm() == Norm.L1) {
        for (int index = 0; index < numOfRegularizedWeights; index++) {
          regularizationScores[index] = regParameters[index] * Math.signum(thetas[index]);
        }
      } else if (featureWiseRegularization.getNorm() == Norm.L2) {
        for (int index = 0; index < numOfRegularizedWeights; index++) {
          regularizationScores[index] = regParameters[index] * thetas[index];
        }
      } else {
        Assert.canNeverHappen();
      }
    }

    return regularizationScores;
  }

  private double norm1(double[] values) {
    double sum = 0;
    for (double value : values) {
      sum += Math.abs(value);
    }
    return sum;
  }

  private double elementWiseRegularizedNorm1(double[] values, double[] regs) {
    double regularizedSum = 0;
    for (int index = 0; index < values.length; index++) {
      regularizedSum += Math.abs(values[index]) * regs[index];
    }
    return regularizedSum;
  }

  private double norm2(double[] values) {
    return Math.sqrt(norm2Square(values));
  }

  private double norm2Square(double[] values) {
    double norm = 0;
    for (double value : values) {
      norm += value * value;
    }
    return norm;
  }

  private double elementWiseRegularizedNorm2Square(double[] values, double[] regs) {
    double norm = 0;
    for (int index = 0; index < values.length; index++) {
      double value = values[index];
      norm += value * value * regs[index];
    }
    return norm;
  }
}
