package edu.uic.cs.purposeful.mpg.learning.binary;

import java.io.File;
import java.util.BitSet;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.MathUtils;
import org.apache.log4j.Logger;

import com.google.common.annotations.VisibleForTesting;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.Problem;
import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.common.assertion.PurposefulBaseException;
import edu.uic.cs.purposeful.common.reflect.ClassFactory;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.learning.MaximizerPredictor;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.NumericalOptimizer;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.objective.impl.MinimizationObjectiveFunctionImpl;
import edu.uic.cs.purposeful.mpg.target.binary.AbstractBinaryOptimizationTarget;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public class MPGBinaryClassifier
    extends MaximizerPredictor<BitSet, Pair<double[], LinkedSparseMatrix>> {
  private static final Logger LOGGER = Logger.getLogger(MPGBinaryClassifier.class);

  private final double targetClassValue;
  private double[] thetas;

  public MPGBinaryClassifier(
      Class<? extends AbstractBinaryOptimizationTarget> optimizationTargetClass,
      double targetClassValue) {
    super(optimizationTargetClass);
    this.targetClassValue = targetClassValue;
  }

  public double[] learn(File trainingDataFile, Regularization regularization) {
    Problem trainingData = null;
    try {
      trainingData = Problem.readFromFile(trainingDataFile, MPGConfig.BIAS_FEATURE_VALUE);
    } catch (Exception e) {
      throw new PurposefulBaseException(e);
    }
    trainingData = binarizeDataset(trainingData);

    thetas = initializeThetas(trainingData, regularization);
    if (LOGGER.isInfoEnabled()) {
      LOGGER.info("Initialized thetas: " + Misc.toDisplay(thetas));
    }

    NumericalOptimizer numericalOptimizer =
        ClassFactory.getInstance(MPGConfig.NUMERICAL_OPTIMIZER_IMPLEMENTATION);
    LOGGER.warn("Using numerical optimizer implementation: " + numericalOptimizer.getClass());
    numericalOptimizer.setMinimizationObjectiveFunction(new MinimizationObjectiveFunctionImpl<>(
        optimizationTargetClass, Pair.of(trainingData.y, createFeatureMatrix(trainingData))));
    boolean converged = numericalOptimizer.optimize(thetas, regularization);
    LOGGER.warn("Finish optimization using numerical optimizer, converged=" + converged);

    return thetas;
  }

  private LinkedSparseMatrix createFeatureMatrix(Problem data) {
    LinkedSparseMatrix featureMatrix = new LinkedSparseMatrix(data.l, data.n);
    int rowIndex = 0;
    for (Feature[] features : data.x) {
      for (Feature feature : features) {
        featureMatrix.set(rowIndex, feature.getIndex() - 1, feature.getValue());
      }
      rowIndex++;
    }
    return featureMatrix;
  }

  public void writeModel(File modelFile) {
    writeModelToFile(modelFile, thetas);
  }

  public void loadModel(File modelFile) {
    thetas = loadModelFromFile(modelFile);
  }

  @VisibleForTesting
  double[] alignDataset(Problem data, double[] thetas) {
    double[] newThetas = new double[data.n];

    int smallerLength = Math.min(thetas.length, newThetas.length);
    if (MPGConfig.BIAS_FEATURE_VALUE >= 0) { // the last one is the bias feature
      System.arraycopy(thetas, 0, newThetas, 0, smallerLength - 1);
      newThetas[newThetas.length - 1] = thetas[thetas.length - 1];
    } else {
      System.arraycopy(thetas, 0, newThetas, 0, smallerLength);
    }

    LOGGER.warn("Data set contains a different number of features [=" + data.n
        + "] from the number of learned thetas [=" + thetas.length
        + "], the thetas vector is aligned to size=" + newThetas.length);
    return newThetas;
  }

  private Problem binarizeDataset(Problem trainingData) {
    boolean hasTargetClassValue = false;
    for (int index = 0; index < trainingData.l; index++) {
      if (MathUtils.equals(trainingData.y[index], targetClassValue)) {
        trainingData.y[index] = AbstractBinaryOptimizationTarget.BINARY_VALUE_ONE;
        hasTargetClassValue = true;
      } else {
        trainingData.y[index] = AbstractBinaryOptimizationTarget.BINARY_VALUE_ZERO;
      }
    }

    Assert.isTrue(hasTargetClassValue,
        "No target calss with value=[" + targetClassValue + "] in data set.");
    return trainingData;
  }

  private double[] initializeThetas(Problem trainingData, Regularization regularization) {
    if (MPGConfig.LEARN_INITIAL_THETAS) {
      LogisticRegressionBinary logisticRegression =
          new LogisticRegressionBinary((int) AbstractBinaryOptimizationTarget.BINARY_VALUE_ONE);
      return logisticRegression.learnWeights(trainingData, regularization);
    } else {
      return new double[trainingData.n];
    }
  }

  public Prediction<BitSet> predict(File testDataFile) {
    Assert.notNull(thetas, "Learn or load the model first.");
    return predict(testDataFile, thetas);
  }

  public Prediction<BitSet> predict(File testDataFile, double[] thetas) {
    Problem testData = null;
    try {
      testData = Problem.readFromFile(testDataFile, MPGConfig.BIAS_FEATURE_VALUE);
    } catch (Exception e) {
      throw new PurposefulBaseException(e);
    }

    // make sure thetas and test data are aligned
    if (thetas.length != testData.n) {
      thetas = alignDataset(testData, thetas);
    }

    testData = binarizeDataset(testData);

    return predict(Pair.of(testData.y, createFeatureMatrix(testData)), thetas);
  }
}
