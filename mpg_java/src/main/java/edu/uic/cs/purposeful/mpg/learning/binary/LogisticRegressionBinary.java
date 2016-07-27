package edu.uic.cs.purposeful.mpg.learning.binary;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.MathUtils;
import org.apache.log4j.Logger;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.common.Regularization.Norm;
import edu.uic.cs.purposeful.mpg.learning.LogisticRegressionHelper;
import edu.uic.cs.purposeful.mpg.learning.MaximizerPredictor.Prediction;

public class LogisticRegressionBinary
    implements LogisticRegressionHelper<List<Prediction<Integer>>, Problem> {
  private static final Logger LOGGER = Logger.getLogger(LogisticRegressionBinary.class);

  private final Integer targetLabel;

  public LogisticRegressionBinary() {
    this(null);
  }

  /**
   * @param targetLabel it is only required for {@link #learnWeights(Problem, Regularization)}
   *        method. Only weights for this specified label is returned.
   */
  LogisticRegressionBinary(Integer targetLabel) {
    this.targetLabel = targetLabel;
  }

  @Override
  public Model learnModel(Problem dataSet, Regularization regularization) {
    double regularizationParameter = regularization.getParameter();
    if (MathUtils.equals(regularizationParameter, 0)) {
      LOGGER.warn(
          "[regularizationParameter==0], in such case, Logistic Regression (LibLinear) would learn all zeros as its weights.");
      regularizationParameter = Double.MIN_VALUE; // use very small number to approximate zero
    } else {
      // LR's regularization is the reciprocal of the regularization in MPG
      regularizationParameter = 1.0 / regularizationParameter;
    }

    Parameter param = null;
    if (regularization.getNorm() == Norm.L1) {
      param = new Parameter(SolverType.L1R_LR, regularizationParameter,
          MPGConfig.LOGISTIC_REGRESSION_STOPPING_CRITERION);
    } else if (regularization.getNorm() == Norm.L2) {
      param = new Parameter(SolverType.L2R_LR, regularizationParameter,
          MPGConfig.LOGISTIC_REGRESSION_STOPPING_CRITERION);
    } else {
      Assert.canNeverHappen();
    }

    Linear.disableDebugOutput();
    return Linear.train(dataSet, param);
  }

  @Override
  public double[] learnWeights(Problem dataSet, Regularization regularization) {
    Assert.notNull(targetLabel,
        "Should invoke the constructor with the Integer parameter first to specify the target label.");

    Model model = learnModel(dataSet, regularization);
    Assert.isTrue(model.getNrClass() == 2);
    int[] possibleLabels = model.getLabels();
    Assert.isTrue(possibleLabels.length == 2);
    Assert.isTrue(ArrayUtils.contains(possibleLabels, targetLabel),
        "Data set doesn't contain target label [" + targetLabel + "].");

    double[] weights = model.getFeatureWeights();
    // learned weights are for tags[0]
    if (possibleLabels[1] == targetLabel) {
      for (int index = 0; index < weights.length; index++) {
        weights[index] = -weights[index];
      }
    } else {
      Assert.isTrue(possibleLabels[0] == targetLabel);
    }

    LOGGER.warn("Weights initialized from Logistic Regression: " + Misc.toDisplay(weights));
    return weights;
  }

  @Override
  public List<Prediction<Integer>> predict(Problem dataSet, Model model) {
    Linear.disableDebugOutput();

    Assert.isTrue(model.getNrClass() == 2);
    int[] possibleLabels = model.getLabels();
    Assert.isTrue(possibleLabels.length == 2);

    List<Prediction<Integer>> predictions = new ArrayList<>(dataSet.l);

    for (int bitIndex = 0; bitIndex < dataSet.l; bitIndex++) {
      Feature[] features = dataSet.x[bitIndex];
      int goldenLabel = (int) dataSet.y[bitIndex];

      double[] probabilities = new double[2]; // binary
      int predictedLabel = (int) Linear.predictProbability(model, features, probabilities);

      double probability = Double.NaN;
      if (predictedLabel == possibleLabels[0]) {
        probability = probabilities[0];
      } else if (predictedLabel == possibleLabels[1]) {
        probability = probabilities[1];
      } else {
        Assert.canNeverHappen();
      }
      Assert.isTrue(probability >= 0.5);

      Prediction<Integer> prediction = new Prediction<Integer>(bitIndex, predictedLabel,
          probability, goldenLabel, predictedLabel == goldenLabel ? 1 : 0);

      predictions.add(prediction);
    }

    return predictions;
  }
}
