/**
 * Copyright (c) 2015 University of Illinois at Chicago to Present. All rights reserved.
 */
package edu.uic.cs.purposeful.mpg.learning.binary;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.File;
import java.util.List;

import org.apache.commons.math3.util.MathUtils;
import org.junit.Test;

import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Problem;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.common.ValuePrecision;
import edu.uic.cs.purposeful.mpg.learning.MaximizerPredictor.Prediction;

public class TestLogisticRegressionBinary {

  @Test
  public void testPredict1() throws Exception {
    testPredict("TestLogisticRegressionBinary_1.train");
  }

  @Test
  public void testPredict2() throws Exception {
    testPredict("TestLogisticRegressionBinary_2.train");
  }

  @Test
  public void testPredict3() throws Exception {
    testPredict("TestLogisticRegressionBinary_3.train");
  }

  @Test
  public void testPredict4() throws Exception {
    testPredict("TestLogisticRegressionBinary_4.train");
  }

  private void testPredict(String dataSetFileName) throws Exception {
    System.out.println(dataSetFileName);

    File file = new File(TestLogisticRegressionBinary.class.getResource(dataSetFileName).toURI());
    Problem dataSet = Problem.readFromFile(file, -1);

    LogisticRegressionBinary logisticRegression = new LogisticRegressionBinary();
    Model model = logisticRegression.learnModel(dataSet, Regularization.l2(1));

    List<Prediction<Integer>> predictions = logisticRegression.predict(dataSet, model);
    for (Prediction<Integer> prediction : predictions) {
      System.out.printf("Index=%d, prediction=%d, gold=%d, probability=%f, score=%f\n",
          prediction.getIndex(), prediction.getPredictionPermutation(),
          +prediction.getGoldenPermutation(), prediction.getProbability(), prediction.getScore());
      assertEquals(prediction.getPredictionPermutation(), prediction.getGoldenPermutation());
      assertEquals(1.0, prediction.getScore(), 0); // when correctly predicted, score should be 1
    }

    System.out.println("=====================================================");
  }

  @Test
  public void testLearnWeights() throws Exception {
    File file1 = new File(TestLogisticRegressionBinary.class
        .getResource("TestLogisticRegressionBinary_1.train").toURI());
    Problem dataSet1 = Problem.readFromFile(file1, -1);

    File file2 = new File(TestLogisticRegressionBinary.class
        .getResource("TestLogisticRegressionBinary_2.train").toURI());
    Problem dataSet2 = Problem.readFromFile(file2, -1);

    Regularization regularization = Regularization.l2(1);

    LogisticRegressionBinary logisticRegression1 = new LogisticRegressionBinary(1);
    double[] weights11 = logisticRegression1.learnWeights(dataSet1, regularization);
    double[] weights12 = logisticRegression1.learnWeights(dataSet2, regularization);
    assertArrayEquals(weights11, weights12, ValuePrecision.POINT_8_ZEROS_ONE.getValuePrecision());

    double[] y1 = dataSet1.y;
    for (int index = 0; index < y1.length; index++) {
      if (MathUtils.equals(y1[index], 1)) {
        y1[index] = 0;
        System.out.print(".");
      }
    }
    System.out.println();
    logisticRegression1 = new LogisticRegressionBinary(0);
    double[] weights13 = logisticRegression1.learnWeights(dataSet1, regularization);
    assertArrayEquals(weights11, weights13, ValuePrecision.POINT_8_ZEROS_ONE.getValuePrecision());

    LogisticRegressionBinary logisticRegression2 = new LogisticRegressionBinary(-1);
    double[] weights21 = logisticRegression2.learnWeights(dataSet1, regularization);
    double[] weights22 = logisticRegression2.learnWeights(dataSet2, regularization);
    assertArrayEquals(weights21, weights22, ValuePrecision.POINT_8_ZEROS_ONE.getValuePrecision());

  }
}
