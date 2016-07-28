package edu.uic.cs.purposeful.mpg.target.binary.precision;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Arrays;
import java.util.BitSet;

import org.junit.Test;

import com.google.common.primitives.Ints;

import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.learning.MaximizerPredictor.Prediction;
import edu.uic.cs.purposeful.mpg.learning.binary.MPGBinaryClassifier;

public class TestPrecisionAtKClassifier {

  @Test
  public void testLearnAndPredictOnSameData() throws Exception {
    MPGBinaryClassifier classifier =
        new MPGBinaryClassifier(PrecisionAtK.class, PrecisionAtK.BINARY_VALUE_ONE);
    File file = new File(
        TestPrecisionAtKClassifier.class.getResource("TestPrecisionAtKClassifier.train").toURI());
    double[] thetas = classifier.learn(file, Regularization.l2(0));

    Prediction<BitSet> prediction = classifier.predict(file);
    System.out.println("Thetas:\t" + Arrays.toString(thetas));
    System.out.println("Golden:\t" + prediction.getGoldenPermutation());
    System.out.println("Prediction:\t" + prediction.getPredictionPermutation());
    System.out.println("P@k:\t" + prediction.getScore());
    System.out.println("Probability:\t" + prediction.getProbability());
    assertTrue(Ints.asList(prediction.getGoldenPermutation().stream().toArray())
        .containsAll(Ints.asList(prediction.getPredictionPermutation().stream().toArray())));
  }
}
