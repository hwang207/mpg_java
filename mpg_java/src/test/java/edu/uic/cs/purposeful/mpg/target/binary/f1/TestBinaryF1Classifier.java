package edu.uic.cs.purposeful.mpg.target.binary.f1;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.util.Arrays;
import java.util.BitSet;

import org.junit.Test;

import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.learning.MaximizerPredictor.Prediction;
import edu.uic.cs.purposeful.mpg.learning.binary.MPGBinaryClassifier;

public class TestBinaryF1Classifier {

  @Test
  public void testLearnAndPredictOnSameData() throws Exception {
    MPGBinaryClassifier classifier =
        new MPGBinaryClassifier(BinaryF1.class, BinaryF1.BINARY_VALUE_ONE);
    File file =
        new File(TestBinaryF1Classifier.class.getResource("TestBinaryF1Classifier.train").toURI());
    double[] thetas = classifier.learn(file, Regularization.l2(0));
    Prediction<BitSet> prediction = classifier.predict(file, thetas);
    // prediction = classifier.predict(file); // use just learned thetas by default

    // You can write the learned model to a local file
    // File modelFile = new File("learned.model");
    // classifier.writeModel(modelFile);
    // classifier.loadModel(modelFile); // then read it back
    // prediction = classifier.predict(file); // and predict

    System.out.println("Thetas:\t" + Arrays.toString(thetas));
    System.out.println("Golden:\t" + prediction.getGoldenPermutation());
    System.out.println("Prediction:\t" + prediction.getPredictionPermutation());
    System.out.println("F1:\t" + prediction.getScore());
    System.out.println("Probability:\t" + prediction.getProbability());
    assertEquals(prediction.getGoldenPermutation(), prediction.getPredictionPermutation());
  }
}
