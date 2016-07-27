package edu.uic.cs.purposeful.mpg.target.binary;

import java.util.Arrays;
import java.util.BitSet;
import java.util.LinkedHashSet;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import edu.uic.cs.purposeful.mpg.common.ValuePrecision;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public class TestAbstractBinaryOptimizationTarget extends Assert {

  private AbstractBinaryOptimizationTarget binaryOptimizationTarget = null;

  @Before
  public void initialize() {
    binaryOptimizationTarget = new AbstractBinaryOptimizationTarget() {

      @Override
      public boolean isLegalMinimizerPermutation(BitSet permutation) {
        throw new UnsupportedOperationException();
      }

      @Override
      public boolean isLegalMaximizerPermutation(BitSet permutation) {
        throw new UnsupportedOperationException();
      }

      @Override
      public Set<BitSet> getInitialMinimizerPermutations() {
        throw new UnsupportedOperationException();
      }

      @Override
      public Set<BitSet> getInitialMaximizerPermutations() {
        throw new UnsupportedOperationException();
      }

      @Override
      public Pair<BitSet, Double> findBestMinimizerResponsePermutation(
          double[] maximizerProbabilities, LinkedHashSet<BitSet> existingMaximizerPermutations,
          double[] lagrangePotentials) {
        throw new UnsupportedOperationException();
      }

      @Override
      public Pair<BitSet, Double> findBestMaximizerResponsePermutation(
          double[] minimizerProbabilities, LinkedHashSet<BitSet> existingMinimizerPermutations,
          double[] lagrangePotentials) {
        throw new UnsupportedOperationException();
      }

      @Override
      public double computeScore(BitSet maximizerPermutation, BitSet minimizerPermutation) {
        throw new UnsupportedOperationException();
      }
    };

    double[] goldenTagValues = new double[] {1, 0, 0, 1};
    // 1, 2, 3
    // 1, 3, 5
    // 1, 4, 7
    // 1, 5, 9
    LinkedSparseMatrix featureMatrix = new LinkedSparseMatrix(4, 3);
    for (int rowIndex = 1; rowIndex <= featureMatrix.numRows(); rowIndex++) {
      for (int columnIndex = 1; columnIndex <= featureMatrix.numColumns(); columnIndex++) {
        featureMatrix.set(rowIndex - 1, columnIndex - 1, rowIndex * columnIndex - rowIndex + 1);
      }
    }
    Pair<double[], LinkedSparseMatrix> trainingData = Pair.of(goldenTagValues, featureMatrix);
    binaryOptimizationTarget.initialize(trainingData, true);
  }

  @Test
  public void testAggregateLagrangePotentials1() {
    System.out.println("\ntestAggregateLagrangePotentials1");
    double[] lagrangePotentials = new double[10];
    for (int index = 0; index < lagrangePotentials.length; index++) {
      lagrangePotentials[index] = index;
    }
    System.out.println(Arrays.toString(lagrangePotentials));

    BitSet minimizerPermutation = new BitSet(lagrangePotentials.length);
    minimizerPermutation.set(1); // 1
    minimizerPermutation.set(3); // 3
    minimizerPermutation.set(5); // 5
    minimizerPermutation.set(9); // 9
    System.out.println(minimizerPermutation);
    assertEquals(4, minimizerPermutation.cardinality());
    assertEquals(10, minimizerPermutation.length());

    double expected = 1 + 3 + 5 + 9;

    double actual = binaryOptimizationTarget.aggregateLagrangePotentials(minimizerPermutation,
        lagrangePotentials);
    assertEquals(expected, actual, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testAggregateLagrangePotentials2() {
    System.out.println("\ntestAggregateLagrangePotentials2");
    double[] lagrangePotentials = new double[13];
    for (int index = 0; index < lagrangePotentials.length; index++) {
      lagrangePotentials[index] = -index - 1;
    }
    System.out.println(Arrays.toString(lagrangePotentials));

    BitSet minimizerPermutation = new BitSet(lagrangePotentials.length);
    System.out.println(minimizerPermutation);
    assertEquals(0, minimizerPermutation.cardinality());
    assertEquals(0, minimizerPermutation.length());

    double expected = 0;

    double actual = binaryOptimizationTarget.aggregateLagrangePotentials(minimizerPermutation,
        lagrangePotentials);
    assertEquals(expected, actual, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testAggregateLagrangePotentials3() {
    System.out.println("\ntestAggregateLagrangePotentials3");
    double[] lagrangePotentials = new double[8];
    for (int index = 0; index < lagrangePotentials.length; index++) {
      lagrangePotentials[index] = -index - 1;
    }
    System.out.println(Arrays.toString(lagrangePotentials));

    BitSet minimizerPermutation = new BitSet(lagrangePotentials.length);
    for (int index = 0; index < lagrangePotentials.length; index++) {
      minimizerPermutation.set(index);
    }
    System.out.println(minimizerPermutation);
    assertEquals(8, minimizerPermutation.cardinality());
    assertEquals(8, minimizerPermutation.length());

    double expected = -(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);

    double actual = binaryOptimizationTarget.aggregateLagrangePotentials(minimizerPermutation,
        lagrangePotentials);
    assertEquals(expected, actual, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testComputeLagrangePotentials() {
    // 1, 2, 3
    // 1, 3, 5
    // 1, 4, 7
    // 1, 5, 9
    double[] thetasArray = new double[] {1.2, 2.3, 3.4};
    double[] actual = binaryOptimizationTarget.computeLagrangePotentials(thetasArray);
    double[] expected = new double[] {1.2 * 1 + 2.3 * 2 + 3.4 * 3, 1.2 * 1 + 2.3 * 3 + 3.4 * 5,
        1.2 * 1 + 2.3 * 4 + 3.4 * 7, 1.2 * 1 + 2.3 * 5 + 3.4 * 9};
    assertArrayEquals(expected, actual, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testGetGoldenPermutation() {
    BitSet actual = binaryOptimizationTarget.getGoldenPermutation();
    BitSet expected = BitSet.valueOf(new long[] {Long.parseLong("1001", 2)});
    assertEquals(expected, actual);
  }

  @Test
  public void testGetGoldenFeatureValues() {
    // 1 | 1, 2, 3
    // 0 | 1, 3, 5
    // 0 | 1, 4, 7
    // 1 | 1, 5, 9
    double[] expecteds =
        new DenseVector(binaryOptimizationTarget.getGoldenFeatureValues()).getData();
    double[] actuals = new double[] {2, 7, 12};
    assertArrayEquals(expecteds, actuals, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testComputeExpectedFeatureValues() {
    LinkedHashSet<BitSet> permutations = new LinkedHashSet<>();
    // note the order of Long-String is reverse of the order of BitSet!
    permutations.add(BitSet.valueOf(new long[] {Long.parseLong("1011", 2)}));
    permutations.add(BitSet.valueOf(new long[] {Long.parseLong("0110", 2)}));
    permutations.add(BitSet.valueOf(new long[] {Long.parseLong("1111", 2)}));
    double[] probabilities = new double[] {0.3, 0.2, 0.5};
    // 1 | 1, 2, 3 * p1=0.3
    // 1 | 1, 3, 5
    // 0 | 1, 4, 7
    // 1 | 1, 5, 9 = [3, 10, 17]*0.3 = [0.9, 3, 5.1]
    // -----------
    // 0 | 1, 2, 3 * p2=0.2
    // 1 | 1, 3, 5
    // 1 | 1, 4, 7
    // 0 | 1, 5, 9 = [2, 7, 12]*0.2 = [0.4, 1.4, 2.4]
    // -----------
    // 1 | 1, 2, 3 * p2=0.5
    // 1 | 1, 3, 5
    // 1 | 1, 4, 7
    // 1 | 1, 5, 9 = [4, 14, 24]*0.5 = [2, 7, 12]
    // -----------
    double[] actuals = new DenseVector(
        binaryOptimizationTarget.computeExpectedFeatureValues(probabilities, permutations))
            .getData();
    System.out.println(Arrays.toString(actuals));
    double[] expecteds = new double[] {0.9 + 0.4 + 2, 3 + 1.4 + 7, 5.1 + 2.4 + 12};
    System.out.println(Arrays.toString(expecteds));
    assertArrayEquals(expecteds, actuals, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

}
