package edu.uic.cs.purposeful.mpg.optimizer.numerical.objective.impl;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Assert;
import org.junit.Test;

import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.common.ValuePrecision;
import edu.uic.cs.purposeful.mpg.target.OptimizationTarget;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;

public class TestMinimizationObjectiveFunctionImpl extends Assert {

  static class MockTarget implements OptimizationTarget<BitSet, Integer> {

    private int index;

    @Override
    public void initialize(Integer index, boolean duringTraining) {
      this.index = index;
    }

    @Override
    public double computeScore(BitSet maximizerPermutation, BitSet minimizerPermutation) {
      return 12.3 * (index + 1);
    }

    @Override
    public Set<BitSet> getInitialMaximizerPermutations() {
      if (index % 2 == 0) {
        return Collections.singleton(BitSet.valueOf(new long[] {Long.parseLong("0110", 2)}));
      } else {
        return Collections.singleton(BitSet.valueOf(new long[] {Long.parseLong("1111", 2)}));
      }
    }

    @Override
    public Set<BitSet> getInitialMinimizerPermutations() {
      if (index % 2 == 0) {
        return Collections.singleton(BitSet.valueOf(new long[] {Long.parseLong("1111", 2)}));
      } else {
        return Collections.singleton(BitSet.valueOf(new long[] {Long.parseLong("0110", 2)}));
      }
    }

    @Override
    public double[] computeLagrangePotentials(double[] thetas) {
      if (index % 2 == 0) {
        return new double[] {4, 3, 2, 1};
      } else {
        return new double[] {-4, -3, -2, -1};
      }
    }

    @Override
    public double aggregateLagrangePotentials(BitSet minimizerPermutation,
        double[] lagrangePotentials) {
      return 0.0;
    }

    @Override
    public Pair<BitSet, Double> findBestMaximizerResponsePermutation(
        double[] minimizerProbabilities, LinkedHashSet<BitSet> existingMinimizerPermutations,
        double[] lagrangePotentials) {
      if (index % 2 == 0) {
        return Pair.of(BitSet.valueOf(new long[] {Long.parseLong("0110", 2)}), Double.NaN);
      } else {
        return Pair.of(BitSet.valueOf(new long[] {Long.parseLong("1111", 2)}), Double.NaN);
      }
    }

    @Override
    public Pair<BitSet, Double> findBestMinimizerResponsePermutation(
        double[] maximizerProbabilities, LinkedHashSet<BitSet> existingMaximizerPermutations,
        double[] lagrangePotentials) {
      if (index % 2 == 0) {
        return Pair.of(BitSet.valueOf(new long[] {Long.parseLong("1111", 2)}), Double.NaN);
      } else {
        return Pair.of(BitSet.valueOf(new long[] {Long.parseLong("0110", 2)}), Double.NaN);
      }
    }

    @Override
    public boolean isLegalMaximizerPermutation(BitSet permutation) {
      return false;
    }

    @Override
    public boolean isLegalMinimizerPermutation(BitSet permutation) {
      return false;
    }

    @Override
    public BitSet getGoldenPermutation() {
      return BitSet.valueOf(new long[] {Long.parseLong("1001", 2)});
    }

    @Override
    public Vector getGoldenFeatureValues() {
      if (index % 2 == 0) {
        return new DenseVector(new double[] {9, 8, 7}, false);
      } else {
        return new DenseVector(new double[] {-8, 7, -6}, false);
      }
    }

    @Override
    public Vector computeExpectedFeatureValues(double[] probabilities,
        LinkedHashSet<BitSet> permutations) {
      if (index % 2 == 0) {
        return new DenseVector(new double[] {5, 6, 7}, false);
      } else {
        return new DenseVector(new double[] {3, -5, 7}, false);
      }
    }
  }

  @Test
  public void testGetValueAndGradients() {
    List<Integer> initialDataList = new ArrayList<>();
    initialDataList.add(0);
    initialDataList.add(1);
    MinimizationObjectiveFunctionImpl<BitSet, Integer> objectiveFunction =
        new MinimizationObjectiveFunctionImpl<>(MockTarget.class, initialDataList);
    objectiveFunction.setRegularization(Regularization.l2(9));

    double[] thetas = new double[] {4.1, 3.2, 2.3};
    Pair<Double, double[]> valueAndGradients = objectiveFunction.getValueAndGradients(thetas);

    double actualValue = valueAndGradients.getLeft();
    // -goldenFeatureValues <dot> thetas - maximizerValue + L2 * ||thetas||^2 / 2
    // = {[-(9, 8, 7)<dot>(4.1, 3.2, 2.3) - 12.3*1] + [(-8, 7, -6)<dot>(4.1, 3.2, 2.3) - 12.3*2]}
    // + 9*(4.1^2 + 3.2^2 + 2.3^2) / 2
    double expectedValue =
        (-9 * 4.1 - 8 * 3.2 - 7 * 2.3 - 12.3 + 8 * 4.1 - 7 * 3.2 + 6 * 2.3 - 12.3 * 2)
            + 9 * (4.1 * 4.1 + 3.2 * 3.2 + 2.3 * 2.3) / 2;
    assertEquals(expectedValue, actualValue, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    // -goldenFeatureValues + expectedFeatureValues + L2 * thetas
    // ={[-(9, 8, 7)+(5, 6, 7)] + [-(-8, 7, -6)+(3, -5, 7)]} + 9*(4.1 + 3.2 + 2.3)
    double[] actualGradients = valueAndGradients.getRight();
    double[] expectedGradients = new double[] {(-9 + 5 + 8 + 3) + 9 * 4.1,
        (-8 + 6 - 7 - 5) + 9 * 3.2, (-7 + 7 + 6 + 7) + 9 * 2.3};
    assertArrayEquals(expectedGradients, actualGradients,
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testGetValue() {
    List<Integer> initialDataList = new ArrayList<>();
    initialDataList.add(0);
    initialDataList.add(1);
    MinimizationObjectiveFunctionImpl<BitSet, Integer> objectiveFunction =
        new MinimizationObjectiveFunctionImpl<>(MockTarget.class, initialDataList);
    objectiveFunction.setRegularization(Regularization.l2(6));

    double[] thetas = new double[] {1.4, 2.3, 3};
    double actualValue = objectiveFunction.getValue(thetas);
    // -goldenFeatureValues <dot> thetas - maximizerValue + L2 * ||thetas||^2
    double expectedValue =
        (-9 * 1.4 - 8 * 2.3 - 7 * 3 - 12.3 + 8 * 1.4 - 7 * 2.3 + 6 * 3 - 12.3 * 2)
            + 6 * (1.4 * 1.4 + 2.3 * 2.3 + 3. * 3) / 2;
    assertEquals(expectedValue, actualValue, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testGetGradients() {
    List<Integer> initialDataList = new ArrayList<>();
    initialDataList.add(0);
    initialDataList.add(1);
    MinimizationObjectiveFunctionImpl<BitSet, Integer> objectiveFunction =
        new MinimizationObjectiveFunctionImpl<>(MockTarget.class, initialDataList);
    objectiveFunction.setRegularization(Regularization.l2(1));

    double[] thetas = new double[] {1, 2, 3};
    // -goldenFeatureValues + expectedFeatureValues + L2 * thetas
    // ={[-(9, 8, 7)+(5, 6, 7)] + [-(-8, 7, -6)+(3, -5, 7)]} + 1*(1 + 2 + 3)
    double[] actualGradients = objectiveFunction.getGradients(thetas);
    double[] expectedGradients =
        new double[] {(-9 + 5 + 8 + 3) + 1 * 1, (-8 + 6 - 7 - 5) + 1 * 2, (-7 + 7 + 6 + 7) + 1 * 3};
    assertArrayEquals(expectedGradients, actualGradients,
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }
}
