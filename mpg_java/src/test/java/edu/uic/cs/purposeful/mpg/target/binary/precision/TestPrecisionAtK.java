package edu.uic.cs.purposeful.mpg.target.binary.precision;

import static org.hamcrest.CoreMatchers.hasItem;

import java.util.BitSet;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.commons.lang3.reflect.FieldUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;

import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;

import edu.uic.cs.purposeful.common.collection.CollectionUtils;
import edu.uic.cs.purposeful.mpg.common.ValuePrecision;
import edu.uic.cs.purposeful.mpg.target.binary.AbstractBinaryOptimizationTarget;

public class TestPrecisionAtK extends TestAbstractPrecisionAtK {

  private PrecisionAtK newInstance(int totalNumOfBits, int k) {
    PrecisionAtK precisionAtK = new PrecisionAtK();
    try {
      FieldUtils.writeField(FieldUtils.getDeclaredField(AbstractBinaryOptimizationTarget.class,
          "totalNumOfBits", true), precisionAtK, totalNumOfBits);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    }
    precisionAtK.k = k;
    precisionAtK.reciprocalK = 1.0 / k;
    return precisionAtK;
  }

  @Test
  public void testGetInitialMinimizerPermutations() {
    System.out.println("\ntestGetInitialMinimizerPermutations");
    PrecisionAtK precisionAtK = newInstance(21, 7);
    Set<BitSet> actual = precisionAtK.getInitialMinimizerPermutations();
    System.out.println(actual);

    BitSet expected_1 = new BitSet(21);
    expected_1.clear();
    System.out.println(expected_1);
    assertEquals(0, expected_1.cardinality());

    BitSet expected_2 = new BitSet(21);
    expected_2.set(0, 21);
    System.out.println(expected_2);
    assertEquals(21, expected_2.cardinality());

    assertEquals(Sets.newHashSet(actual), Sets.newHashSet(expected_1, expected_2));
  }

  @Test
  public void testFindBestMinimizerResponsePermutation1() {
    PrecisionAtK precisionAtK = newInstance(4, 3);
    double[] maximizerProbabilities = new double[] {0.5, 0.3, 0.2};
    LinkedHashSet<BitSet> maximizerPermutations = new LinkedHashSet<>();
    BitSet maximizerPermutation1 = new BitSet(4);
    maximizerPermutation1.set(0);
    maximizerPermutation1.set(2);
    maximizerPermutation1.set(3);
    maximizerPermutations.add(maximizerPermutation1);
    BitSet maximizerPermutation2 = new BitSet(4);
    maximizerPermutation2.set(0);
    maximizerPermutation2.set(1);
    maximizerPermutation2.set(2);
    maximizerPermutations.add(maximizerPermutation2);
    BitSet maximizerPermutation3 = new BitSet(4);
    maximizerPermutation3.set(1);
    maximizerPermutation3.set(2);
    maximizerPermutation3.set(3);
    maximizerPermutations.add(maximizerPermutation3);
    double[] lagrangePotentials = new double[] {-6, 5, 2, -3};

    Pair<BitSet, Double> actual = precisionAtK.findBestMinimizerResponsePermutation(
        maximizerProbabilities, maximizerPermutations, lagrangePotentials);
    System.out.println(actual.getRight() + "\t" + actual.getLeft());

    assertEquals(-6.5, actual.getRight(), ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
    assertEquals(2, actual.getLeft().cardinality());
    // 0 1 1 0
    assertTrue(actual.getLeft().get(1));
    assertTrue(actual.getLeft().get(2));
  }

  @Test
  public void testFindBestMinimizerResponsePermutation2() {
    for (int totalNumOfBits = 3; totalNumOfBits <= 15; totalNumOfBits++) {
      for (int testTime = 1; testTime <= 10; testTime++) {
        testFindBestMinimizerResponsePermutation(totalNumOfBits, testTime);
      }
    }
  }

  private void testFindBestMinimizerResponsePermutation(int totalNumOfBits, int testTime) {
    Random random = new Random();
    int k = (int) Math.round(random.nextDouble() * (totalNumOfBits - 1) + 1);

    System.out.println("\testFindBestMinimizerResponsePermutation, totalNumOfBits=" + totalNumOfBits
        + ", testTime=" + testTime + ", k=" + k);

    double[] lagrangePotentials = new double[totalNumOfBits];
    for (int index = 0; index < totalNumOfBits; index++) {
      lagrangePotentials[index] = -1 + 2 * random.nextDouble(); // range [-1, 1)
    }

    LinkedHashSet<BitSet> maximizerPermutations =
        generateAllPossiblePermutations(totalNumOfBits, k);
    double[] maximizerProbabilities = generateProbabilities(random, maximizerPermutations.size());

    Entry<Double, HashSet<BitSet>> expected = bruteForceFindBestMinimizerResponsePermutation(
        maximizerProbabilities, maximizerPermutations, lagrangePotentials, totalNumOfBits, k);
    System.out.println(expected.getKey() + "\t" + expected.getValue());

    PrecisionAtK precisionAtK = newInstance(totalNumOfBits, k);
    Pair<BitSet, Double> actual = precisionAtK.findBestMinimizerResponsePermutation(
        maximizerProbabilities, maximizerPermutations, lagrangePotentials);
    System.out.println(actual.getRight() + "\t" + actual.getLeft());

    assertEquals(expected.getKey(), actual.getRight(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
    assertThat(expected.getValue(), hasItem(actual.getLeft()));
  }

  private Entry<Double, HashSet<BitSet>> bruteForceFindBestMinimizerResponsePermutation(
      double[] maximizerProbabilities, LinkedHashSet<BitSet> maximizerPermutations,
      double[] lagrangePotentials, int totalNumOfBits, int k) {
    LinkedHashSet<BitSet> minimizerPermutations = generateAllPossiblePermutations(totalNumOfBits);
    assertEquals((int) Math.pow(2, totalNumOfBits), minimizerPermutations.size());

    TreeMap<Double, HashSet<BitSet>> minimizerPermutationsByScore = new TreeMap<>(); // smallest
                                                                                     // first
    for (BitSet minimizerPermutation : minimizerPermutations) {
      double lagrangePotentialsSum = 0.0;
      for (int index = minimizerPermutation.nextSetBit(0); index >= 0
          && index < totalNumOfBits; index = minimizerPermutation.nextSetBit(index + 1)) {
        lagrangePotentialsSum += lagrangePotentials[index];
      }

      double minimizerScore = 0.0;
      int maximizerIndex = 0;
      for (BitSet maximizerPermutation : maximizerPermutations) {
        assertEquals(k, maximizerPermutation.cardinality());

        double maximizerProbability = maximizerProbabilities[maximizerIndex++];

        double precision = 0.0;
        for (int bitIndex = maximizerPermutation.nextSetBit(0); bitIndex >= 0
            && bitIndex < totalNumOfBits; bitIndex =
                maximizerPermutation.nextSetBit(bitIndex + 1)) {
          if (minimizerPermutation.get(bitIndex)) {
            precision += 1;
          }
        }

        precision /= k;
        minimizerScore += maximizerProbability * (precision - lagrangePotentialsSum);
      }

      CollectionUtils.putInHashSetValueMap(minimizerScore, minimizerPermutation,
          minimizerPermutationsByScore);
    }

    return Iterables.getFirst(minimizerPermutationsByScore.entrySet(), null);
  }

  @Test
  public void testIsLegalMinimizerPermutation() {
    int totalNumOfBits = 10;

    for (int testTime = 0; testTime < 10; testTime++) {
      Random random = new Random();
      int k = random.nextInt(totalNumOfBits);
      PrecisionAtK precisionAtK = newInstance(totalNumOfBits, k);
      System.out.println("testIsLegalMinimizerPermutation, testTime=" + testTime + ", k=" + k);

      Set<Integer> generated = new TreeSet<>();
      while (generated.size() < k) {
        Integer next = random.nextInt(totalNumOfBits);
        generated.add(next);
      }

      BitSet action = new BitSet(totalNumOfBits);
      for (int bitIndex : generated) {
        action.set(bitIndex);
      }

      assertTrue(precisionAtK.isLegalMinimizerPermutation(action));
    }
  }
}
