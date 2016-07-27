package edu.uic.cs.purposeful.mpg.target.binary.precision;

import static org.hamcrest.CoreMatchers.hasItem;

import java.util.BitSet;
import java.util.Comparator;
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
import org.paukov.combinatorics.Factory;
import org.paukov.combinatorics.Generator;
import org.paukov.combinatorics.ICombinatoricsVector;

import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;

import edu.uic.cs.purposeful.common.collection.CollectionUtils;
import edu.uic.cs.purposeful.mpg.common.ValuePrecision;
import edu.uic.cs.purposeful.mpg.target.binary.AbstractBinaryOptimizationTarget;
import edu.uic.cs.purposeful.mpg.target.binary.TestAbstractBinaryOptimizationTarget;

public class TestAbstractPrecisionAtK extends TestAbstractBinaryOptimizationTarget {

  private AbstractPrecisionAtK newInstance(int totalNumOfBits, int k) {
    AbstractPrecisionAtK precisionAtK = new AbstractPrecisionAtK() {
      @Override
      public boolean isLegalMinimizerPermutation(BitSet goldPermutation) {
        throw new UnsupportedOperationException();
      }

      @Override
      public Set<BitSet> getInitialMinimizerPermutations() {
        throw new UnsupportedOperationException();
      }

      @Override
      public Pair<BitSet, Double> findBestMinimizerResponsePermutation(
          double[] maximizerProbabilities, LinkedHashSet<BitSet> existingMaximizerPermutations,
          double[] lagrangePotentials) {
        throw new UnsupportedOperationException();
      }
    };

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
  public void testComputeScore() {
    BitSet maximizerPermutation = new BitSet(10);
    maximizerPermutation.set(1);
    maximizerPermutation.set(2);
    maximizerPermutation.set(5);
    maximizerPermutation.set(9);
    BitSet minimizerPermutation = new BitSet(10);
    minimizerPermutation.set(0);
    minimizerPermutation.set(3);
    minimizerPermutation.set(5);

    double expected = 1.0 / 4.0;
    AbstractPrecisionAtK precisionAtK = newInstance(10, 4);
    double actual = precisionAtK.computeScore(maximizerPermutation, minimizerPermutation);
    assertEquals(expected, actual, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testFindBestMaximizerResponsePermutation() {
    for (int totalNumOfBits = 3; totalNumOfBits <= 15; totalNumOfBits++) {
      for (int testTime = 1; testTime <= 10; testTime++) {
        testFindBestMaximizerResponsePermutation(totalNumOfBits, testTime);
      }
    }
  }

  private void testFindBestMaximizerResponsePermutation(int totalNumOfBits, int testTime) {
    Random random = new Random();
    int k = (int) Math.round(random.nextDouble() * (totalNumOfBits - 1) + 1);

    System.out.println("\ntestFindBestMaximizerResponsePermutation, totalNumOfBits="
        + totalNumOfBits + ", testTime=" + testTime + ", k=" + k);

    double[] lagrangePotentials = new double[totalNumOfBits];
    for (int index = 0; index < totalNumOfBits; index++) {
      lagrangePotentials[index] = -1 + 2 * random.nextDouble(); // range [-1, 1)
    }

    LinkedHashSet<BitSet> minimizerPermutations = generateAllPossiblePermutations(totalNumOfBits);
    assertEquals((int) Math.pow(2, totalNumOfBits), minimizerPermutations.size());
    double[] minimizerProbabilities = generateProbabilities(random, minimizerPermutations.size());

    Entry<Double, HashSet<BitSet>> expected = bruteForceFindBestMaximizerResponsePermutation(
        minimizerProbabilities, minimizerPermutations, lagrangePotentials, totalNumOfBits, k);
    System.out.println(expected.getKey() + "\t" + expected.getValue());

    AbstractPrecisionAtK precisionAtK = newInstance(totalNumOfBits, k);
    Pair<BitSet, Double> actual = precisionAtK.findBestMaximizerResponsePermutation(
        minimizerProbabilities, minimizerPermutations, lagrangePotentials);
    System.out.println(actual.getRight() + "\t" + actual.getLeft());

    assertEquals(expected.getKey(), actual.getRight(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
    assertThat(expected.getValue(), hasItem(actual.getLeft()));
  }

  protected double[] generateProbabilities(Random random, int numOfPermutations) {
    double[] minimizerProbabilities = new double[numOfPermutations];
    double minimizerProbabilitiySum = 0.0;
    for (int index = 0; index < numOfPermutations; index++) {
      minimizerProbabilities[index] = random.nextDouble();
      minimizerProbabilitiySum += minimizerProbabilities[index];
    }
    for (int index = 0; index < numOfPermutations; index++) {
      minimizerProbabilities[index] = minimizerProbabilities[index] / minimizerProbabilitiySum;
    }
    return minimizerProbabilities;
  }

  private Entry<Double, HashSet<BitSet>> bruteForceFindBestMaximizerResponsePermutation(
      double[] minimizerProbabilities, LinkedHashSet<BitSet> minimizerPermutations,
      double[] lagrangePotentials, int totalNumOfBits, int k) {
    LinkedHashSet<BitSet> maximizerPermutations =
        generateAllPossiblePermutations(totalNumOfBits, k);

    TreeMap<Double, HashSet<BitSet>> maximizerPermutationsByScore =
        new TreeMap<>(Comparator.reverseOrder()); // largest first
    for (BitSet maximizerPermutation : maximizerPermutations) {
      assertEquals(k, maximizerPermutation.cardinality());

      double maximizerScore = 0.0;
      int minimizerIndex = 0;
      for (BitSet minimizerPermutation : minimizerPermutations) {
        double minimizerProbability = minimizerProbabilities[minimizerIndex++];

        double lagrangePotentialsSum = 0.0;
        double precision = 0.0;
        for (int index = minimizerPermutation.nextSetBit(0); index >= 0
            && index < totalNumOfBits; index = minimizerPermutation.nextSetBit(index + 1)) {
          lagrangePotentialsSum += lagrangePotentials[index];

          if (maximizerPermutation.get(index)) {
            precision += 1;
          }
        }

        precision /= k;
        maximizerScore += minimizerProbability * (precision - lagrangePotentialsSum);
      }

      CollectionUtils.putInHashSetValueMap(maximizerScore, maximizerPermutation,
          maximizerPermutationsByScore);
    }

    return Iterables.getFirst(maximizerPermutationsByScore.entrySet(), null);
  }

  protected LinkedHashSet<BitSet> generateAllPossiblePermutations(int totalNumOfBits) {
    return generateAllPossiblePermutations(totalNumOfBits, null);
  }

  protected LinkedHashSet<BitSet> generateAllPossiblePermutations(int totalNumOfBits, Integer k) {
    Generator<Boolean> permutationGenerator = Factory.createPermutationWithRepetitionGenerator(
        Factory.createVector(new Boolean[] {true, false}), totalNumOfBits);

    LinkedHashSet<BitSet> result =
        new LinkedHashSet<>((int) permutationGenerator.getNumberOfGeneratedObjects());
    for (ICombinatoricsVector<Boolean> perm : permutationGenerator) {
      BitSet permutation = new BitSet(totalNumOfBits);
      for (int index = 0; index < totalNumOfBits; index++) {
        if (perm.getValue(index)) {
          permutation.set(index);
        }
      }

      if (k == null || permutation.cardinality() == k) {
        result.add(permutation);
      }
    }
    return result;
  }

  @Test
  public void testGetInitialMaximizerPermutations() {
    System.out.println("\ntestGetInitialMaximizerPermutations");
    AbstractPrecisionAtK precisionAtK = newInstance(13, 7);
    Set<BitSet> actual = precisionAtK.getInitialMaximizerPermutations();
    System.out.println(actual);

    BitSet expected_1 = new BitSet(13);
    expected_1.set(0, 7);
    System.out.println(expected_1);
    assertEquals(7, expected_1.cardinality());

    BitSet expected_2 = new BitSet(13);
    expected_2.set(6, 13);
    System.out.println(expected_2);
    assertEquals(7, expected_2.cardinality());

    assertEquals(Sets.newHashSet(actual), Sets.newHashSet(expected_1, expected_2));
  }

  @Test
  public void testIsLegalMaximizerPermutation() {
    int totalNumOfBits = 13;

    for (int testTime = 0; testTime < 10; testTime++) {
      Random random = new Random();
      int k = random.nextInt(totalNumOfBits); // 0 to (totalNumOfBits-1)
      AbstractPrecisionAtK precisionAtK = newInstance(totalNumOfBits, k);
      System.out.println("testIsLegalMaximizerPermutation, testTime=" + testTime + ", k=" + k);

      Set<Integer> generated = new TreeSet<>();
      while (generated.size() < k) {
        Integer next = random.nextInt(totalNumOfBits);
        generated.add(next);
      }
      BitSet actionWithKOnes = new BitSet(totalNumOfBits);
      for (int bitIndex : generated) {
        actionWithKOnes.set(bitIndex);
      }
      assertTrue(precisionAtK.isLegalMaximizerPermutation(actionWithKOnes));

      int extraNumOfOneBits = random.nextInt(totalNumOfBits - k) + 1;
      while (generated.size() < k + extraNumOfOneBits) {
        Integer next = random.nextInt(totalNumOfBits);
        generated.add(next);
      }
      BitSet actionWithMoreThanKOnes = new BitSet(totalNumOfBits);
      for (int bitIndex : generated) {
        actionWithMoreThanKOnes.set(bitIndex);
      }
      assertFalse(precisionAtK.isLegalMaximizerPermutation(actionWithMoreThanKOnes));
    }
  }
}
