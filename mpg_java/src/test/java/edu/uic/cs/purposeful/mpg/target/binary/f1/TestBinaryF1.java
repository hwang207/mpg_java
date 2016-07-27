package edu.uic.cs.purposeful.mpg.target.binary.f1;

import static org.hamcrest.CoreMatchers.hasItem;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;
import org.paukov.combinatorics.Factory;
import org.paukov.combinatorics.Generator;
import org.paukov.combinatorics.ICombinatoricsVector;

import com.google.common.collect.Iterables;

import edu.uic.cs.purposeful.common.collection.CollectionUtils;
import edu.uic.cs.purposeful.mpg.common.ValuePrecision;
import edu.uic.cs.purposeful.mpg.target.binary.TestAbstractBinaryOptimizationTarget;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public class TestBinaryF1 extends TestAbstractBinaryOptimizationTarget {

  private BinaryF1 buildBinaryF1(int totalNumOfBits) {
    BinaryF1 binaryF1 = new BinaryF1();
    double[] tags = new double[totalNumOfBits];
    LinkedSparseMatrix featureMatrix = new LinkedSparseMatrix(totalNumOfBits, 0);
    Pair<double[], LinkedSparseMatrix> trainingData = Pair.of(tags, featureMatrix);
    binaryF1.initialize(trainingData, true);
    return binaryF1;
  }

  @Test
  public void testComputeScore1() {
    BitSet maximizerPermutation = new BitSet();
    BitSet minimizerPermutation = new BitSet();
    double actual = buildBinaryF1(10).computeScore(maximizerPermutation, minimizerPermutation);
    assertEquals(1.0, actual, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testComputeScore2() {
    BitSet maximizerPermutation = new BitSet();
    maximizerPermutation.set(1);
    maximizerPermutation.set(3);
    maximizerPermutation.set(5);
    maximizerPermutation.set(7);

    BitSet minimizerPermutation = new BitSet();
    minimizerPermutation.set(2);
    minimizerPermutation.set(4);
    minimizerPermutation.set(6);
    minimizerPermutation.set(8);

    double actual = buildBinaryF1(10).computeScore(maximizerPermutation, minimizerPermutation);
    assertEquals(0, actual, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testComputeScore3() {
    BitSet maximizerPermutation = new BitSet();
    maximizerPermutation.set(1);
    maximizerPermutation.set(3);
    maximizerPermutation.set(5);
    maximizerPermutation.set(7);

    BitSet minimizerPermutation = new BitSet();
    minimizerPermutation.set(2);
    minimizerPermutation.set(3);
    minimizerPermutation.set(5);
    minimizerPermutation.set(8);
    minimizerPermutation.set(9);

    double actual = buildBinaryF1(10).computeScore(maximizerPermutation, minimizerPermutation);
    assertEquals(2.0 * 2 / (4 + 5), actual, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testGetInitialMaximizerPermutations() {
    int totalNumOfBits = 10;
    Set<BitSet> actual = buildBinaryF1(totalNumOfBits).getInitialMaximizerPermutations();

    Set<BitSet> expected = new HashSet<>();
    BitSet allZeros = new BitSet(totalNumOfBits);
    BitSet allOnes = new BitSet(totalNumOfBits);
    allOnes.set(0, totalNumOfBits);
    expected.add(allOnes);
    expected.add(allZeros);

    assertEquals(expected, actual);
  }

  @Test
  public void testGetInitialMinimizerPermutations() {
    int totalNumOfBits = 10;
    Set<BitSet> actual = buildBinaryF1(totalNumOfBits).getInitialMinimizerPermutations();

    Set<BitSet> expected = new HashSet<>();
    BitSet allZeros = new BitSet(totalNumOfBits);
    BitSet allOnes = new BitSet(totalNumOfBits);
    allOnes.set(0, totalNumOfBits);
    expected.add(allOnes);
    expected.add(allZeros);

    assertEquals(expected, actual);
  }

  @Test
  public void testComputeMarginalProbabilityMatrixP() {
    int totalNumOfBits = 10;
    BinaryF1 binaryF1 = buildBinaryF1(totalNumOfBits);

    double[] probabilities = new double[] {0.1, 0.2, 0.3, 0.4};
    LinkedHashSet<BitSet> permutations = new LinkedHashSet<>();
    BitSet permutation0 = new BitSet(totalNumOfBits); // 1100100000
    permutation0.set(0); // 1,3 * 0.1
    permutation0.set(1); // 2,3
    permutation0.set(4); // 5,3
    permutations.add(permutation0);

    BitSet permutation1 = new BitSet(totalNumOfBits); // 0000000000
    permutations.add(permutation1);

    BitSet permutation2 = new BitSet(totalNumOfBits); // 1010101010
    permutation2.set(0); // 1,5 * 0.3
    permutation2.set(2); // 3,5
    permutation2.set(4); // 5,5
    permutation2.set(6); // 7,5
    permutation2.set(8); // 9,5
    permutations.add(permutation2);

    BitSet permutation3 = new BitSet(totalNumOfBits); // 0110000100
    permutation3.set(1); // 2,3 * 0.4
    permutation3.set(2); // 3,3
    permutation3.set(7); // 8,3
    permutations.add(permutation3);

    Pair<Double, LinkedSparseMatrix> p0AndMatrixP =
        binaryF1.computeMarginalProbabilityMatrixP(probabilities, permutations);
    double p0 = p0AndMatrixP.getLeft();
    DenseMatrix matrixP = new DenseMatrix(p0AndMatrixP.getRight());
    System.out.println(matrixP);

    assertEquals(0.2, p0, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    // ___ 1 2 3 _4 5 _6 7 8 9 10
    // 1 | 0 0 .1 0 .3 0 0 0 0 0
    // 2 | 0 0 .5 0 .0 0 0 0 0 0
    // 3 | 0 0 .4 0 .3 0 0 0 0 0
    // 4 | 0 0 .0 0 .0 0 0 0 0 0
    // 5 | 0 0 .1 0 .3 0 0 0 0 0
    // 6 | 0 0 .0 0 .0 0 0 0 0 0
    // 7 | 0 0 .0 0 .3 0 0 0 0 0
    // 8 | 0 0 .4 0 .0 0 0 0 0 0
    // 9 | 0 0 .0 0 .3 0 0 0 0 0
    // 10| 0 0 .0 0 .0 0 0 0 0 0
    double[] expecteds = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        .1, .5, .4, 0, .1, 0, 0, .4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .3, 0, .3, 0, .3, 0, .3, 0,
        .3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    assertArrayEquals(expecteds, matrixP.getData(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void testIsLegalMaximizerPermutation() {
    int totalNumOfBits = 10;
    BinaryF1 binaryF1 = buildBinaryF1(totalNumOfBits);

    assertFalse(binaryF1.isLegalMaximizerPermutation(null));

    BitSet permutation = new BitSet();
    assertTrue(binaryF1.isLegalMaximizerPermutation(permutation));

    permutation = new BitSet();
    permutation.set(123456);
    assertTrue(binaryF1.isLegalMaximizerPermutation(permutation));
  }

  @Test
  public void testIsLegalMinimizerPermutation() {
    int totalNumOfBits = 10;
    BinaryF1 binaryF1 = buildBinaryF1(totalNumOfBits);

    assertFalse(binaryF1.isLegalMinimizerPermutation(null));

    BitSet permutation = new BitSet();
    assertTrue(binaryF1.isLegalMinimizerPermutation(permutation));

    permutation = new BitSet();
    permutation.set(654321);
    assertTrue(binaryF1.isLegalMinimizerPermutation(permutation));
  }

  private LinkedHashSet<BitSet> generateAllPossiblePermutations(int totalNumOfBits) {
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
      result.add(permutation);
    }
    return result;
  }

  private double[] generateProbabilities(Random random, int numOfPermutations,
      boolean generateZeroProb) {
    Set<Integer> zeroProbIndices = new HashSet<>();
    if (generateZeroProb) {
      while (zeroProbIndices.size() < numOfPermutations / 3 * 2) {
        zeroProbIndices.add(random.nextInt(numOfPermutations));
      }
    }

    double[] probabilities = new double[numOfPermutations];
    double probabilitiySum = 0.0;
    for (int index = 0; index < numOfPermutations; index++) {
      if (zeroProbIndices.contains(index)) {
        continue;
      }
      probabilities[index] = -Math.log(1.0 - random.nextDouble());
      probabilitiySum += probabilities[index];
    }
    for (int index = 0; index < numOfPermutations; index++) {
      probabilities[index] = probabilities[index] / probabilitiySum;
    }
    return probabilities;
  }

  @Test
  public void testFindBestMaximizerResponsePermutation() {
    for (int totalNumOfBits = 3; totalNumOfBits <= 13; totalNumOfBits++) {
      for (int testTime = 1; testTime <= 10; testTime++) {
        testFindBestMaximizerResponsePermutation(totalNumOfBits, testTime);
      }
    }
  }

  private void testFindBestMaximizerResponsePermutation(int totalNumOfBits, int testTime) {
    System.out.println("testFindBestMaximizerResponsePermutation, totalNumOfBits=" + totalNumOfBits
        + ", testTime=" + testTime);

    Random random = new Random();

    double[] lagrangePotentials = new double[totalNumOfBits];
    for (int index = 0; index < totalNumOfBits; index++) {
      lagrangePotentials[index] = -1 + 2 * random.nextDouble(); // range [-1, 1)
    }

    LinkedHashSet<BitSet> minimizerPermutations = generateAllPossiblePermutations(totalNumOfBits);
    double[] minimizerProbabilities =
        generateProbabilities(random, minimizerPermutations.size(), totalNumOfBits % 2 == 0);
    System.out.println(Arrays.toString(minimizerProbabilities));

    Entry<Double, HashSet<BitSet>> expected = bruteForceFindBestResponsePermutation(
        minimizerProbabilities, minimizerPermutations, lagrangePotentials, totalNumOfBits, true);
    System.out.println(expected.getKey() + "\t" + expected.getValue());

    BinaryF1 binaryF1 = buildBinaryF1(totalNumOfBits);
    Pair<BitSet, Double> actual = binaryF1.findBestMaximizerResponsePermutation(
        minimizerProbabilities, minimizerPermutations, lagrangePotentials);
    System.out.println(actual.getRight() + "\t" + actual.getLeft());

    assertEquals(expected.getKey(), actual.getRight(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
    assertThat(expected.getValue(), hasItem(actual.getLeft()));
  }

  private Entry<Double, HashSet<BitSet>> bruteForceFindBestResponsePermutation(
      double[] probabilities, LinkedHashSet<BitSet> permutations, double[] lagrangePotentials,
      int totalNumOfBits, boolean findMaximizer) {
    BinaryF1 binaryF1Helper = buildBinaryF1(0);

    LinkedHashSet<BitSet> responsePermutations = generateAllPossiblePermutations(totalNumOfBits);
    TreeMap<Double, HashSet<BitSet>> maximizerPermutationsByScore =
        findMaximizer ? new TreeMap<>(Comparator.reverseOrder()) : new TreeMap<>();
    double lagrangePotentialsSum = 0;

    for (BitSet responsePermutation : responsePermutations) {
      if (!findMaximizer) { // responsePermutation is minimizer
        lagrangePotentialsSum =
            binaryF1Helper.aggregateLagrangePotentials(responsePermutation, lagrangePotentials);
      }

      double responseScore = 0.0;
      int permutationIndex = 0;
      for (BitSet permutation : permutations) {
        if (findMaximizer) { // permutation is minimizer
          lagrangePotentialsSum =
              binaryF1Helper.aggregateLagrangePotentials(permutation, lagrangePotentials);
        }

        double f1 = binaryF1Helper.computeScore(responsePermutation, permutation);
        double minimizerProbability = probabilities[permutationIndex++];
        responseScore += minimizerProbability * (f1 - lagrangePotentialsSum);
      }
      CollectionUtils.putInHashSetValueMap(responseScore, responsePermutation,
          maximizerPermutationsByScore);
    }
    return Iterables.getFirst(maximizerPermutationsByScore.entrySet(), null);
  }

  @Test
  public void testFindBestMinimizerResponsePermutation() {
    for (int totalNumOfBits = 3; totalNumOfBits <= 13; totalNumOfBits++) {
      for (int testTime = 1; testTime <= 10; testTime++) {
        testFindBestMinimizerResponsePermutation(totalNumOfBits, testTime);
      }
    }
  }

  private void testFindBestMinimizerResponsePermutation(int totalNumOfBits, int testTime) {
    System.out.println("testFindBestMinimizerResponsePermutation, totalNumOfBits=" + totalNumOfBits
        + ", testTime=" + testTime);

    Random random = new Random();

    double[] lagrangePotentials = new double[totalNumOfBits];
    for (int index = 0; index < totalNumOfBits; index++) {
      lagrangePotentials[index] = -1 + 2 * random.nextDouble(); // range [-1, 1)
    }

    LinkedHashSet<BitSet> maximizerPermutations = generateAllPossiblePermutations(totalNumOfBits);
    double[] maximizerProbabilities =
        generateProbabilities(random, maximizerPermutations.size(), totalNumOfBits % 2 == 0);

    Entry<Double, HashSet<BitSet>> expected = bruteForceFindBestResponsePermutation(
        maximizerProbabilities, maximizerPermutations, lagrangePotentials, totalNumOfBits, false);
    System.out.println(expected.getKey() + "\t" + expected.getValue());

    BinaryF1 binaryF1 = buildBinaryF1(totalNumOfBits);
    Pair<BitSet, Double> actual = binaryF1.findBestMinimizerResponsePermutation(
        maximizerProbabilities, maximizerPermutations, lagrangePotentials);
    System.out.println(actual.getRight() + "\t" + actual.getLeft());

    assertEquals(expected.getKey(), actual.getRight(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
    assertThat(expected.getValue(), hasItem(actual.getLeft()));
  }
}
