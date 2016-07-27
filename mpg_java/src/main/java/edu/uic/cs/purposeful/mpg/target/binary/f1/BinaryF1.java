package edu.uic.cs.purposeful.mpg.target.binary.f1;

import java.util.BitSet;
import java.util.LinkedHashSet;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Sets;

import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.target.binary.AbstractBinaryOptimizationTarget;
import edu.uic.cs.purposeful.mpg.target.common.GeneralFMeasureMaximizer;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public class BinaryF1 extends AbstractBinaryOptimizationTarget {
  private GeneralFMeasureMaximizer gfm;

  @Override
  public void initialize(Pair<double[], LinkedSparseMatrix> trainingData, boolean duringTraining) {
    super.initialize(trainingData, duringTraining);
    gfm = new GeneralFMeasureMaximizer(totalNumOfBits);
  }

  @Override
  public double computeScore(BitSet maximizerPermutation, BitSet minimizerPermutation) {
    int totalNumOfOnes = maximizerPermutation.cardinality() + minimizerPermutation.cardinality();
    if (totalNumOfOnes == 0) {
      return 1.0;
    }

    BitSet permutationsIntersection = (BitSet) minimizerPermutation.clone();
    permutationsIntersection.and(maximizerPermutation);

    return 2.0 * permutationsIntersection.cardinality() / totalNumOfOnes;
  }

  @Override
  public Set<BitSet> getInitialMaximizerPermutations() {
    BitSet allZeros = new BitSet(totalNumOfBits);
    BitSet allOnes = new BitSet(totalNumOfBits);
    allOnes.set(0, totalNumOfBits);
    return Sets.newHashSet(allZeros, allOnes);
  }

  @Override
  public Set<BitSet> getInitialMinimizerPermutations() {
    return getInitialMaximizerPermutations();
  }

  @Override
  public Pair<BitSet, Double> findBestMaximizerResponsePermutation(double[] minimizerProbabilities,
      LinkedHashSet<BitSet> existingMinimizerPermutations, double[] lagrangePotentials) {
    Pair<Double, LinkedSparseMatrix> p0AndMatrixP =
        computeMarginalProbabilityMatrixP(minimizerProbabilities, existingMinimizerPermutations);
    double p0 = p0AndMatrixP.getLeft();
    LinkedSparseMatrix matrixP = p0AndMatrixP.getRight();
    Pair<BitSet, Double> bestMaximizerResponse = gfm.gfm(p0, matrixP);

    double lagrangePotentialsExpectation = computeLagrangePotentialsExpectation(
        minimizerProbabilities, existingMinimizerPermutations, lagrangePotentials);
    double bestResponseValue = bestMaximizerResponse.getRight() - lagrangePotentialsExpectation;

    return Pair.of(bestMaximizerResponse.getLeft(), bestResponseValue);
  }

  private double computeLagrangePotentialsExpectation(double[] minimizerProbabilities,
      LinkedHashSet<BitSet> existingMinimizerPermutations, double[] lagrangePotentials) {
    double lagrangePotentialsExpectation = 0;
    int permutationIndex = 0;
    for (BitSet permutation : existingMinimizerPermutations) {
      double probability = minimizerProbabilities[permutationIndex++];
      if (Misc.roughlyEquals(probability, 0)) {
        continue;
      }
      double lagrangePotential = aggregateLagrangePotentials(permutation, lagrangePotentials);
      lagrangePotentialsExpectation += probability * lagrangePotential;
    }
    return lagrangePotentialsExpectation;
  }

  @Override
  public Pair<BitSet, Double> findBestMinimizerResponsePermutation(double[] maximizerProbabilities,
      LinkedHashSet<BitSet> existingMaximizerPermutations, double[] lagrangePotentials) {
    Pair<Double, LinkedSparseMatrix> p0AndMatrixP =
        computeMarginalProbabilityMatrixP(maximizerProbabilities, existingMaximizerPermutations);
    double p0 = p0AndMatrixP.getLeft();
    LinkedSparseMatrix matrixP = p0AndMatrixP.getRight();
    return gfm.gfm(p0, matrixP, lagrangePotentials);
  }

  @VisibleForTesting
  Pair<Double, LinkedSparseMatrix> computeMarginalProbabilityMatrixP(double[] probabilities,
      LinkedHashSet<BitSet> permutations) {
    double p0 = 0.0;
    LinkedSparseMatrix matrixP = new LinkedSparseMatrix(totalNumOfBits, totalNumOfBits);

    int permutationIndex = 0;
    for (BitSet permutation : permutations) {
      double probability = probabilities[permutationIndex++];
      if (Misc.roughlyEquals(probability, 0)) {
        continue;
      }

      int numOfOnes = permutation.cardinality(); // s
      if (numOfOnes == 0) {
        p0 += probability;
      } else {
        for (int index = permutation.nextSetBit(0); index >= 0; index =
            permutation.nextSetBit(index + 1)) { // i
          // the probability that there are total s '1's, and i-th position is '1'
          matrixP.add(index, numOfOnes - 1, probability);
        }
      }
    }

    return Pair.of(p0, matrixP);
  }

  @Override
  public boolean isLegalMaximizerPermutation(BitSet permutation) {
    return permutation != null;
  }

  @Override
  public boolean isLegalMinimizerPermutation(BitSet permutation) {
    return permutation != null;
  }
}
