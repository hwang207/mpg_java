package edu.uic.cs.purposeful.mpg.target;

import java.util.LinkedHashSet;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;

import no.uib.cipr.matrix.Vector;

public interface OptimizationTarget<Permutation, InitialData> {

  void initialize(InitialData initialData, boolean duringTraining);

  double computeScore(Permutation maximizerPermutation, Permutation minimizerPermutation);

  Set<Permutation> getInitialMaximizerPermutations();

  Set<Permutation> getInitialMinimizerPermutations();

  double[] computeLagrangePotentials(double[] thetas);

  double aggregateLagrangePotentials(Permutation minimizerPermutation, double[] lagrangePotentials);

  Pair<Permutation, Double> findBestMaximizerResponsePermutation(double[] minimizerProbabilities,
      LinkedHashSet<Permutation> existingMinimizerPermutations, double[] lagrangePotentials);

  Pair<Permutation, Double> findBestMinimizerResponsePermutation(double[] maximizerProbabilities,
      LinkedHashSet<Permutation> existingMaximizerPermutations, double[] lagrangePotentials);

  boolean isLegalMaximizerPermutation(Permutation permutation);

  boolean isLegalMinimizerPermutation(Permutation permutation);

  Permutation getGoldenPermutation();

  /**
   * feature(goldenPermutation)
   */
  Vector getGoldenFeatureValues();

  /**
   * SUM_permutation{probabilities[permutation]*feature(permutation)}
   */
  Vector computeExpectedFeatureValues(double[] minimizerProbabilities,
      LinkedHashSet<Permutation> minimizerPermutations);
}
