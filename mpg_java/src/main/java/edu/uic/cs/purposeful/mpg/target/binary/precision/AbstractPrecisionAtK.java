package edu.uic.cs.purposeful.mpg.target.binary.precision;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.MathUtils;

import com.google.common.collect.Sets;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.common.config.AbstractConfig;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.target.binary.AbstractBinaryOptimizationTarget;
import net.mintern.primitive.pair.MutableDoubleIntPair;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

abstract class AbstractPrecisionAtK extends AbstractBinaryOptimizationTarget {
  static class PrecisionAtKConfig extends AbstractConfig {
    private PrecisionAtKConfig() {
      super("mpg_precision_at_k_config.properties");
    }

    private static final PrecisionAtKConfig INSTANCE = new PrecisionAtKConfig();
    static final double K_PERCENT = INSTANCE.getDoubleValue("k_percent");
  }

  protected int k;
  protected double reciprocalK;

  @Override
  public void initialize(Pair<double[], LinkedSparseMatrix> trainingData, boolean duringTraining) {
    super.initialize(trainingData, duringTraining);

    int countOnes = 0;
    for (double goldenTagValue : goldenTagValues) {
      if (MathUtils.equals(goldenTagValue, BINARY_VALUE_ONE)) {
        countOnes++;
      } else {
        Assert.isTrue(MathUtils.equals(goldenTagValue, BINARY_VALUE_ZERO));
      }
    }
    k = Math.min((int) Math.floor(countOnes * PrecisionAtKConfig.K_PERCENT), totalNumOfBits);
    if (MPGConfig.SHOW_RUNNING_TRACING) {
      System.err
          .println("k=" + k + ", countOnes=" + countOnes + ", totalNumOfBits=" + totalNumOfBits);
    }
    Assert.isTrue(k > 0, "k must larger than zero!");
    reciprocalK = 1.0 / k;
  }

  @Override
  public double computeScore(BitSet maximizerPermutation, BitSet minimizerPermutation) {
    Assert.isTrue(maximizerPermutation.cardinality() == k, "Maximizer should be restricted to k.");

    BitSet actionsIntersection = (BitSet) minimizerPermutation.clone();
    actionsIntersection.and(maximizerPermutation);

    return ((double) actionsIntersection.cardinality()) / k;
  }

  @Override
  public Set<BitSet> getInitialMaximizerPermutations() {
    BitSet lowerBitOnes = new BitSet(totalNumOfBits);
    lowerBitOnes.set(0, k);

    BitSet upperBitOnes = new BitSet(totalNumOfBits);
    upperBitOnes.set(totalNumOfBits - k, totalNumOfBits);

    return Sets.newHashSet(lowerBitOnes, upperBitOnes);
  }

  @Override
  public Pair<BitSet, Double> findBestMaximizerResponsePermutation(double[] minimizerProbabilities,
      LinkedHashSet<BitSet> existingMinimizerActions, double[] lagrangePotentials) {
    List<MutableDoubleIntPair> marginalProbabilities = new ArrayList<>(totalNumOfBits);
    for (int bitIndex = 0; bitIndex < totalNumOfBits; bitIndex++) {
      marginalProbabilities.add(new MutableDoubleIntPair(0.0, bitIndex));
    }

    int actionIndex = 0;
    for (BitSet minimizerAction : existingMinimizerActions) {
      double minimizerProbability = minimizerProbabilities[actionIndex++];
      if (Misc.roughlyEquals(minimizerProbability, 0.0)) {
        continue;
      }

      for (int bitIndex = 0; bitIndex < totalNumOfBits; bitIndex++) {
        if (minimizerAction.get(bitIndex)) {
          MutableDoubleIntPair marginalProbability = marginalProbabilities.get(bitIndex);
          marginalProbability.left += minimizerProbability;
        }
      }
    }

    Collections.sort(marginalProbabilities, Comparator.reverseOrder());

    double lagrangePotentialsSum = 0.0;
    double bestResponseValue = 0.0;
    BitSet bestResponse = new BitSet(totalNumOfBits);
    for (int bitIndex = 0; bitIndex < totalNumOfBits; bitIndex++) {
      MutableDoubleIntPair marginalProbability = marginalProbabilities.get(bitIndex);
      if (bitIndex < k) {
        bestResponseValue += marginalProbability.left;
        bestResponse.set(marginalProbability.right);
      }

      lagrangePotentialsSum +=
          (marginalProbability.left * lagrangePotentials[marginalProbability.right]);
    }
    bestResponseValue = bestResponseValue / k - lagrangePotentialsSum;

    return Pair.of(bestResponse, bestResponseValue);
  }

  @Override
  public boolean isLegalMaximizerPermutation(BitSet action) {
    if (action == null) {
      return false;
    }
    return action.cardinality() == k;
  }

  protected List<MutableDoubleIntPair> computeMinimizerMarginalProbabilitiesWithLagrangePotentials(
      double[] maximizerProbabilities, LinkedHashSet<BitSet> existingMaximizerActions,
      double[] lagrangePotentials) {
    List<MutableDoubleIntPair> marginalProbabilitiesWithLagrangePotentials =
        new ArrayList<>(totalNumOfBits);
    for (int bitIndex = 0; bitIndex < totalNumOfBits; bitIndex++) {
      marginalProbabilitiesWithLagrangePotentials.add(new MutableDoubleIntPair(0.0, bitIndex));
    }

    int maximizerActionIndex = 0;
    for (BitSet maximizerAction : existingMaximizerActions) {
      double maximizerProbability = maximizerProbabilities[maximizerActionIndex++];
      if (Misc.roughlyEquals(maximizerProbability, 0.0)) {
        continue;
      }

      for (int bitIndex = 0; bitIndex < totalNumOfBits; bitIndex++) {
        MutableDoubleIntPair marginalProbabilityWithLagrangePotential =
            marginalProbabilitiesWithLagrangePotentials.get(bitIndex);
        marginalProbabilityWithLagrangePotential.left += maximizerProbability
            * ((maximizerAction.get(bitIndex) ? reciprocalK : 0) - lagrangePotentials[bitIndex]);
      }
    }
    return marginalProbabilitiesWithLagrangePotentials;
  }
}
