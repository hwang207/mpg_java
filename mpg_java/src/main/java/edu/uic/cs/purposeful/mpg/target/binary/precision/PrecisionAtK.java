package edu.uic.cs.purposeful.mpg.target.binary.precision;

import java.util.BitSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;

import com.google.common.collect.Sets;

import net.mintern.primitive.pair.MutableDoubleIntPair;

public class PrecisionAtK extends AbstractPrecisionAtK {

  @Override
  public Set<BitSet> getInitialMinimizerPermutations() {
    BitSet allZeros = new BitSet(totalNumOfBits);
    BitSet allOnes = new BitSet(totalNumOfBits);
    allOnes.set(0, totalNumOfBits);

    return Sets.newHashSet(allZeros, allOnes);
  }

  @Override
  public boolean isLegalMinimizerPermutation(BitSet action) {
    if (action == null) {
      return false;
    }
    return true;
  }

  @Override
  public Pair<BitSet, Double> findBestMinimizerResponsePermutation(double[] maximizerProbabilities,
      LinkedHashSet<BitSet> existingMaximizerActions, double[] lagrangePotentials) {
    List<MutableDoubleIntPair> marginalProbabilitiesWithLagrangePotentials =
        computeMinimizerMarginalProbabilitiesWithLagrangePotentials(maximizerProbabilities,
            existingMaximizerActions, lagrangePotentials);

    double bestResponseValue = 0.0;
    BitSet bestResponse = new BitSet(totalNumOfBits);
    for (int bitIndex = 0; bitIndex < totalNumOfBits; bitIndex++) {
      MutableDoubleIntPair marginalProbabilityWithLagrangePotential =
          marginalProbabilitiesWithLagrangePotentials.get(bitIndex);
      // as long as the value is still negative, adding it will make the sum smaller
      if (marginalProbabilityWithLagrangePotential.left < 0) {
        bestResponseValue += marginalProbabilityWithLagrangePotential.left;
        bestResponse.set(marginalProbabilityWithLagrangePotential.right);
      }
    }

    return Pair.of(bestResponse, bestResponseValue);
  }
}
