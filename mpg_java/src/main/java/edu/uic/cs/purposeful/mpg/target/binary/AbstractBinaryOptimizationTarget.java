package edu.uic.cs.purposeful.mpg.target.binary;

import java.util.BitSet;
import java.util.LinkedHashSet;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.MathUtils;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.target.OptimizationTarget;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public abstract class AbstractBinaryOptimizationTarget
    implements OptimizationTarget<BitSet, Pair<double[], LinkedSparseMatrix>> {
  public static final double BINARY_VALUE_ONE = 1.0;
  public static final double BINARY_VALUE_ZERO = 0.0;

  // we consider the whole training data as one instance, each y_i is one bit
  protected int totalNumOfBits;
  protected int totalNumOfFeatures;

  protected double[] goldenTagValues;
  private BitSet goldenPermutation;
  protected LinkedSparseMatrix featureMatrix;
  private Vector goldenFeatureValues;

  @Override
  public void initialize(Pair<double[], LinkedSparseMatrix> trainingData, boolean duringTraining) {
    this.goldenTagValues = trainingData.getLeft();
    this.featureMatrix = trainingData.getRight();

    this.totalNumOfBits = featureMatrix.numRows();
    this.totalNumOfFeatures = featureMatrix.numColumns();

    this.goldenPermutation = buildPermutationFromTagValues(goldenTagValues);
    if (duringTraining) {
      this.goldenFeatureValues = computeGoldenFeatureValues();
    }
  }

  private Vector computeGoldenFeatureValues() {
    // 1 * #bits
    DenseMatrix goldTagValueVector = new DenseMatrix(1, totalNumOfBits, goldenTagValues, false);
    // 1 * #features
    double[] goldFeatureValues = new double[totalNumOfFeatures];
    // (1 * #bits) * (#bits * #features) = (1 * #features)
    goldTagValueVector.mult(featureMatrix,
        new DenseMatrix(1, totalNumOfFeatures, goldFeatureValues, false));
    return new DenseVector(goldFeatureValues, false);
  }

  @Override
  public double aggregateLagrangePotentials(BitSet minimizerAction, double[] lagrangePotentials) {
    double sum = 0.0;
    for (int index = minimizerAction.nextSetBit(0); index >= 0
        && index < lagrangePotentials.length; index = minimizerAction.nextSetBit(index + 1)) {
      sum += lagrangePotentials[index];
    }
    return sum;
  }

  @Override
  public double[] computeLagrangePotentials(double[] thetas) {
    double[] lagrangePotentials = new double[totalNumOfBits];
    featureMatrix.mult(new DenseVector(thetas, false), new DenseVector(lagrangePotentials, false));
    return lagrangePotentials;
  }

  @Override
  public BitSet getGoldenPermutation() {
    return goldenPermutation;
  }

  @Override
  public Vector getGoldenFeatureValues() {
    return goldenFeatureValues;
  }

  @Override
  public Vector computeExpectedFeatureValues(double[] probabilities,
      LinkedHashSet<BitSet> permutations) {
    double[] bitMarginalProbabilities = new double[totalNumOfBits];
    int permutationIndex = 0;
    for (BitSet permutation : permutations) {
      double probability = probabilities[permutationIndex++];
      if (Misc.roughlyEquals(probability, 0)) {
        continue; // permutation has no contribution, skip
      }

      for (int bitIndex = permutation.nextSetBit(0); bitIndex >= 0
          && bitIndex < totalNumOfBits; bitIndex = permutation.nextSetBit(bitIndex + 1)) {
        bitMarginalProbabilities[bitIndex] += probability;
      }
    }

    double[] featureValueExpectations = new double[totalNumOfFeatures];
    new DenseMatrix(1, totalNumOfBits, bitMarginalProbabilities, false).mult(featureMatrix,
        new DenseMatrix(1, totalNumOfFeatures, featureValueExpectations, false));

    return new DenseVector(featureValueExpectations, false);
  }

  protected BitSet buildPermutationFromTagValues(double[] tagValues) {
    BitSet permutation = new BitSet(tagValues.length);
    for (int bitIndex = 0; bitIndex < tagValues.length; bitIndex++) {
      double goldTagValue = tagValues[bitIndex];
      if (MathUtils.equals(goldTagValue, BINARY_VALUE_ONE)) {
        permutation.set(bitIndex);
      } else {
        Assert.isTrue(MathUtils.equals(goldTagValue, BINARY_VALUE_ZERO));
      }
    }
    return permutation;
  }
}
