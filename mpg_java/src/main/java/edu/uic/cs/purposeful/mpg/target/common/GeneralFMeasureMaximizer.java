package edu.uic.cs.purposeful.mpg.target.common;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

import org.apache.commons.lang3.tuple.Pair;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import net.mintern.primitive.pair.DoubleIntPair;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.UpperSymmDenseMatrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public class GeneralFMeasureMaximizer {
  private static class ComputeMatrixW implements Function<Integer, UpperSymmDenseMatrix> {
    @Override
    public UpperSymmDenseMatrix apply(Integer totalNumOfPositions) {
      outputToConsole("Begin computing matrixW for " + totalNumOfPositions);
      UpperSymmDenseMatrix w = new UpperSymmDenseMatrix(totalNumOfPositions);
      for (int rowIndex = 0; rowIndex < totalNumOfPositions; rowIndex++) {
        for (int columnIndex = rowIndex; columnIndex < totalNumOfPositions; columnIndex++) {
          w.set(rowIndex, columnIndex, 1.0 / (rowIndex + columnIndex + 2));
        }
      }
      outputToConsole("End computing matrixW for " + totalNumOfPositions);
      return w;
    }
  }

  private static final ConcurrentHashMap<Integer, UpperSymmDenseMatrix> MATRIX_W_CACHE =
      new ConcurrentHashMap<>();

  protected final int numOfPositions;

  protected static void outputToConsole(String info) {
    if (MPGConfig.SHOW_RUNNING_TRACING) {
      System.err.println(info);
    }
  }

  public GeneralFMeasureMaximizer(int totalNumOfPositions) {
    this.numOfPositions = totalNumOfPositions;
  }

  protected UpperSymmDenseMatrix getMatrixW() {
    return MATRIX_W_CACHE.computeIfAbsent(numOfPositions, new ComputeMatrixW());
  }

  /**
   * This method returns a permutation that <b>MAXIMIZE</b> the F1 score;
   */
  public Pair<BitSet, Double> gfm(double p0, LinkedSparseMatrix matrixP) {
    return gfm(p0, matrixP, null);
  }

  /**
   * If lagrangePotentials <b>is null</b>, this method returns a permutation that <b>MAXIMIZE</b>
   * the F1 score; otherwise, it returns a permutation <b>MINIMIZE</b> the F1 score.
   */
  public Pair<BitSet, Double> gfm(double p0, LinkedSparseMatrix matrixP,
      double[] lagrangePotentials) {
    boolean findMaximize = lagrangePotentials == null;
    Assert.isTrue(findMaximize || lagrangePotentials.length == numOfPositions);

    DenseMatrix scoreMatrix = null;
    if (findMaximize) {
      scoreMatrix = new DenseMatrix(numOfPositions, numOfPositions);
      matrixP.mult(2, getMatrixW(), scoreMatrix);
    } else {
      scoreMatrix = initializeScoreMatrixWithNegativeLagrangePotentials(lagrangePotentials);
      matrixP.multAdd(2, getMatrixW(), scoreMatrix);
    }

    List<List<DoubleIntPair>> rowIndicesByOrderedValueInColumns =
        storeAndSortPositionIndices(scoreMatrix, findMaximize);

    return findTheBestResponse(p0, rowIndicesByOrderedValueInColumns, findMaximize);
  }

  protected List<List<DoubleIntPair>> storeAndSortPositionIndices(DenseMatrix scoreMatrix,
      boolean findMaximize) {
    List<List<DoubleIntPair>> rowIndicesByOrderedValueInColumns =
        new ArrayList<>(scoreMatrix.numColumns());
    for (int index = 0; index < scoreMatrix.numColumns(); index++) {
      rowIndicesByOrderedValueInColumns.add(new ArrayList<>(scoreMatrix.numRows()));
    }

    for (MatrixEntry entry : scoreMatrix) {
      List<DoubleIntPair> indicesByOrderedValue =
          rowIndicesByOrderedValueInColumns.get(entry.column());
      indicesByOrderedValue.add(DoubleIntPair.of(entry.get(), entry.row()));
    }

    for (List<DoubleIntPair> indicesByOrderedValue : rowIndicesByOrderedValueInColumns) {
      if (findMaximize) {
        Collections.sort(indicesByOrderedValue, Collections.reverseOrder());
      } else {
        Collections.sort(indicesByOrderedValue);
      }
    }

    return rowIndicesByOrderedValueInColumns;
  }

  protected Pair<BitSet, Double> findTheBestResponse(double p0,
      List<List<DoubleIntPair>> rowIndicesByOrderedValueInColumns, boolean findMaximize) {
    double bestValueSum = p0;
    int bestPermutationNumOfOnes = 0;
    BitSet bestPermutation = new BitSet(numOfPositions); // all zeros

    for (int columnIndex = 0; columnIndex < rowIndicesByOrderedValueInColumns
        .size(); columnIndex++) {
      List<DoubleIntPair> indicesByOrderedValue =
          rowIndicesByOrderedValueInColumns.get(columnIndex);
      int numOfOnes = columnIndex + 1;
      BitSet permutation = new BitSet(numOfPositions);
      double valueSum = 0.0;

      for (int retriveIndex = 0; retriveIndex < numOfOnes; retriveIndex++) {
        DoubleIntPair indexByValue = indicesByOrderedValue.get(retriveIndex);
        valueSum += indexByValue.getLeft();
        permutation.set(indexByValue.getRight());
      }

      if ((findMaximize && valueSum > bestValueSum) || (!findMaximize && valueSum < bestValueSum)) {
        bestValueSum = valueSum;
        bestPermutationNumOfOnes = numOfOnes;
        bestPermutation = permutation;
      }
    }

    // If using sparse matrix it is possible the best permutation has only several high/low order
    // bits which has non-zero values; we still need to fill more bits to reach the total number of
    // ones, even those bits have zero values.
    // int oneBitsNeedToFill = bestPermutationNumOfOnes - bestPermutation.cardinality();
    // Assert.isTrue(oneBitsNeedToFill >= 0);
    // if (oneBitsNeedToFill > 0) {
    // BitSet oneBitsAvailable = new BitSet(totalNumOfPositions);
    // oneBitsAvailable.set(0, totalNumOfPositions);
    // oneBitsAvailable.xor(bestPermutation); // find bit positions haven't been used
    //
    // int count = 0;
    // for (int bitIndex = oneBitsAvailable.nextSetBit(0); count < oneBitsNeedToFill
    // && bitIndex >= 0; bitIndex = oneBitsAvailable.nextSetBit(bitIndex + 1), count++) {
    // bestPermutation.set(bitIndex);
    // }
    // }

    Assert.isTrue(bestPermutation.cardinality() == bestPermutationNumOfOnes);
    return Pair.of(bestPermutation, bestValueSum);
  }

  private DenseMatrix initializeScoreMatrixWithNegativeLagrangePotentials(
      double[] lagrangePotentials) {
    double[] negativeLagrangePotentials = new double[lagrangePotentials.length];
    for (int index = 0; index < lagrangePotentials.length; index++) {
      negativeLagrangePotentials[index] = -lagrangePotentials[index];
    }
    double[] negativeLagrangePotentialMatrixInArray = new double[numOfPositions * numOfPositions];
    for (int index = 0; index < numOfPositions; index++) {
      System.arraycopy(negativeLagrangePotentials, 0, negativeLagrangePotentialMatrixInArray,
          index * numOfPositions, numOfPositions);
    }
    return new DenseMatrix(numOfPositions, numOfPositions, negativeLagrangePotentialMatrixInArray,
        false);
  }
}
