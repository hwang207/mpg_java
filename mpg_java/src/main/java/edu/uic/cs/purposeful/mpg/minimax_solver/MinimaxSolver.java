package edu.uic.cs.purposeful.mpg.minimax_solver;

import org.apache.commons.lang3.tuple.Pair;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.common.ScoreMatrix;

public abstract class MinimaxSolver {
  protected class MatrixWrapper {
    private final ScoreMatrix matrix;
    private final boolean isNegativeTransposed;

    MatrixWrapper(ScoreMatrix matrix) {
      this(matrix, false);
    }

    MatrixWrapper(ScoreMatrix matrix, boolean isTransposed) {
      this.matrix = matrix;
      this.isNegativeTransposed = isTransposed;
    }

    public int getNumberOfRows() {
      return isNegativeTransposed ? matrix.getColumnSize() : matrix.getRowSize();
    }

    public int getNumberOfColumns() {
      return isNegativeTransposed ? matrix.getRowSize() : matrix.getColumnSize();
    }

    public double getValue(int rowIndex, int columnIndex) {
      return isNegativeTransposed ? (-matrix.get(columnIndex, rowIndex))
          : matrix.get(rowIndex, columnIndex);
    }
  }

  protected static final double OBJECTIVE_COEFFICIENT = 1.0;
  protected static final double RHS_VALUE = 1.0;
  protected static final double TIME_OUT_SECONDS = 60;

  public Pair<double[], Double> findMaximizerProbabilities(ScoreMatrix scoreMatrix) {
    Assert.isTrue(scoreMatrix.getRowSize() > 0, "scoreMatrix.getRowSize() <= 0");
    Assert.isTrue(scoreMatrix.getColumnSize() > 0, "scoreMatrix.getColumnSize() <= 0");
    return findMaximizerProbabilities(new MatrixWrapper(scoreMatrix), scoreMatrix.getMinimum(),
        scoreMatrix.getMaximum());
  }

  public Pair<double[], Double> findMinimizerProbabilities(ScoreMatrix scoreMatrix) {
    Assert.isTrue(scoreMatrix.getRowSize() > 0, "scoreMatrix.getRowSize() <= 0");
    Assert.isTrue(scoreMatrix.getColumnSize() > 0, "scoreMatrix.getColumnSize() <= 0");
    Pair<double[], Double> internalMinResult = findMaximizerProbabilities(
        new MatrixWrapper(scoreMatrix, true), -scoreMatrix.getMaximum(), -scoreMatrix.getMinimum());
    // value is has the same sign with maximizer
    return Pair.of(internalMinResult.getLeft(), -internalMinResult.getRight());
  }

  abstract protected Pair<double[], Double> findMaximizerProbabilities(MatrixWrapper matrixWrapper,
      double minimum, double maximum);
}
