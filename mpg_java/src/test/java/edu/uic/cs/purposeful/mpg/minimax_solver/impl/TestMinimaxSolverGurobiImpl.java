package edu.uic.cs.purposeful.mpg.minimax_solver.impl;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Before;
import org.junit.Test;

import edu.uic.cs.purposeful.mpg.common.ScoreMatrix;
import edu.uic.cs.purposeful.mpg.common.ValuePrecision;
import edu.uic.cs.purposeful.mpg.minimax_solver.MinimaxSolver;

public class TestMinimaxSolverGurobiImpl {
  private MinimaxSolver solver;

  @Before
  public void initializeSolverInstance() {
    solver = new MinimaxSolverGurobiImpl();
  }

  private ScoreMatrix createScoreMatrix(double[][] matrix) {
    ScoreMatrix scoreMatrix = new ScoreMatrix();
    for (int rowIndex = 0; rowIndex < matrix.length; rowIndex++) {
      for (int columnIndex = 0; columnIndex < matrix[0].length; columnIndex++) {
        scoreMatrix.put(rowIndex, columnIndex, matrix[rowIndex][columnIndex]);
      }
    }

    return scoreMatrix;
  }

  @Test
  public void test_1() {
    ScoreMatrix matrix = createScoreMatrix(new double[][] {{0, 4, 6}, {5, 7, 4}, {9, 6, 3}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    // p = (1/2, 0, 1/2), q = (1/4, 0, 3/4), v = 4.5
    assertEquals(actualMaximizer.getRight(), actualMinimizer.getRight());
    assertEquals(4.5, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.5, 0, 0.5}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.25, 0, 0.75}, actualMinimizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void test_2() {
    ScoreMatrix matrix = createScoreMatrix(new double[][] {{-2, 3}, {3, -4}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    // p = (0.5833, 0.4167), q = (0.5833, 0.4167), v = 0.0833
    assertEquals(actualMaximizer.getRight(), actualMinimizer.getRight());
    assertEquals(0.0833, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_3_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.5833, 0.4167}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_3_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.5833, 0.4167}, actualMinimizer.getLeft(),
        ValuePrecision.POINT_3_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void test_3() {
    ScoreMatrix matrix = createScoreMatrix(new double[][] {{4, 1, -3}, {3, 2, 5}, {0, 1, 6}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    // p = (0, 1, 0), q = (0, 1, 0), v = 2
    assertEquals(actualMaximizer.getRight(), actualMinimizer.getRight());
    assertEquals(2, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0, 1, 0}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0, 1, 0}, actualMinimizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void test_4() {
    ScoreMatrix matrix = createScoreMatrix(new double[][] {{2, -1, 6}, {0, 1, -1}, {-2, 2, 1}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    // p = (0.25, 0.75, 0), q = (0.5, 0.5, 0), v = 0.5
    assertEquals(actualMaximizer.getRight(), actualMinimizer.getRight());
    assertEquals(0.5, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.25, 0.75, 0}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.5, 0.5, 0}, actualMinimizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void test_5() {
    ScoreMatrix matrix = createScoreMatrix(new double[][] {{2, 3, 1, 5}, {4, 1, 6, 0}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    // p = (0.7143, 0.2857), q = (0, 0.7143, 0.2857, 0), v = 2.4286
    assertEquals(actualMaximizer.getRight(), actualMinimizer.getRight());
    assertEquals(2.4286, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_3_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.7143, 0.2857}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_3_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0, 0.7143, 0.2857, 0}, actualMinimizer.getLeft(),
        ValuePrecision.POINT_3_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void test_6() {
    ScoreMatrix matrix = createScoreMatrix(new double[][] {{1, 5}, {4, 4}, {6, 2}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    // p = (0, 1, 0), q = (q1=1-q2, q2=1 - any value between 1/4 and 1/2 inclusive), v = 4
    assertEquals(actualMaximizer.getRight(), actualMinimizer.getRight());
    assertEquals(4, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0, 1, 0}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertTrue(actualMinimizer.getLeft()[0] + "", actualMinimizer.getLeft()[0] >= (1.0 / 4));
    assertTrue(actualMinimizer.getLeft()[0] + "", actualMinimizer.getLeft()[0] <= (1.0 / 2));
    assertEquals(1.0, actualMinimizer.getLeft()[0] + actualMinimizer.getLeft()[1],
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void test_7() {
    ScoreMatrix matrix = createScoreMatrix(new double[][] {{1, 2, 3, 3, 6}, {2, 6, 1, 3, 3},
        {3, 1, 3, 6, 2}, {3, 3, 6, 2, 1}, {6, 3, 2, 1, 3}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    assertEquals(actualMaximizer.getRight().doubleValue(), actualMinimizer.getRight().doubleValue(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
    assertEquals(3, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.2, 0.2, 0.2, 0.2, 0.2}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.2, 0.2, 0.2, 0.2, 0.2}, actualMinimizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void test_8() {
    ScoreMatrix matrix = createScoreMatrix(
        new double[][] {{1, -2, 3, -4}, {0, 1, -2, 3}, {0, 0, 1, -2}, {0, 0, 0, 1}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    assertEquals(actualMaximizer.getRight(), actualMinimizer.getRight());
    assertEquals(0.083333, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_3_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.0833, 0.2500, 0.3333, 0.3333}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_3_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.3333, 0.3333, 0.2500, 0.0833}, actualMinimizer.getLeft(),
        ValuePrecision.POINT_3_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void test_9() {
    ScoreMatrix matrix = createScoreMatrix(new double[][] {{0, 1, -2}, {1, -2, 3}, {-2, 3, -4}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    assertEquals(actualMaximizer.getRight(), actualMinimizer.getRight(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
    assertEquals(0, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.25, 0.5, 0.25}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.25, 0.5, 0.25}, actualMinimizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void test_10() {
    ScoreMatrix matrix = createScoreMatrix(new double[][] {{1, 2, -1}, {2, -1, 4}, {-1, 4, -3}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    assertEquals(actualMaximizer.getRight(), actualMinimizer.getRight());
    assertEquals(1, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.25, 0.5, 0.25}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.25, 0.5, 0.25}, actualMinimizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  @Test
  public void test_11() {
    ScoreMatrix matrix =
        createScoreMatrix(new double[][] {{1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, 4}});

    Pair<double[], Double> actualMaximizer = solver.findMaximizerProbabilities(matrix);
    Pair<double[], Double> actualMinimizer = solver.findMinimizerProbabilities(matrix);

    assertEquals(actualMaximizer.getRight(), actualMinimizer.getRight(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
    assertEquals(0.48, actualMaximizer.getRight().doubleValue(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.48, 0.24, 0.16, 0.12}, actualMaximizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    assertArrayEquals(new double[] {0.48, 0.24, 0.16, 0.12}, actualMinimizer.getLeft(),
        ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }
}
