package edu.uic.cs.purposeful.mpg.optimizer.game.impl;

import java.util.LinkedHashSet;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.common.assertion.PurposefulBaseException;
import edu.uic.cs.purposeful.common.reflect.ClassFactory;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.common.ScoreMatrix;
import edu.uic.cs.purposeful.mpg.minimax_solver.MinimaxSolver;
import edu.uic.cs.purposeful.mpg.optimizer.game.ZeroSumGameSolver;
import edu.uic.cs.purposeful.mpg.target.OptimizationTarget;

public class DoubleOracleGameSolver<Permutation> implements ZeroSumGameSolver<Permutation> {
  private static final Logger LOGGER = Logger.getLogger(DoubleOracleGameSolver.class);

  private static final int MAX_NUM_OF_PERMUTATIONS =
      MPGConfig.MAX_NUM_OF_DOUBLE_ORACLE_PERMUTATIONS;

  private final OptimizationTarget<Permutation, ?> optimizationTarget;
  private final MinimaxSolver minimaxSolver;
  private final MinimaxSolver minimaxBackupSolver;

  private ScoreMatrix scoreMatrix;
  private LinkedHashSet<Permutation> existingMaximizerPermutations;
  private LinkedHashSet<Permutation> existingMinimizerPermutations;
  private double[] maximizerProbabilities;
  private double[] minimizerProbabilities;
  private double maximizerValue;
  private double minimizerValue;

  private boolean hasTriedToSolve = false;

  public DoubleOracleGameSolver(OptimizationTarget<Permutation, ?> optimizationTarget) {
    this.optimizationTarget = optimizationTarget;
    this.minimaxSolver = ClassFactory.getInstance(MPGConfig.MINIMAX_SOLVER_CLASS);
    if (!MPGConfig.MINIMAX_SOLVER_CLASS_BACKUP.isEmpty()) {
      this.minimaxBackupSolver = ClassFactory.getInstance(MPGConfig.MINIMAX_SOLVER_CLASS_BACKUP);
    } else {
      this.minimaxBackupSolver = null;
    }
  }

  @Override
  public boolean solve(double[] thetas) {
    return solve(thetas, null);
  }

  @Override
  public boolean solve(double[] thetas, Permutation goldPermutation) {
    try {
      return solve(thetas, goldPermutation, minimaxSolver);
    } catch (PurposefulBaseException e) {
      if (minimaxBackupSolver == null) {
        throw e;
      }
      LOGGER.error(
          MPGConfig.MINIMAX_SOLVER_CLASS + " failed, try " + MPGConfig.MINIMAX_SOLVER_CLASS_BACKUP,
          e);
    }
    return solve(thetas, goldPermutation, minimaxBackupSolver);
  }

  private boolean solve(double[] thetas, Permutation goldPermutation, MinimaxSolver minimaxSolver) {
    double[] lagrangePotentials = optimizationTarget.computeLagrangePotentials(thetas);
    initializeScoreMatrix(lagrangePotentials, goldPermutation);

    double previousGameValue = Double.NaN;
    boolean converged = false;
    while (true) {
      // compute maximizer's distribution
      Pair<double[], Double> maximizerDistribution =
          minimaxSolver.findMaximizerProbabilities(scoreMatrix);
      maximizerProbabilities = maximizerDistribution.getLeft();
      maximizerValue = maximizerDistribution.getRight();
      // reaches Nash equilibrium
      if (Misc.roughlyEquals(previousGameValue, maximizerValue)) {
        outputToConsole("$");
        if (LOGGER.isDebugEnabled()) {
          LOGGER.debug("Optimized - maximizer's game value [" + maximizerValue
              + "] doesn't change anymore.");
        }
        converged = true;
        break;
      }
      previousGameValue = maximizerValue;

      // find minimizer's response
      Assert.isTrue(existingMaximizerPermutations.size() == maximizerProbabilities.length);
      if (MAX_NUM_OF_PERMUTATIONS > 0
          && MAX_NUM_OF_PERMUTATIONS <= existingMaximizerPermutations.size()) {
        outputToConsole("X");
        if (LOGGER.isDebugEnabled()) {
          LOGGER.debug("Not optimized - max number of maximizers' permutations["
              + MAX_NUM_OF_PERMUTATIONS + "] is reached.");
        }
        converged = false;
        break;
      }

      Pair<Permutation, Double> bestMinimizerResponse =
          optimizationTarget.findBestMinimizerResponsePermutation(maximizerProbabilities,
              existingMaximizerPermutations, lagrangePotentials);
      Permutation bestMinimizerResponsePermutation = bestMinimizerResponse.getLeft();
      double bestMinimizerResponseValue = bestMinimizerResponse.getRight();

      boolean reachesBestMinimizerValue =
          Misc.roughlyEquals(maximizerValue, bestMinimizerResponseValue);
      if (!reachesBestMinimizerValue) {
        if (!existingMinimizerPermutations.contains(bestMinimizerResponsePermutation)) {
          recordMinimizerPermutationAndExpandScoreMatrix(bestMinimizerResponsePermutation,
              lagrangePotentials);
        } else {
          reachesBestMinimizerValue = true;
          outputToConsole("<");
        }
      } else {
        outputToConsole("v");
      }

      // ///////////////////////////////////////////////////////////////////////
      // compute minimizer's distribution
      Pair<double[], Double> minimizerDistribution =
          minimaxSolver.findMinimizerProbabilities(scoreMatrix);
      minimizerProbabilities = minimizerDistribution.getLeft();
      minimizerValue = minimizerDistribution.getRight();
      // reaches Nash equilibrium
      if (Misc.roughlyEquals(previousGameValue, minimizerValue)) {
        outputToConsole("&");
        if (LOGGER.isDebugEnabled()) {
          LOGGER.debug("Optimized - minimizer's game value [" + minimizerValue
              + "] doesn't change anymore.");
        }
        converged = true;
        break;
      }
      previousGameValue = minimizerValue;

      // find maximizer's response
      Assert.isTrue(existingMinimizerPermutations.size() == minimizerProbabilities.length);
      if (MAX_NUM_OF_PERMUTATIONS > 0
          && MAX_NUM_OF_PERMUTATIONS <= existingMinimizerPermutations.size()) {
        outputToConsole("x");
        if (LOGGER.isDebugEnabled()) {
          LOGGER.debug("Not optimized - max number of minimizers' permutations["
              + MAX_NUM_OF_PERMUTATIONS + "] is reached.");
        }
        converged = false;
        break;
      }

      Pair<Permutation, Double> bestMaximizerResponse =
          optimizationTarget.findBestMaximizerResponsePermutation(minimizerProbabilities,
              existingMinimizerPermutations, lagrangePotentials);
      Permutation bestMaximizerResponsePermutation = bestMaximizerResponse.getLeft();
      double bestMaximizerResponseValue = bestMaximizerResponse.getRight();

      boolean reachesBestMaximizerValue =
          Misc.roughlyEquals(minimizerValue, bestMaximizerResponseValue);
      if (!reachesBestMaximizerValue) {
        if (!existingMaximizerPermutations.contains(bestMaximizerResponsePermutation)) {
          recordMaximizerPermutationAndExpandScoreMatrix(bestMaximizerResponsePermutation,
              lagrangePotentials);
        } else {
          reachesBestMaximizerValue = true;
          outputToConsole(">");
        }
      } else {
        outputToConsole("^");
      }

      if (reachesBestMinimizerValue && reachesBestMaximizerValue) {
        outputToConsole("%");
        if (LOGGER.isDebugEnabled()) {
          LOGGER.debug("Optimized - both players' responses don't change the game anymore.");
        }
        converged = true;
        break;
      }
    } // end while

    outputToConsole("\n");
    hasTriedToSolve = true;
    return converged;
  }

  private void outputToConsole(String info) {
    if (MPGConfig.SHOW_RUNNING_TRACING) {
      System.err.print(info);
    }
  }

  private void recordMaximizerPermutationAndExpandScoreMatrix(Permutation maximizerPermutation,
      double[] lagrangePotentials) {
    existingMaximizerPermutations.add(maximizerPermutation);

    int columnIndex = 0;
    int rowIndex = scoreMatrix.getRowSize();
    for (Permutation minimizerPermutation : existingMinimizerPermutations) {
      // Lagrange potentials are computed from minimizer permutation
      double aggregatedLagrangePotential =
          optimizationTarget.aggregateLagrangePotentials(minimizerPermutation, lagrangePotentials);
      double score = optimizationTarget.computeScore(maximizerPermutation, minimizerPermutation)
          - aggregatedLagrangePotential;
      Assert.isFalse(Double.isNaN(score),
          optimizationTarget.getClass().getName() + ".computeScore() should not return NaN.");
      scoreMatrix.put(rowIndex, columnIndex, score);
      columnIndex++;
    }
  }

  private void recordMinimizerPermutationAndExpandScoreMatrix(Permutation minimizerPermutation,
      double[] lagrangePotentials) {
    existingMinimizerPermutations.add(minimizerPermutation);

    double aggregatedLagrangePotential =
        optimizationTarget.aggregateLagrangePotentials(minimizerPermutation, lagrangePotentials);

    int columnIndex = scoreMatrix.getColumnSize();
    int rowIndex = 0;
    for (Permutation maximizerPermutation : existingMaximizerPermutations) {
      double score = optimizationTarget.computeScore(maximizerPermutation, minimizerPermutation)
          - aggregatedLagrangePotential;
      Assert.isFalse(Double.isNaN(score),
          optimizationTarget.getClass().getName() + ".computeScore() should not return NaN.");
      scoreMatrix.put(rowIndex, columnIndex, score);
      rowIndex++;
    }
  }

  private void initializeScoreMatrix(double[] lagrangePotentials, Permutation goldPermutation) {
    scoreMatrix = new ScoreMatrix();
    existingMaximizerPermutations =
        new LinkedHashSet<>(optimizationTarget.getInitialMaximizerPermutations());
    Assert.notEmpty(existingMaximizerPermutations);
    existingMinimizerPermutations =
        new LinkedHashSet<>(optimizationTarget.getInitialMinimizerPermutations());
    Assert.notEmpty(existingMinimizerPermutations);

    if (optimizationTarget.isLegalMaximizerPermutation(goldPermutation)) {
      existingMaximizerPermutations.add(goldPermutation);
    }
    if (optimizationTarget.isLegalMinimizerPermutation(goldPermutation)) {
      existingMinimizerPermutations.add(goldPermutation);
    }

    int columnIndex = 0;
    for (Permutation minimizerPermutation : existingMinimizerPermutations) {
      // Lagrange potentials are computed from minimizer permutation
      double aggregatedLagrangePotential =
          optimizationTarget.aggregateLagrangePotentials(minimizerPermutation, lagrangePotentials);

      int rowIndex = 0;
      for (Permutation maximizerPermutation : existingMaximizerPermutations) {
        double score = optimizationTarget.computeScore(maximizerPermutation, minimizerPermutation)
            - aggregatedLagrangePotential;
        Assert.isFalse(Double.isNaN(score),
            optimizationTarget.getClass().getName() + ".computeScore() should not return NaN.");
        scoreMatrix.put(rowIndex, columnIndex, score);
        rowIndex++;
      }
      columnIndex++;
    }
  }

  @Override
  public double getMaximizerValue() {
    Assert.isTrue(hasTriedToSolve, "Should call 'solve(...)' first.");
    return maximizerValue;
  }

  @Override
  public double getMinimizerValue() {
    Assert.isTrue(hasTriedToSolve, "Should call 'solve(...)' first.");
    return minimizerValue;
  }

  @Override
  public double[] getMaximizerProbabilities() {
    Assert.isTrue(hasTriedToSolve, "Should call 'solve(...)' first.");
    return maximizerProbabilities;
  }

  @Override
  public double[] getMinimizerProbabilities() {
    Assert.isTrue(hasTriedToSolve, "Should call 'solve(...)' first.");
    return minimizerProbabilities;
  }

  @Override
  public LinkedHashSet<Permutation> getMaximizerPermutations() {
    Assert.isTrue(hasTriedToSolve, "Should call 'solve(...)' first.");
    return existingMaximizerPermutations;
  }

  @Override
  public LinkedHashSet<Permutation> getMinimizerPermutations() {
    Assert.isTrue(hasTriedToSolve, "Should call 'solve(...)' first.");
    return existingMinimizerPermutations;
  }
}
