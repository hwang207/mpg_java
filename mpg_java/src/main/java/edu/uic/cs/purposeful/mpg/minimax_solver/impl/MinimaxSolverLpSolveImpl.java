package edu.uic.cs.purposeful.mpg.minimax_solver.impl;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.MathUtils;
import org.apache.log4j.Logger;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.common.assertion.PurposefulBaseException;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.minimax_solver.MinimaxSolver;
import lpsolve.LpSolve;
import lpsolve.LpSolveException;

public class MinimaxSolverLpSolveImpl extends MinimaxSolver {
  private static final Logger LOGGER = Logger.getLogger(MinimaxSolverLpSolveImpl.class);

  private static final Map<Integer, Pair<String, String>> ERROR_CODES = initializeErrorCodes();

  private static Map<Integer, Pair<String, String>> initializeErrorCodes() {
    Map<Integer, Pair<String, String>> descriptionsByErrorCode = new HashMap<>();
    descriptionsByErrorCode.put(LpSolve.UNKNOWNERROR, Pair.of("UNKNOWNERROR", ""));
    descriptionsByErrorCode.put(LpSolve.DATAIGNORED, Pair.of("DATAIGNORED", ""));
    descriptionsByErrorCode.put(LpSolve.NOBFP, Pair.of("NOBFP", ""));
    descriptionsByErrorCode.put(LpSolve.NOMEMORY, Pair.of("NOMEMORY", "Out of memory"));
    descriptionsByErrorCode.put(LpSolve.NOTRUN, Pair.of("NOTRUN", ""));
    descriptionsByErrorCode.put(LpSolve.OPTIMAL,
        Pair.of("OPTIMAL", "An optimal solution was obtained"));
    descriptionsByErrorCode.put(LpSolve.SUBOPTIMAL,
        Pair.of("SUBOPTIMAL", "The model is sub-optimal"));
    descriptionsByErrorCode.put(LpSolve.INFEASIBLE,
        Pair.of("INFEASIBLE", "The model is infeasible"));
    descriptionsByErrorCode.put(LpSolve.UNBOUNDED, Pair.of("UNBOUNDED", "The model is unbounded"));
    descriptionsByErrorCode.put(LpSolve.DEGENERATE,
        Pair.of("DEGENERATE", "The model is degenerative"));
    descriptionsByErrorCode.put(LpSolve.NUMFAILURE,
        Pair.of("NUMFAILURE", "Numerical failure encountered"));
    descriptionsByErrorCode.put(LpSolve.USERABORT,
        Pair.of("USERABORT", "The abort routine returned TRUE"));
    descriptionsByErrorCode.put(LpSolve.TIMEOUT, Pair.of("TIMEOUT", "A timeout occurred"));
    descriptionsByErrorCode.put(LpSolve.RUNNING, Pair.of("RUNNING", ""));
    descriptionsByErrorCode.put(LpSolve.PRESOLVED, Pair.of("PRESOLVED",
        "The model could be solved by presolve. This can only happen if presolve is active via set_presolve"));
    descriptionsByErrorCode.put(LpSolve.PROCFAIL, Pair.of("PROCFAIL", "The B&B routine failed"));
    descriptionsByErrorCode.put(LpSolve.PROCBREAK, Pair.of("PROCBREAK",
        "The B&B was stopped because of a break-at-first or a break-at-value"));
    descriptionsByErrorCode.put(LpSolve.FEASFOUND,
        Pair.of("FEASFOUND", "A feasible B&B solution was found"));
    descriptionsByErrorCode.put(LpSolve.NOFEASFOUND,
        Pair.of("NOFEASFOUND", "No feasible B&B solution found"));
    return descriptionsByErrorCode;
  }

  protected Pair<double[], Double> findMaximizerProbabilities(MatrixWrapper scoreMatrix,
      double minimumScore) {
    // make sure the matrix is positive
    double nonPositiveCompensate = (minimumScore <= 0) ? (1 - minimumScore) : 0.0;

    int numberOfVariables = scoreMatrix.getNumberOfRows();
    int numberOfConstraints = scoreMatrix.getNumberOfColumns();

    double[] objectiveCoefficients = new double[numberOfVariables + 1];
    Arrays.fill(objectiveCoefficients, 1, objectiveCoefficients.length, OBJECTIVE_COEFFICIENT);

    LpSolve solver = null;
    try {
      solver = LpSolve.makeLp(0, numberOfVariables);
      solver.setTimeout((long) TIME_OUT_SECONDS);

      double minInMatrix = Double.POSITIVE_INFINITY;
      solver.setAddRowmode(true);
      for (int constraintIndex = 0; constraintIndex < numberOfConstraints; constraintIndex++) {
        double[] constraint = new double[numberOfVariables + 1];
        for (int variableIndex = 0; variableIndex < numberOfVariables; variableIndex++) {
          double originalScore = scoreMatrix.getValue(variableIndex, constraintIndex);
          minInMatrix = Math.min(minInMatrix, originalScore);

          double score = originalScore + nonPositiveCompensate;
          if (Misc.roughlyEquals(score, 0.0)) {
            continue;
          }
          Assert.isTrue(score > 0, "Score passed to LpSolve should be positive.");
          constraint[variableIndex + 1] = score;
        }
        solver.addConstraint(constraint, LpSolve.GE, RHS_VALUE);
      }
      solver.setAddRowmode(false);

      Assert.isTrue(MathUtils.equals(minInMatrix, minimumScore),
          "minInMatrix=" + minInMatrix + " != minimumScore=" + minimumScore);

      solver.setObjFn(objectiveCoefficients);
      // default lower bound of each variable is 0
      // default upper bound of a variable is infinity

      // solver.printLp();
      solver.setVerbose(LpSolve.CRITICAL);

      int status = solver.solve();
      if (MPGConfig.SHOW_RUNNING_TRACING) {
        System.err.print(".");
      }
      Assert.isTrue(status == LpSolve.OPTIMAL,
          "LpSolve error! status=[" + status + "], description=[" + ERROR_CODES.get(status) + "]");

      double[] xArray = new double[numberOfVariables];
      solver.getVariables(xArray);
      double xSum = 0.0;
      for (int variableIndex = 0; variableIndex < numberOfVariables; variableIndex++) {
        if (xArray[variableIndex] >= 0) {
          xSum += xArray[variableIndex];
        } else {
          xArray[variableIndex] = 0;
          if (MPGConfig.SHOW_RUNNING_TRACING) {
            System.err.print("!");
          }
        }
      }
      for (int index = 0; index < xArray.length; index++) {
        // the probabilities, round to specified value precision
        xArray[index] = Misc.roundValue(xArray[index] / xSum);
      }
      double value = Misc.roundValue(1.0 / xSum - nonPositiveCompensate);
      return Pair.of(xArray, value);
    } catch (LpSolveException e) {
      throw new PurposefulBaseException(e);
    } finally {
      if (solver != null) {
        try {
          solver.deleteLp();
        } catch (Exception e) {
          LOGGER.error("Error when calling solver.deleteLp()", e);
        }
      }
    }
  }
}
