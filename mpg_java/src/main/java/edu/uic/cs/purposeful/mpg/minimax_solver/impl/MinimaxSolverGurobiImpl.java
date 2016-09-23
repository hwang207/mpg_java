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
import gurobi.GRB;
import gurobi.GRBEnv;
import gurobi.GRBException;
import gurobi.GRBLinExpr;
import gurobi.GRBModel;
import gurobi.GRBVar;

public class MinimaxSolverGurobiImpl extends MinimaxSolver {
  private static final Logger LOGGER = Logger.getLogger(MinimaxSolverGurobiImpl.class);

  private static final GRBEnv ENV = createGRBEnv();

  private static final Map<Integer, Pair<String, String>> ERROR_CODES = initializeErrorCodes();

  private static GRBEnv createGRBEnv() {
    GRBEnv env = null;
    try {
      env = new GRBEnv();
      env.set(GRB.DoubleParam.TimeLimit, TIME_OUT_SECONDS);
      env.set(GRB.IntParam.OutputFlag, 0); // no Gurobi output
    } catch (GRBException e) {
      throw new PurposefulBaseException(e);
    }
    return env;
  }

  private static Map<Integer, Pair<String, String>> initializeErrorCodes() {
    Map<Integer, Pair<String, String>> descriptionsByErrorCode = new HashMap<>();
    descriptionsByErrorCode.put(1,
        Pair.of("LOADED", "Model is loaded, but no solution information is available."));
    descriptionsByErrorCode.put(2,
        Pair.of("OPTIMAL", "Model was solved to optimality (subject to tolerances), "
            + "and an optimal solution is available."));
    descriptionsByErrorCode.put(3, Pair.of("INFEASIBLE", "Model was proven to be infeasible."));
    descriptionsByErrorCode.put(4,
        Pair.of("INF_OR_UNBD",
            "Model was proven to be either infeasible or unbounded. "
                + "To obtain a more definitive conclusion, "
                + "set the DualReductions parameter to 0 and reoptimize."));
    descriptionsByErrorCode.put(5,
        Pair.of("UNBOUNDED",
            "Model was proven to be unbounded. Important note: "
                + "an unbounded status indicates the presence of an unbounded ray "
                + "that allows the objective to improve without limit. "
                + "It says nothing about whether the model has a feasible solution. "
                + "If you require information on feasibility, "
                + "you should set the objective to zero and reoptimize."));
    descriptionsByErrorCode.put(6,
        Pair.of("CUTOFF", "Optimal objective for model was proven to be worse than the value "
            + "specified in the Cutoff parameter. No solution information is available."));
    descriptionsByErrorCode.put(7,
        Pair.of("ITERATION_LIMIT",
            "Optimization terminated because the total number of simplex iterations "
                + "performed exceeded the value specified in the IterationLimit parameter, "
                + "or because the total number of barrier iterations exceeded the value "
                + "specified in the BarIterLimit parameter."));
    descriptionsByErrorCode.put(8,
        Pair.of("NODE_LIMIT", "Optimization terminated because the total number of branch-and-cut "
            + "nodes explored exceeded the value specified in the NodeLimit parameter."));
    descriptionsByErrorCode.put(9,
        Pair.of("TIME_LIMIT", "Optimization terminated because the time expended exceeded the "
            + "value specified in the TimeLimit parameter."));
    descriptionsByErrorCode.put(10,
        Pair.of("SOLUTION_LIMIT",
            "Optimization terminated because the number of solutions found reached "
                + "the value specified in the SolutionLimit parameter."));
    descriptionsByErrorCode.put(11,
        Pair.of("INTERRUPTED", "Optimization was terminated by the user."));
    descriptionsByErrorCode.put(12, Pair.of("NUMERIC",
        "Optimization was terminated due to unrecoverable numerical difficulties."));
    descriptionsByErrorCode.put(13, Pair.of("SUBOPTIMAL",
        "Unable to satisfy optimality tolerances; a sub-optimal solution is available."));
    descriptionsByErrorCode.put(14,
        Pair.of("INPROGRESS", "An asynchronous optimization call was made, "
            + "but the associated optimization run is not yet complete."));
    return descriptionsByErrorCode;
  }

  @Override
  protected Pair<double[], Double> findMaximizerProbabilities(MatrixWrapper scoreMatrix,
      double minimumScore, double maximumScore) {
    GRBModel model = null;
    try {
      // make sure the matrix is positive
      double nonPositiveCompensate = (minimumScore <= 0) ? (1 - minimumScore) : 0.0;
      double compensatedMaximumScore = maximumScore + nonPositiveCompensate;

      model = new GRBModel(ENV);

      int numberOfRows = scoreMatrix.getNumberOfRows();
      int numberOfColumns = scoreMatrix.getNumberOfColumns();

      double[] objectiveCoefficients = new double[numberOfRows];
      Arrays.fill(objectiveCoefficients, OBJECTIVE_COEFFICIENT);

      GRBVar[] variables = model.addVars(/* default lower bound=0.0 */null,
          /* default upper bound=infinite */null, objectiveCoefficients,
          /* default continuous variables */null, /* all variables are given default names */null);
      model.update();

      // add matrix A
      double minInMatrix = Double.POSITIVE_INFINITY;
      for (int columnIndex = 0; columnIndex < numberOfColumns; columnIndex++) {
        GRBLinExpr lhsExpression = new GRBLinExpr();
        for (int rowIndex = 0; rowIndex < numberOfRows; rowIndex++) {
          double originalScore = scoreMatrix.getValue(rowIndex, columnIndex);
          minInMatrix = Math.min(minInMatrix, originalScore);

          // normalize score to ensure they are not too large
          double score = (originalScore + nonPositiveCompensate) / compensatedMaximumScore;
          if (Misc.roughlyEquals(score, 0.0)) {
            continue;
          }
          Assert.isTrue(score > 0, "Score passed to Gurobi solver should be positive.");
          lhsExpression.addTerm(score, variables[rowIndex]);
        }
        // TODO consider using multi-thread?
        model.addConstr(lhsExpression, GRB.GREATER_EQUAL, RHS_VALUE, /* constraint name */
            String.valueOf(columnIndex));
      }
      Assert.isTrue(MathUtils.equals(minInMatrix, minimumScore),
          "minInMatrix=" + minInMatrix + " != minimumScore=" + minimumScore);

      // add objective
      GRBLinExpr objective = new GRBLinExpr();
      for (int rowIndex = 0; rowIndex < numberOfRows; rowIndex++) {
        objective.addTerm(OBJECTIVE_COEFFICIENT, variables[rowIndex]);
      }
      model.setObjective(objective);

      model.optimize();
      if (MPGConfig.SHOW_RUNNING_TRACING) {
        System.err.print("*");
      }

      int status = model.get(GRB.IntAttr.Status);
      Assert.isTrue(status == GRB.Status.OPTIMAL,
          "Gurobi error! status=[" + status + "], description=[" + ERROR_CODES.get(status) + "]");

      double[] xArray = new double[numberOfRows];
      double xSum = 0.0;
      for (int rowIndex = 0; rowIndex < numberOfRows; rowIndex++) {
        double x = variables[rowIndex].get(GRB.DoubleAttr.X);
        if (x >= 0) {
          xArray[rowIndex] = x;
          xSum += x;
        } else {
          if (MPGConfig.SHOW_RUNNING_TRACING) {
            System.err.print("!");
          }
        }
      }

      for (int index = 0; index < xArray.length; index++) {
        // the probabilities, round to specified value precision
        xArray[index] = Misc.roundValue(xArray[index] / xSum);
        Assert.isFalse(Double.isNaN(xArray[index]));
      }

      double value = Misc.roundValue(compensatedMaximumScore / xSum - nonPositiveCompensate);
      Assert.isFalse(Double.isNaN(value));
      return Pair.of(xArray, value);
    } catch (GRBException e) {
      throw new PurposefulBaseException(e);
    } finally {
      if (model != null) {
        try {
          model.dispose();
        } catch (Exception e) {
          LOGGER.error("Error when calling model.dispose()", e);
        }
      }
    }
  }
}
