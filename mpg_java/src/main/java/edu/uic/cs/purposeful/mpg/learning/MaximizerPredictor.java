package edu.uic.cs.purposeful.mpg.learning;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.io.FileUtils;

import com.google.common.collect.Iterables;
import com.google.common.primitives.Doubles;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.common.assertion.PurposefulBaseException;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.optimizer.game.ZeroSumGameSolver;
import edu.uic.cs.purposeful.mpg.optimizer.game.impl.DoubleOracleGameSolver;
import edu.uic.cs.purposeful.mpg.target.OptimizationTarget;

public class MaximizerPredictor<Permutation, InitialData> {
  public static class Prediction<Permutation> {
    private final int index;
    private final Permutation predictionPermutation;
    private final double probability;
    private final Permutation goldenPermutation;
    private final double score;

    public Prediction(int index, Permutation predictionPermutation, double probability,
        Permutation goldenPermutation, double score) {
      this.index = index;
      this.predictionPermutation = predictionPermutation;
      this.probability = probability;
      this.goldenPermutation = goldenPermutation;
      this.score = score;
    }

    public Permutation getGoldenPermutation() {
      return goldenPermutation;
    }

    public Permutation getPredictionPermutation() {
      return predictionPermutation;
    }

    public double getProbability() {
      return probability;
    }

    public double getScore() {
      return score;
    }

    public int getIndex() {
      return index;
    }

    @Override
    public String toString() {
      return score + "(" + probability + "%)";
    }
  }

  private static class PredictionTask<Permutation, InitialData>
      implements Callable<Prediction<Permutation>> {
    private final int index;
    private final InitialData initialData;
    private final double[] thetas;
    private final OptimizationTarget<Permutation, InitialData> optimizationTarget;

    private PredictionTask(int index, InitialData initialData, double[] thetas,
        Class<? extends OptimizationTarget<Permutation, InitialData>> optimizationTargetClass) {
      try {
        this.optimizationTarget = optimizationTargetClass.newInstance();
      } catch (Exception e) {
        throw new PurposefulBaseException(e);
      }
      this.index = index;
      this.initialData = initialData;
      this.thetas = thetas;
    }

    @Override
    public Prediction<Permutation> call() throws Exception {
      if (MPGConfig.SHOW_RUNNING_TRACING) {
        System.err.print("[" + Thread.currentThread().getId() + "]");
      }

      optimizationTarget.initialize(initialData, false);
      ZeroSumGameSolver<Permutation> gameSolver = new DoubleOracleGameSolver<>(optimizationTarget);
      gameSolver.solve(thetas);

      LinkedHashSet<Permutation> maximizerPermutations = gameSolver.getMaximizerPermutations();
      double[] maximizerProbabilities = gameSolver.getMaximizerProbabilities();
      Assert.isTrue(maximizerPermutations.size() == maximizerProbabilities.length,
          maximizerPermutations.size() + " != " + maximizerProbabilities.length);

      Permutation maxProbPermutation = null;
      double maxProbability = Double.NEGATIVE_INFINITY;
      int permutationIndex = 0;
      for (Permutation maximizerPermutation : maximizerPermutations) {
        double maximizerProbability = maximizerProbabilities[permutationIndex++];
        if (maximizerProbability > maxProbability) {
          maxProbability = maximizerProbability;
          maxProbPermutation = maximizerPermutation;
        }
      }

      double score = optimizationTarget.computeScore(maxProbPermutation,
          optimizationTarget.getGoldenPermutation());
      Assert.isFalse(Double.isNaN(score),
          optimizationTarget.getClass().getName() + ".computeScore() should not return NaN.");
      return new Prediction<Permutation>(index, maxProbPermutation, maxProbability,
          optimizationTarget.getGoldenPermutation(), score);
    }
  }

  protected final Class<? extends OptimizationTarget<Permutation, InitialData>> optimizationTargetClass;

  protected MaximizerPredictor(
      Class<? extends OptimizationTarget<Permutation, InitialData>> optimizationTargetClass) {
    this.optimizationTargetClass = optimizationTargetClass;
  }

  protected List<Prediction<Permutation>> predict(List<InitialData> initialDataList,
      double[] thetas) {
    ExecutorService threadPool = Executors.newFixedThreadPool(MPGConfig.THREAD_POOL_SIZE);
    CompletionService<Prediction<Permutation>> completionService =
        new ExecutorCompletionService<>(threadPool);

    try {
      int index = 0;
      for (InitialData initialData : initialDataList) {
        completionService
            .submit(new PredictionTask<>(index++, initialData, thetas, optimizationTargetClass));
      }

      List<Prediction<Permutation>> result =
          new ArrayList<>(Collections.nCopies(initialDataList.size(), null));
      for (int targetIndex = 0; targetIndex < initialDataList.size(); targetIndex++) {
        Prediction<Permutation> prediction = completionService.take().get();
        Assert.isNull(result.set(prediction.index, prediction));
      }
      return result;
    } catch (Exception e) {
      throw new PurposefulBaseException(e);
    } finally {
      threadPool.shutdown();
    }
  }

  protected Prediction<Permutation> predict(InitialData initialData, double[] thetas) {
    return Iterables.getOnlyElement(predict(Collections.singletonList(initialData), thetas));
  }

  protected void writeModelToFile(File modelFile, double[] thetas) {
    try {
      FileUtils.writeLines(modelFile, Doubles.asList(thetas));
    } catch (IOException e) {
      throw new PurposefulBaseException(e);
    }
  }

  protected double[] loadModelFromFile(File modelFile) {
    List<String> lines = null;
    try {
      lines = FileUtils.readLines(modelFile);
    } catch (IOException e) {
      throw new PurposefulBaseException(e);
    }
    double[] thetas = new double[lines.size()];
    int index = 0;
    for (String line : lines) {
      thetas[index++] = Double.parseDouble(line);
    }
    return thetas;
  }
}
