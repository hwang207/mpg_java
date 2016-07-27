package edu.uic.cs.purposeful.mpg.optimizer.numerical.adadelta.jhu;

import java.util.LinkedList;

import org.apache.commons.lang3.mutable.MutableDouble;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.AdaDelta;
import edu.jhu.hlt.optimize.AdaDelta.AdaDeltaPrm;
import edu.jhu.hlt.optimize.BatchSampler;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.util.Lambda.FnIntDoubleToDouble;
import edu.jhu.prim.util.Lambda.FnIntDoubleToVoid;
import edu.jhu.prim.vector.IntDoubleVector;
import edu.jhu.util.Timer;
import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.IterationCallback;

/**
 * Stochastic gradient descent with minibatches.
 * 
 * @author mgormley
 */
public class AdaDeltaSGDLite {
  private static final Logger LOGGER = Logger.getLogger(AdaDeltaSGDLite.class);

  private class Record {
    private static final int MAX_SIZE = 100;
    private double gNormLast;
    private IntDoubleVector xLast;
    private final LinkedList<Double> values = new LinkedList<>();

    private void start(IntDoubleVector gradient, IntDoubleVector point) {
      gNormLast = Math.sqrt(gradient.getL2Norm());
      xLast = point;
      values.clear();
    }

    private void add(double value, IntDoubleVector gradient, IntDoubleVector point) {
      if (values.size() > MAX_SIZE) {
        values.pollFirst();
      }
      values.addLast(value);

      add(gradient, point);
    }

    private void add(IntDoubleVector gradient, IntDoubleVector point) {
      gNormLast = Math.sqrt(gradient.getL2Norm());
      xLast = point;
    }

    private boolean doStop() {
      if (MPGConfig.ADADELTA_USE_TERMINATE_VALUE) {
        int size = values.size();
        double newestVal = values.get(size - 1);
        double previousVal = (size >= 10 ? values.get(size - 10) : values.get(0));
        double averageImprovement = (previousVal - newestVal) / (size >= 10 ? 10 : size);

        if (size > 5 && Math
            .abs(averageImprovement / newestVal) < MPGConfig.ADADELTA_TERMINATE_VALUE_TOLERANCE) {
          LOGGER.warn("AdaDelta terminates: value is no longer improved.");
          return true;
        }
      }

      // Checks if the gradient is sufficiently small compared to x that it is treated as zero.
      // First do the one norm, because that's easiest, and always bigger.
      if (gNormLast < MPGConfig.ADADELTA_TERMINATE_GRADIENT_TOLERANCE
          * Math.max(1.0, norm1(xLast))) {
        // Now actually compare with the two norm if we have to.
        if (gNormLast < MPGConfig.ADADELTA_TERMINATE_GRADIENT_TOLERANCE
            * Math.max(1.0, Math.sqrt(xLast.getL2Norm()))) {
          LOGGER.warn("AdaDelta terminates: gradients are no longer improved.");
          return true;
        }
      }

      return false;
    }

    private double norm1(IntDoubleVector values) {
      final edu.jhu.prim.Primitives.MutableDouble sum =
          new edu.jhu.prim.Primitives.MutableDouble(0);
      values.iterate(new FnIntDoubleToVoid() {
        public void call(int index, double value) {
          sum.v += Math.abs(value);
        }
      });
      return sum.v;
    }
  }

  /** The number of gradient steps to run. */
  private int iterations;
  private BatchSampler batchSampler;
  private int batchSize;
  private AdaDelta adaDelta;

  private void initialize(DifferentiableBatchFunction function) {
    int numExamples = function.getNumExamples();
    batchSize = Math.min(numExamples, MPGConfig.THREAD_POOL_SIZE);
    Assert.isTrue(batchSize > 0);

    batchSampler = new BatchSampler(MPGConfig.ADADELTA_SAMPLE_BATCHES_WITH_REPLACEMENT, numExamples,
        batchSize);

    iterations = (int) Math.ceil(MPGConfig.ADADELTA_NUMBER_OF_ITERATIONS * numExamples / batchSize);
    Assert.isTrue(iterations > 0);
    if (LOGGER.isInfoEnabled()) {
      LOGGER.info("Setting number of batch gradient steps: " + iterations);
    }

    AdaDeltaPrm adaDeltaPrm = new AdaDeltaPrm();
    adaDeltaPrm.decayRate = MPGConfig.ADADELTA_DECAY_RATE;
    adaDeltaPrm.constantAddend = MPGConfig.ADADELTA_SMOOTHING_CONSTANT_ADDEND;
    adaDelta = new AdaDelta(adaDeltaPrm);
    adaDelta.init(function);
  }

  /**
   * Maximize the function starting at the given initial point.
   */
  public boolean maximize(DifferentiableBatchFunction function, IntDoubleVector point,
      IterationCallback iterationCallback) {
    return optimize(function, point, true, iterationCallback);
  }

  /**
   * Minimize the function starting at the given initial point.
   */
  public boolean minimize(DifferentiableBatchFunction function, IntDoubleVector point,
      IterationCallback iterationCallback) {
    return optimize(function, point, false, iterationCallback);
  }

  private boolean optimize(DifferentiableBatchFunction function, final IntDoubleVector point,
      final boolean maximize, IterationCallback iterationCallback) {
    initialize(function);

    int passCount = 0;
    double passCountFrac = 0;
    int iterCount = 0; // number of iterations performed thus far

    if (MPGConfig.ADADELTA_USE_TERMINATE_VALUE && LOGGER.isDebugEnabled()) {
      double value = function.getValue(point);
      LOGGER.debug(
          String.format("Function value on all examples = %g at iteration = %d on pass = %.2f",
              value, iterCount, passCountFrac));
    }

    Assert.isTrue(function.getNumDimensions() >= point.getNumImplicitEntries());

    Timer passTimer = new Timer();
    passTimer.start();
    Record record = null;

    for (; iterCount < iterations; iterCount++) {
      int[] batch = batchSampler.sampleBatch();

      IntDoubleVector gradient;
      if (MPGConfig.ADADELTA_USE_TERMINATE_VALUE && LOGGER.isTraceEnabled()) {
        ValueGradient vg = function.getValueGradient(point, batch);
        gradient = vg.getGradient();
        LOGGER.trace(String.format("Function value on batch = %g at iteration = %d", vg.getValue(),
            iterCount));
      } else {
        gradient = function.getGradient(point, batch);
      }

      if (record == null) {
        record = new Record();
        record.start(gradient, point);
      }
      adaDelta.takeNoteOfGradient(gradient);

      // Scale the gradient by the parameter-specific learning rate.
      final int _iterCount = iterCount;
      gradient.apply(new FnIntDoubleToDouble() {
        @Override
        public double call(int index, double value) {
          double lr = adaDelta.getLearningRate(_iterCount, index);
          if (maximize) {
            value = lr * value;
          } else {
            value = -lr * value;
          }
          Assert.isFalse(Double.isNaN(value));
          Assert.isFalse(Double.isInfinite(value));
          return value;
        }
      });

      // Take a step in the direction of the gradient.
      point.add(gradient);

      if (LOGGER.isTraceEnabled()) {
        LOGGER.trace(String.format("min=%g max=%g infnorm=%g l2=%g", point.getMin(), point.getMax(),
            point.getInfNorm(), point.getL2Norm()));
      }

      int nextIterCount = iterCount + 1;
      passCountFrac = ((double) nextIterCount) * batchSize / function.getNumExamples();
      boolean completedPass = (int) Math.floor(passCountFrac) > passCount;

      // Another full pass through the data has been completed or we're on the last iteration.
      if (completedPass || nextIterCount == iterations) {
        if (MPGConfig.ADADELTA_USE_TERMINATE_VALUE) {
          // Report the value of the function on all the examples.
          double value = function.getValue(point);
          record.add(value, gradient, point);
          if (LOGGER.isDebugEnabled()) {
            LOGGER.debug(String.format(
                "Function value on all examples = %g at iteration = %d on pass = %.2f", value,
                nextIterCount, passCountFrac));
          }
        } else {
          record.add(gradient, point);
        }

        if (LOGGER.isDebugEnabled()) {
          logAvgLrAndStepSize(iterCount, point, gradient);
          LOGGER.debug(String.format("Average time per pass (min): %.2g",
              passTimer.totSec() / 60.0 / passCountFrac));
        }
      }

      if (completedPass) {
        if (iterationCallback != null) {
          try {
            iterationCallback.call(passCount, point.toNativeArray());
          } catch (Exception e) {
            LOGGER.error("", e);
          }
        }

        if (record.doStop()) {
          return true;
        }

        // Another full pass through the data has been completed.
        passCount++;
      }
    }

    return false;
  }

  private void logAvgLrAndStepSize(int iterCount, IntDoubleVector point, IntDoubleVector gradient) {
    // Compute the average learning rate and the average step size.
    final MutableDouble avgLr = new MutableDouble(0.0);
    final MutableDouble avgStep = new MutableDouble(0d);
    final MutableInt numNonZeros = new MutableInt(0);
    gradient.apply(new FnIntDoubleToDouble() {
      @Override
      public double call(int index, double value) {
        double lr = adaDelta.getLearningRate(iterCount, index);
        Assert.isFalse(Double.isNaN(point.get(index)));
        if (value != 0.0) {
          avgLr.add(lr);
          avgStep.add(gradient.get(index));
          numNonZeros.increment();
        }
        return value;
      }
    });
    avgLr.setValue(avgLr.doubleValue() / numNonZeros.doubleValue());
    avgStep.setValue(avgStep.doubleValue() / numNonZeros.doubleValue());
    if (numNonZeros.doubleValue() == 0) {
      avgLr.setValue(0.0);
      avgStep.setValue(0.0);
    }
    LOGGER.debug("Average learning rate: " + avgLr);
    LOGGER.debug("Average step size: " + avgStep);
  }
}
