package edu.uic.cs.purposeful.mpg.common;

import org.apache.commons.io.IOUtils;

import edu.uic.cs.purposeful.common.assertion.Assert;
import gnu.trove.impl.Constants;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import net.mintern.primitive.pair.IntPair;

/**
 * This is a matrix that can grow the size when needed, and it keeps tracking the maximum and
 * minimum score within it.
 */
public class ScoreMatrix {
  private final TObjectDoubleHashMap<IntPair> data;
  private double maximunScore = Double.NEGATIVE_INFINITY;
  private double minimumScore = Double.POSITIVE_INFINITY;
  private int maxRowIndex = -1;
  private int maxColumnIndex = -1;

  public ScoreMatrix() {
    this(Constants.DEFAULT_CAPACITY, 1);
  }

  public ScoreMatrix(int rowCapacity, int columnCapacity) {
    data = new TObjectDoubleHashMap<>(rowCapacity * columnCapacity, Constants.DEFAULT_LOAD_FACTOR,
        Double.NaN);
  }

  public void put(int rowIndex, int columnIndex, double value) {
    Assert.isTrue(rowIndex >= 0);
    Assert.isTrue(columnIndex >= 0);
    Assert.isTrue(Double.isNaN(data.put(IntPair.of(rowIndex, columnIndex), value)),
        "Cell[" + rowIndex + "," + columnIndex + "] already has value.");
    maximunScore = Math.max(maximunScore, value);
    minimumScore = Math.min(minimumScore, value);
    maxRowIndex = Math.max(maxRowIndex, rowIndex);
    maxColumnIndex = Math.max(maxColumnIndex, columnIndex);
  }

  public double get(int rowIndex, int columnIndex) {
    Assert.isTrue(rowIndex >= 0);
    Assert.isTrue(columnIndex >= 0);

    double value = data.get(IntPair.of(rowIndex, columnIndex));
    Assert.isTrue(!Double.isNaN(value),
        "No value stored at position [" + rowIndex + "," + columnIndex + "]");
    return value;
  }

  public double getMaximun() {
    return maximunScore;
  }

  public double getMinimum() {
    return minimumScore;
  }

  public int getRowSize() {
    return maxRowIndex + 1;
  }

  public int getColumnSize() {
    return maxColumnIndex + 1;
  }

  @Override
  public String toString() {
    StringBuilder toString = new StringBuilder();
    for (int rowIndex = 0; rowIndex <= maxRowIndex; rowIndex++) {
      for (int columnIndex = 0; columnIndex <= maxColumnIndex; columnIndex++) {
        toString.append(data.get(IntPair.of(rowIndex, columnIndex))).append(" ");
      }
      toString.append(IOUtils.LINE_SEPARATOR);
    }
    return toString.toString();
  }
}
