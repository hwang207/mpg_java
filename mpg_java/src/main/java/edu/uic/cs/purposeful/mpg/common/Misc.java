package edu.uic.cs.purposeful.mpg.common;

import java.util.Arrays;

import org.apache.commons.math3.util.MathUtils;

import edu.uic.cs.purposeful.mpg.MPGConfig;

public class Misc {
  public static String toDisplay(double[] vector) {
    if (vector.length > MPGConfig.MAX_DISPLAY_VECTOR_LENGTH) {
      return Arrays.toString(Arrays.copyOfRange(vector, 0, MPGConfig.MAX_DISPLAY_VECTOR_LENGTH))
          .replace("]", ", ... (total=" + vector.length + ")]");
    }
    return Arrays.toString(vector);
  }

  public static boolean roughlyEquals(double x, double y) {
    if (Double.isNaN(x) && Double.isNaN(y)) {
      return true;
    }
    if (Double.isNaN(x) || Double.isNaN(y)) {
      return false;
    }
    return MathUtils.equals(MPGConfig.VALUE_PRECISION.roundToValuePrecision(x),
        MPGConfig.VALUE_PRECISION.roundToValuePrecision(y));
  }

  public static double roundValue(double x) {
    return MPGConfig.VALUE_PRECISION.roundToValuePrecision(x);
  }

  public static String byteCountToHumanReadableSize(long bytes, boolean si) {
    int unit = si ? 1000 : 1024;
    if (bytes < unit)
      return bytes + " B";
    int exp = (int) (Math.log(bytes) / Math.log(unit));
    String pre = (si ? "kMGTPE" : "KMGTPE").charAt(exp - 1) + (si ? "" : "i");
    return String.format("%.1f %sB", bytes / Math.pow(unit, exp), pre);
  }
}
