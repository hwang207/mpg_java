package edu.uic.cs.purposeful.mpg.common;

import java.util.HashMap;
import java.util.Map;

import edu.uic.cs.purposeful.common.assertion.Assert;

public enum ValuePrecision {
  ONE(1), POINT_ONE(.1), POINT_ZERO_ONE(.01), POINT_2_ZEROS_ONE(.001), POINT_3_ZEROS_ONE(
      .0001), POINT_4_ZEROS_ONE(.00001), POINT_5_ZEROS_ONE(.000001), POINT_6_ZEROS_ONE(
          .0000001), POINT_7_ZEROS_ONE(.00000001), POINT_8_ZEROS_ONE(
              .000000001), POINT_9_ZEROS_ONE(.0000000001), POINT_10_ZEROS_ONE(.00000000001);

  private final double valuePrecision;
  private final double valuePrecisionModifier;

  private static final Map<Double, ValuePrecision> VALUE_PRECISIONS_BY_VALUE = initialize();

  private ValuePrecision(double valuePrecision) {
    this.valuePrecision = valuePrecision;
    this.valuePrecisionModifier = Math.round(1 / valuePrecision);
  }

  private static Map<Double, ValuePrecision> initialize() {
    Map<Double, ValuePrecision> result = new HashMap<>();
    for (ValuePrecision valuePrecision : values()) {
      result.put(valuePrecision.getValuePrecision(), valuePrecision);
    }
    return result;
  }

  public double getValuePrecision() {
    return valuePrecision;
  }

  public double getValuePrecisionModifier() {
    return valuePrecisionModifier;
  }

  public double roundToValuePrecision(double value) {
    return Math.round(value * valuePrecisionModifier) / valuePrecisionModifier;
  }

  public static ValuePrecision parse(double value) {
    ValuePrecision valuePrecision = VALUE_PRECISIONS_BY_VALUE.get(value);
    Assert.notNull(valuePrecision,
        "ValuePrecision[" + value + "] is not any one of " + VALUE_PRECISIONS_BY_VALUE.keySet());
    return valuePrecision;
  }
}
