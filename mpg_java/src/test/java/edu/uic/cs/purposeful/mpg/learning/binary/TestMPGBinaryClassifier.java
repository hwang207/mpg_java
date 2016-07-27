package edu.uic.cs.purposeful.mpg.learning.binary;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

import org.apache.commons.lang3.reflect.FieldUtils;
import org.junit.Assert;
import org.junit.Test;

import de.bwaldvogel.liblinear.Problem;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.ValuePrecision;

public class TestMPGBinaryClassifier extends Assert {

  @Test
  public void testAlignDataset() throws Exception {
    Problem data1 = new Problem();
    data1.n = 9;
    double[] thetas1 = new double[] {1, 2, 3, 4, 5};
    setStaticFinal(MPGConfig.class, "BIAS_FEATURE_VALUE", 1.0); // >=0 means has bias
    double[] actual1 = new MPGBinaryClassifier(null, 0).alignDataset(data1, thetas1);
    double[] expected1 = new double[] {1, 2, 3, 4, 0, 0, 0, 0, 5};
    assertArrayEquals(expected1, actual1, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    Problem data2 = new Problem();
    data2.n = 9;
    double[] thetas2 = new double[] {1, 2, 3, 4, 5};
    setStaticFinal(MPGConfig.class, "BIAS_FEATURE_VALUE", -1.0); // <0 means has no bias
    double[] actual2 = new MPGBinaryClassifier(null, 0).alignDataset(data2, thetas2);
    double[] expected2 = new double[] {1, 2, 3, 4, 5, 0, 0, 0, 0};
    assertArrayEquals(expected2, actual2, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    Problem data3 = new Problem();
    data3.n = 6;
    double[] thetas3 = new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    setStaticFinal(MPGConfig.class, "BIAS_FEATURE_VALUE", 1.0); // >=0 means has bias
    double[] actual3 = new MPGBinaryClassifier(null, 0).alignDataset(data3, thetas3);
    double[] expected3 = new double[] {10, 9, 8, 7, 6, 1};
    assertArrayEquals(expected3, actual3, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());

    Problem data4 = new Problem();
    data4.n = 6;
    double[] thetas4 = new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    setStaticFinal(MPGConfig.class, "BIAS_FEATURE_VALUE", -1.0); // <0 means has no bias
    double[] actual4 = new MPGBinaryClassifier(null, 0).alignDataset(data4, thetas4);
    double[] expected4 = new double[] {10, 9, 8, 7, 6, 5};
    assertArrayEquals(expected4, actual4, ValuePrecision.POINT_6_ZEROS_ONE.getValuePrecision());
  }

  private void setStaticFinal(Class<?> cls, String fieldName, Object newValue) throws Exception {
    Field field = FieldUtils.getField(cls, fieldName);
    field.setAccessible(true);
    Field modifiersField = Field.class.getDeclaredField("modifiers");
    modifiersField.setAccessible(true);
    modifiersField.setInt(field, field.getModifiers() & ~Modifier.FINAL);
    field.set(null, newValue);
  }
}
