/**
 * Copyright (c) 2015 University of Illinois at Chicago to Present. All rights reserved.
 */
package edu.uic.cs.purposeful.mpg.common;

public class FeatureWiseRegularization {
  private final Norm norm;
  private final double[] parameters;

  private FeatureWiseRegularization(Norm norm, double[] parameters) {
    this.norm = norm;
    this.parameters = parameters;
  }

  public static FeatureWiseRegularization l1(double[] parameters) {
    return new FeatureWiseRegularization(Norm.L1, parameters);
  }

  public static FeatureWiseRegularization l2(double[] parameters) {
    return new FeatureWiseRegularization(Norm.L2, parameters);
  }

  public Norm getNorm() {
    return norm;
  }

  public double[] getParameters() {
    return parameters;
  }

  @Override
  public String toString() {
    return norm.toString() + "_" + Misc.toDisplay(parameters);
  }
}
