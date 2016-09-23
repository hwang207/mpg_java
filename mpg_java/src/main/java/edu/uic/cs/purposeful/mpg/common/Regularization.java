package edu.uic.cs.purposeful.mpg.common;

public class Regularization {
  private final Norm norm;
  private final double parameter;

  private Regularization(Norm norm, double parameter) {
    this.norm = norm;
    this.parameter = parameter;
  }

  public static Regularization l1(double parameter) {
    return new Regularization(Norm.L1, parameter);
  }

  public static Regularization l2(double parameter) {
    return new Regularization(Norm.L2, parameter);
  }

  public Norm getNorm() {
    return norm;
  }

  public double getParameter() {
    return parameter;
  }

  // public Regularization scale(double scale) {
  // return new Regularization(this.norm, parameter * scale);
  // }

  @Override
  public String toString() {
    return norm.toString() + "_" + parameter;
  }
}
