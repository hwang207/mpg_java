package edu.uic.cs.purposeful.mpg;

import java.lang.reflect.Field;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.reflect.FieldUtils;

import edu.uic.cs.purposeful.common.assertion.PurposefulBaseException;
import edu.uic.cs.purposeful.common.config.AbstractConfig;
import edu.uic.cs.purposeful.mpg.common.ValuePrecision;

public class MPGConfig extends AbstractConfig {
  private MPGConfig() {
    super("mpg_config.properties");
  }

  public static void overrideConfig(String key, Object value) {
    Field field = FieldUtils.getDeclaredField(MPGConfig.class, StringUtils.upperCase(key));
    FieldUtils.removeFinalModifier(field);
    try {
      FieldUtils.writeStaticField(field, value);
    } catch (IllegalAccessException e) {
      throw new PurposefulBaseException(e);
    }
  }

  private static final MPGConfig INSTANCE = new MPGConfig();

  private static final int _THREAD_POOL_SIZE = INSTANCE.getIntValue("thread_pool_size");
  public static final int THREAD_POOL_SIZE =
      _THREAD_POOL_SIZE > 0 ? _THREAD_POOL_SIZE : Runtime.getRuntime().availableProcessors() + 1;

  public static final ValuePrecision VALUE_PRECISION =
      ValuePrecision.parse(INSTANCE.getDoubleValue("value_precision"));

  public static final String MINIMAX_SOLVER_CLASS = INSTANCE.getStringValue("minimax_solver_class");
  public static final String MINIMAX_SOLVER_CLASS_BACKUP =
      INSTANCE.getStringValue("minimax_solver_class_backup");

  public static final double BIAS_FEATURE_VALUE = INSTANCE.getDoubleValue("bias_feature_value");
  public static final boolean REGULARIZE_BIAS_FEATURE =
      INSTANCE.getBooleanValue("regularize_bias_feature");
  public static final boolean FEATURE_QUADRATIC_EXPANSION =
      INSTANCE.getBooleanValue("feature_quadratic_expansion");
  public static final int MAX_NUM_OF_FEATURES_FOR_QUADRATIC_EXPANSION =
      INSTANCE.getIntValue("max_num_of_features_for_quadratic_expansion");
  public static final boolean LEARN_INITIAL_THETAS =
      INSTANCE.getBooleanValue("learn_initial_thetas");

  public static final double LOGISTIC_REGRESSION_STOPPING_CRITERION =
      INSTANCE.getDoubleValue("logistic_regression_stopping_criterion");

  public static final int MAX_NUM_OF_DOUBLE_ORACLE_PERMUTATIONS =
      INSTANCE.getIntValue("max_num_of_double_oracle_permutations");

  public static final int MAX_DISPLAY_VECTOR_LENGTH =
      INSTANCE.getIntValue("max_display_vector_length");

  public static final String NUMERICAL_OPTIMIZER_IMPLEMENTATION =
      INSTANCE.getStringValue("numerical_optimizer_implementation");

  public static final double LBFGS_TERMINATE_GRADIENT_TOLERANCE =
      INSTANCE.getDoubleValue("lbfgs_terminate_gradient_tolerance");
  public static final double LBFGS_TERMINATE_VALUE_TOLERANCE =
      INSTANCE.getDoubleValue("lbfgs_terminate_value_tolerance");
  public static final int LBFGS_MAX_NUMBER_OF_ITERATIONS =
      INSTANCE.getIntValue("lbfgs_max_number_of_iterations");

  public static final double ADADELTA_DECAY_RATE = INSTANCE.getDoubleValue("adadelta_decay_rate");
  public static final double ADADELTA_SMOOTHING_CONSTANT_ADDEND =
      INSTANCE.getDoubleValue("adadelta_smoothing_constant_addend");
  public static final double ADADELTA_TERMINATE_GRADIENT_TOLERANCE =
      INSTANCE.getDoubleValue("adadelta_terminate_gradient_tolerance");
  public static final boolean ADADELTA_USE_TERMINATE_VALUE =
      INSTANCE.getBooleanValue("adadelta_use_terminate_value");
  public static final double ADADELTA_TERMINATE_VALUE_TOLERANCE =
      INSTANCE.getDoubleValue("adadelta_terminate_value_tolerance");
  public static final int ADADELTA_NUMBER_OF_ITERATIONS =
      INSTANCE.getIntValue("adadelta_number_of_iterations");

  // not final, so that it can be changed
  public static boolean SHOW_RUNNING_TRACING = INSTANCE.getBooleanValue("show_running_tracing");
}
