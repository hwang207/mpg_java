package edu.uic.cs.purposeful.common.assertion;

import java.util.Collection;
import java.util.Map;

public class Assert {
  private static final String EXCEPTION_MESSAGE_NOT_NULL = "Parameter can not be null. ";
  private static final String EXCEPTION_MESSAGE_IS_TRUE = "Condition must be [true]. ";
  private static final String EXCEPTION_MESSAGE_IS_FALSE = "Condition must be [false]. ";
  private static final String EXCEPTION_MESSAGE_IS_NULL = "Parameter must be null. ";
  private static final String EXCEPTION_MESSAGE_NOT_EMPTY = "Parameter can not be empty. ";
  private static final String EXCEPTION_MESSAGE_CAN_NEVER_HAPPEN = "Can never happen. ";

  public static void isNull(Object obj) {
    isNull(obj, EXCEPTION_MESSAGE_IS_NULL);
  }

  public static void isNull(Object obj, String message) {
    if (obj != null) {
      throw new PurposefulBaseException(message);
    }
  }

  public static void notNull(Object obj) {
    notNull(obj, null);
  }

  public static void notNull(Object obj, String message) {
    if (obj == null) {
      throw (message == null) ? new PurposefulBaseException(EXCEPTION_MESSAGE_NOT_NULL)
          : new PurposefulBaseException(EXCEPTION_MESSAGE_NOT_NULL + "[Message]" + message);
    }
  }

  public static void notEmpty(Object obj) {
    notEmpty(obj, null);
  }

  public static void notEmpty(Object obj, String message) {
    notNull(obj, message);
    boolean isEmptyString = obj instanceof String && ((String) obj).trim().length() == 0;
    boolean isEmptyCollection = obj instanceof Collection<?> && ((Collection<?>) obj).isEmpty();
    boolean isEmptyMap = obj instanceof Map<?, ?> && ((Map<?, ?>) obj).isEmpty();
    if (isEmptyString || isEmptyCollection || isEmptyMap) {
      throw (message == null) ? new PurposefulBaseException(EXCEPTION_MESSAGE_NOT_EMPTY)
          : new PurposefulBaseException(EXCEPTION_MESSAGE_NOT_EMPTY + "[Message]" + message);
    }
  }

  public static void isTrue(boolean condition) {
    isTrue(condition, null);
  }

  public static void isTrue(boolean condition, String message) {
    if (!condition) {
      throw (message == null) ? new PurposefulBaseException(EXCEPTION_MESSAGE_IS_TRUE)
          : new PurposefulBaseException(EXCEPTION_MESSAGE_IS_TRUE + "[Message]" + message);
    }
  }

  public static void isFalse(boolean condition) {
    isFalse(condition, null);
  }

  public static void isFalse(boolean condition, String message) {
    if (condition) {
      throw (message == null) ? new PurposefulBaseException(EXCEPTION_MESSAGE_IS_FALSE)
          : new PurposefulBaseException(EXCEPTION_MESSAGE_IS_FALSE + "[Message]" + message);
    }
  }

  public static void canNeverHappen() {
    canNeverHappen(null);
  }

  public static void canNeverHappen(String message) {
    throw (message == null) ? new PurposefulBaseException(EXCEPTION_MESSAGE_CAN_NEVER_HAPPEN)
        : new PurposefulBaseException(EXCEPTION_MESSAGE_CAN_NEVER_HAPPEN + "[Message]" + message);
  }
}
