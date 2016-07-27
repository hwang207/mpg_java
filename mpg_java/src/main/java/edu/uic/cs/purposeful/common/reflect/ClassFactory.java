package edu.uic.cs.purposeful.common.reflect;

import org.apache.commons.lang3.ClassUtils;
import org.apache.commons.lang3.reflect.ConstructorUtils;

import edu.uic.cs.purposeful.common.assertion.PurposefulBaseException;

public abstract class ClassFactory {

  @SuppressWarnings("unchecked")
  public static <T> T getInstance(String className, Object... args) {
    try {
      Class<?> clazz = ClassUtils.getClass(className);
      Object instance = ConstructorUtils.invokeConstructor(clazz, args);
      return (T) instance;
    } catch (Exception e) {
      throw new PurposefulBaseException(e);
    }
  }
}
