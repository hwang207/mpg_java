package edu.uic.cs.purposeful.common.assertion;

public class PurposefulBaseException extends RuntimeException {
  private static final long serialVersionUID = 1L;

  public PurposefulBaseException() {
    super();
  }

  public PurposefulBaseException(String message, Throwable cause) {
    super(message, cause);
  }

  public PurposefulBaseException(String message) {
    super(message);
  }

  public PurposefulBaseException(Throwable cause) {
    super(cause);
  }
}
