package edu.uic.cs.purposeful.common.config;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.commons.io.IOUtils;
import org.apache.log4j.Logger;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.common.assertion.PurposefulBaseException;

abstract public class AbstractConfig {

  private static final Logger LOGGER = Logger.getLogger(AbstractConfig.class);

  private static final Set<Class<? extends AbstractConfig>> CLASSES_LOADED =
      new HashSet<Class<? extends AbstractConfig>>();

  private static final Map<String, String> CONFIGURATIONS = new HashMap<String, String>();

  protected AbstractConfig(String configFileName) {

    Class<? extends AbstractConfig> clazz = this.getClass();
    if (!CLASSES_LOADED.contains(clazz)) {
      synchronized (AbstractConfig.class) {
        if (!CLASSES_LOADED.contains(clazz)) {
          loadConfigurations(clazz, configFileName);
          CLASSES_LOADED.add(clazz);
        }
      }
    }
  }

  private void loadConfigurations(Class<? extends AbstractConfig> clazz, String configFileName) {

    InputStream inStream = null;
    try {
      Properties properties = new Properties();
      inStream = clazz.getResourceAsStream(configFileName);
      properties.load(inStream);

      for (Entry<Object, Object> entry : properties.entrySet()) {
        String key = entry.getKey().toString();
        String value = entry.getValue().toString();

        Assert.isNull(CONFIGURATIONS.put(key, value),
            "Duplicate key [" + key + "] from configuration file [" + configFileName + "]");
      }
    } catch (Exception e) {
      throw new PurposefulBaseException(e);
    } finally {
      IOUtils.closeQuietly(inStream);
    }

    // load the extension configurations
    String extensionConfigFileName = configFileName + ".extension";
    InputStream extensionInStream = null;
    try {
      extensionInStream = clazz.getResourceAsStream(extensionConfigFileName);
      if (extensionInStream != null) {
        Properties properties = new Properties();
        properties.load(extensionInStream);

        for (Entry<Object, Object> entry : properties.entrySet()) {
          String key = entry.getKey().toString();
          String value = entry.getValue().toString();
          String originalValue = CONFIGURATIONS.put(key, value);

          if (originalValue == null) {
            continue;
          }

          LOGGER.warn("Overriding configuration [" + key + "=" + originalValue + "] by " + "["
              + entry + "]");
        }
      }
    } catch (Exception e) {
      throw new PurposefulBaseException(e);
    } finally {
      IOUtils.closeQuietly(extensionInStream);
    }
  }

  protected String getStringValue(String key) {
    String value = CONFIGURATIONS.get(key);
    Assert.notNull(value, "Cannot find configuration key [" + key + "]");
    return value;
  }

  protected int getIntValue(String key) {
    return Integer.parseInt(getStringValue(key));
  }

  protected double getDoubleValue(String key) {
    return Double.parseDouble(getStringValue(key));
  }

  protected boolean getBooleanValue(String key) {
    return Boolean.parseBoolean(getStringValue(key));
  }

  protected List<String> getStringValues(String key) {
    return getStringValues(key, false);
  }

  protected List<String> getStringValues(String key, boolean inLowerCase) {
    String valuesString = getStringValue(key);
    if (inLowerCase) {
      valuesString = valuesString.toLowerCase(Locale.US);
    }

    LinkedHashSet<String> valuesSet = new LinkedHashSet<String>();
    StringTokenizer tokenizer = new StringTokenizer(valuesString, ",;");
    while (tokenizer.hasMoreTokens()) {
      valuesSet.add(tokenizer.nextToken().trim());
    }

    return new ArrayList<String>(valuesSet);
  }
}
