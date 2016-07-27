package edu.uic.cs.purposeful.common.collection;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.collect.Iterables;

import edu.uic.cs.purposeful.common.assertion.Assert;

public class CollectionUtils extends org.apache.commons.collections.CollectionUtils {

  public synchronized static <K, V> void putInArrayListValueMap(K key, V value,
      Map<K, ArrayList<V>> map) {

    ArrayList<V> values = map.get(key);
    if (values == null) {
      synchronized (map) {
        values = map.get(key);
        if (values == null) {
          values = new ArrayList<V>();
          Assert.isNull(map.put(key, values), "Map should not has key [" + key + "].");
        }
      }
    }

    values.add(value);
  }

  public synchronized static <K, V> void putInLinkedListValueMap(K key, V value,
      Map<K, LinkedList<V>> map) {

    LinkedList<V> values = map.get(key);
    if (values == null) {
      synchronized (map) {
        values = map.get(key);
        if (values == null) {
          values = new LinkedList<V>();
          Assert.isNull(map.put(key, values), "Map should not has key [" + key + "].");
        }
      }
    }

    values.add(value);
  }

  public synchronized static <K, V> void putInTreeSetValueMap(K key, V value,
      Map<K, TreeSet<V>> map) {

    TreeSet<V> values = map.get(key);
    if (values == null) {
      synchronized (map) {
        values = map.get(key);
        if (values == null) {
          values = new TreeSet<V>();
          Assert.isNull(map.put(key, values), "Map should not has key [" + key + "].");
        }
      }
    }

    values.add(value);

  }

  public synchronized static <K, V> void putInHashSetValueMap(K key, V value,
      Map<K, HashSet<V>> map) {

    HashSet<V> values = map.get(key);
    if (values == null) {
      synchronized (map) {
        values = map.get(key);
        if (values == null) {
          values = new HashSet<V>();
          Assert.isNull(map.put(key, values), "Map should not has key [" + key + "].");
        }
      }
    }

    values.add(value);
  }

  public static <T extends Comparable<T>> TreeSet<T> addAllIntoTreeSet(List<List<T>> lists) {

    Iterator<T> iterator = Iterables.concat(lists).iterator();
    TreeSet<T> result = new TreeSet<T>();
    addAll(result, iterator);

    return result;
  }

  public static <T> LinkedHashSet<T> addAllIntoLinkedHashSet(List<List<T>> lists) {

    Iterator<T> iterator = Iterables.concat(lists).iterator();
    LinkedHashSet<T> result = new LinkedHashSet<T>();
    addAll(result, iterator);

    return result;
  }

  public static <E extends Enum<E>> String[] convertEnumToStringArray(Class<E> enumClass) {
    return convertEnumToStringArray(enumClass.getEnumConstants());
  }

  public static <E extends Enum<E>> String[] convertEnumToStringArray(E[] enumvalues) {
    String[] result = new String[enumvalues.length];
    int index = 0;
    for (E constant : enumvalues) {
      result[index++] = constant.name();
    }

    return result;
  }

  public static boolean containsAnyIgnoreCase(Collection<String> coll1, Collection<String> coll2) {

    TreeSet<String> set1 = new TreeSet<String>(String.CASE_INSENSITIVE_ORDER);
    set1.addAll(coll1);

    TreeSet<String> set2 = new TreeSet<String>(String.CASE_INSENSITIVE_ORDER);
    set2.addAll(coll2);

    return containsAny(set1, set2);
  }

  private static final Integer INTEGER_ONE = Integer.valueOf(1);

  public static <K> void countKey(K key, Map<K, Integer> countsByKey) {
    synchronized (countsByKey) {
      Integer count = countsByKey.get(key);
      if (count == null) {
        countsByKey.put(key, Integer.valueOf(INTEGER_ONE));
      } else {
        countsByKey.put(key, Integer.valueOf(count.intValue() + 1));
      }
    }
  }

  public static <K> void countKey2(K key, Map<K, AtomicInteger> countsByKey) {
    synchronized (countsByKey) {
      AtomicInteger count = countsByKey.get(key);
      if (count == null) {
        countsByKey.put(key, new AtomicInteger(1));
      } else {
        count.incrementAndGet();
      }
    }
  }

  public static <K, V extends Comparable<? super V>> Map<K, V> sortByValueInReverseOrder(
      Map<K, V> map) {
    return sortByValue(map, true);
  }

  public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
    return sortByValue(map, false);
  }

  private static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map,
      final boolean reverseOrder) {

    List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
    Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
      public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
        if (reverseOrder) {
          return (o2.getValue()).compareTo(o1.getValue());
        } else {
          return (o1.getValue()).compareTo(o2.getValue());
        }
      }
    });

    Map<K, V> result = new LinkedHashMap<K, V>(list.size());
    for (Map.Entry<K, V> entry : list) {
      result.put(entry.getKey(), entry.getValue());
    }

    return Collections.unmodifiableMap(result);
  }
}
