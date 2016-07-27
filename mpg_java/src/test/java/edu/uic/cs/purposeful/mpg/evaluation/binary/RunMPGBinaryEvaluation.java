package edu.uic.cs.purposeful.mpg.evaluation.binary;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.output.ByteArrayOutputStream;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Iterables;
import com.google.common.primitives.Doubles;

import edu.uic.cs.purposeful.common.assertion.PurposefulBaseException;
import edu.uic.cs.purposeful.common.collection.CollectionUtils;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.learning.MaximizerPredictor.Prediction;
import edu.uic.cs.purposeful.mpg.learning.binary.MPGBinaryClassifier;
import edu.uic.cs.purposeful.mpg.target.binary.AbstractBinaryOptimizationTarget;
import edu.uic.cs.purposeful.mpg.target.binary.f1.BinaryF1;
import edu.uic.cs.purposeful.mpg.target.binary.precision.PrecisionAtK;

public class RunMPGBinaryEvaluation {
  private static final Logger LOGGER = Logger.getLogger(RunMPGBinaryEvaluation.class);

  private static final double[] DEFAULT_REGULARIZATION_PARAMETERS = new double[] {0.0,
      Math.pow(2, -6), Math.pow(2, -5), Math.pow(2, -4), Math.pow(2, -3), Math.pow(2, -2),
      Math.pow(2, -1), Math.pow(2, 0), Math.pow(2, 1), Math.pow(2, 2), Math.pow(2, 3),
      Math.pow(2, 4), Math.pow(2, 5), Math.pow(2, 6), Math.pow(2, 7), Math.pow(2, 8)};

  private static final File DATA_SETS_FOLDER = new File("datasets");

  private static final String TRAINING_DATA_SET_FILE_NAME_EXTENSION = "train";
  private static final String VALIDATION_DATA_SET_FILE_NAME_EXTENSION = "valid";
  private static final String TEST_DATA_SET_FILE_NAME_EXTENSION = "test";
  private static final String TRAINING_DATA_SET_FILE_NAME_SUFFIX =
      "." + TRAINING_DATA_SET_FILE_NAME_EXTENSION;
  private static final String VALIDATION_DATA_SET_FILE_NAME_SUFFIX =
      "." + VALIDATION_DATA_SET_FILE_NAME_EXTENSION;
  private static final String TEST_DATA_SET_FILE_NAME_SUFFIX =
      "." + TEST_DATA_SET_FILE_NAME_EXTENSION;
  private static final String THETA_FILE_NAME_SUFFIX = ".theta";
  private static final String FAILURE_FILE_NAME_SURFFIX = ".fail";

  private static final String KEY_WORDS_F1 = "f1";
  private static final String KEY_WORDS_PRECISION_AT_K = "p@k";

  private static final String OPTION_NAME_DATASET = "d";
  private static final String OPTION_NAME_MEASURE = "m";
  private static final String OPTION_NAME_FILE_NAME_PREFIXES = "f";
  private static final String OPTION_NAME_TARGET = "t";
  private static final String OPTION_NAME_REG = "r";
  private static final String OPTION_NAME_SKIP_IF_THETA_FILE_EXISTS = "s";
  private static final String OPTION_LONG_NAME_DATASET = "dataset";
  private static final String OPTION_LONG_NAME_MEASURE = "measure";
  private static final String OPTION_LONG_NAME_FILE_NAME_PREFIXES = "file_name_prefixes";
  private static final String OPTION_LONG_NAME_TARGET = "target";
  private static final String OPTION_LONG_NAME_REG = "reg";
  private static final String OPTION_LONG_NAME_SKIP_IF_THETA_FILE_EXISTS =
      "skip_if_theta_file_exists";
  private static final Options COMMAND_LINE_OPTIONS = new Options();
  static {
    Option dataSetName = Option.builder(OPTION_NAME_DATASET).longOpt(OPTION_LONG_NAME_DATASET)
        .hasArg().argName("name").desc("the name of data set which is located in 'datasets' folder")
        .required().build();
    Option measure = Option.builder(OPTION_NAME_MEASURE).longOpt(OPTION_LONG_NAME_MEASURE).hasArg()
        .argName(KEY_WORDS_F1 + "|" + KEY_WORDS_PRECISION_AT_K)
        .desc("the measure to optimize, can be either '" + KEY_WORDS_F1 + "' or '"
            + KEY_WORDS_PRECISION_AT_K + "'")
        .required().build();
    Option fileNamePrefixes = Option.builder(OPTION_NAME_FILE_NAME_PREFIXES)
        .longOpt(OPTION_LONG_NAME_FILE_NAME_PREFIXES).hasArgs().argName("prefix1 prefix2 ...")
        .desc(
            "name prefixes of '.train/valid/test' files in the specified 'dataset'; using all the files under the 'dataset' folder by default")
        .build();
    Option targetTag = Option.builder(OPTION_NAME_TARGET).longOpt(OPTION_LONG_NAME_TARGET).hasArg()
        .argName("tag").desc("target tag for training/prediction; default tag is 1").build();
    Option regularizationParameters = Option.builder(OPTION_NAME_REG).longOpt(OPTION_LONG_NAME_REG)
        .hasArgs().argName("value1 value2 ...")
        .desc("regularization parameters to try, separate by whitespace; default values: "
            + StringUtils.join(DEFAULT_REGULARIZATION_PARAMETERS, ' '))
        .build();
    Option skipIfThetaFileExists = new Option(OPTION_NAME_SKIP_IF_THETA_FILE_EXISTS,
        OPTION_LONG_NAME_SKIP_IF_THETA_FILE_EXISTS, false,
        "skip learning if theta file already exists; if not specified, learning is not skipped by default");
    COMMAND_LINE_OPTIONS.addOption(dataSetName);
    COMMAND_LINE_OPTIONS.addOption(measure);
    COMMAND_LINE_OPTIONS.addOption(fileNamePrefixes);
    COMMAND_LINE_OPTIONS.addOption(targetTag);
    COMMAND_LINE_OPTIONS.addOption(regularizationParameters);
    COMMAND_LINE_OPTIONS.addOption(skipIfThetaFileExists);
  }

  private static void train(String dataSetName, String[] fileNamePrefixes,
      Class<? extends AbstractBinaryOptimizationTarget> targetMeasureClass, int targetTag,
      double[] regularizationParameters, boolean skipIfThetaFileExists) {
    File dataSetFolder = new File(DATA_SETS_FOLDER, dataSetName);

    for (String fileNamePrefix : fileNamePrefixes) {
      File trainingDataSetFile =
          new File(dataSetFolder, fileNamePrefix + TRAINING_DATA_SET_FILE_NAME_SUFFIX);

      LOGGER.warn("Processing " + trainingDataSetFile);

      for (double regularizationParameter : regularizationParameters) {
        Regularization regularization = Regularization.l2(regularizationParameter);
        System.err.println("**** " + regularization + " target=" + targetTag);

        File thetaFile = createThetaFile(dataSetFolder, fileNamePrefix, targetMeasureClass,
            targetTag, regularization);
        if (thetaFile.exists() && skipIfThetaFileExists) {
          System.out.println(
              "Skip " + FilenameUtils.getBaseName(thetaFile.getName()) + " since it exists.");
          continue;
        }

        MPGBinaryClassifier binaryClassifier =
            new MPGBinaryClassifier(targetMeasureClass, targetTag);
        try {
          binaryClassifier.learn(trainingDataSetFile, regularization);
        } catch (PurposefulBaseException e) {
          LOGGER.error("Error during learning!", e);
          File failedThetaFile = new File(thetaFile.getAbsolutePath() + FAILURE_FILE_NAME_SURFFIX);
          try {
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            e.printStackTrace(new PrintStream(outputStream, true));
            FileUtils.write(failedThetaFile, outputStream.toString());
          } catch (IOException ioe) {
            LOGGER.error("", ioe);
          }
          continue;
        }

        binaryClassifier.writeModel(thetaFile);
        System.out.println("Wrote model to file: " + thetaFile);
      }
    }
  }

  private static void validateAndTest(String dataSetName, String[] fileNamePrefixes,
      Class<? extends AbstractBinaryOptimizationTarget> targetMeasureClass, int targetTag,
      double[] regularizationParameters) {
    File dataSetFolder = new File(DATA_SETS_FOLDER, dataSetName);

    for (String fileNamePrefix : fileNamePrefixes) {
      TreeMap<Double, TreeSet<Double>> regularizationParametersByValidationScore =
          new TreeMap<>(Comparator.reverseOrder());

      File validationDataSetFile =
          new File(dataSetFolder, fileNamePrefix + VALIDATION_DATA_SET_FILE_NAME_SUFFIX);
      File testDataSetFile =
          new File(dataSetFolder, fileNamePrefix + TEST_DATA_SET_FILE_NAME_SUFFIX);
      LOGGER.warn("Processing " + validationDataSetFile);
      LOGGER.warn("Processing " + testDataSetFile);

      for (double regularizationParameter : regularizationParameters) {
        Regularization regularization = Regularization.l2(regularizationParameter);
        System.err.println("**** " + regularization + " target=" + targetTag);

        File thetaFile = createThetaFile(dataSetFolder, fileNamePrefix, targetMeasureClass,
            targetTag, regularization);
        if (!thetaFile.exists()) {
          LOGGER.error(thetaFile + " doesn't exist!");
          continue;
        }

        MPGBinaryClassifier binaryClassifier =
            new MPGBinaryClassifier(targetMeasureClass, targetTag);
        binaryClassifier.loadModel(thetaFile);

        Prediction<BitSet> validationPrediction = null;
        try {
          validationPrediction = binaryClassifier.predict(validationDataSetFile);
        } catch (PurposefulBaseException e) {
          LOGGER.error("Error during validation!", e);
          continue;
        }

        double validationScore = validationPrediction.getScore();
        System.out.println(String.format("%s\treg=%s\tvalidScore=%f\tvalidProb=%f",
            validationDataSetFile.getName(), regularization, validationScore,
            validationPrediction.getProbability()));

        CollectionUtils.putInTreeSetValueMap(validationScore, regularizationParameter,
            regularizationParametersByValidationScore);
      }

      System.out.println("===================================================");
      Entry<Double, TreeSet<Double>> bestParametersByValidationScore =
          Iterables.getFirst(regularizationParametersByValidationScore.entrySet(), null);
      for (double regularizationParameter : bestParametersByValidationScore.getValue()) {
        Regularization regularization = Regularization.l2(regularizationParameter);
        File thetaFile = createThetaFile(dataSetFolder, fileNamePrefix, targetMeasureClass,
            targetTag, regularization);

        MPGBinaryClassifier binaryClassifier =
            new MPGBinaryClassifier(targetMeasureClass, targetTag);
        binaryClassifier.loadModel(thetaFile);

        Prediction<BitSet> testPrediction = null;
        Stopwatch stopwatch = Stopwatch.createStarted();
        try {
          testPrediction = binaryClassifier.predict(testDataSetFile);
        } catch (PurposefulBaseException e) {
          LOGGER.error("Error during test!", e);
          continue;
        }
        stopwatch.stop();

        System.out.println(String.format(
            ">>>>>\t%s\tvalidationScore=%f\treg=%s\ttestScore=%f\ttestProb=%f\ttestCost=%s",
            testDataSetFile.getName(), bestParametersByValidationScore.getKey(), regularization,
            testPrediction.getScore(), testPrediction.getProbability(), stopwatch));
      }
    }
  }

  private static File createThetaFile(File dataSetFolder, String fileNamePrefix,
      Class<? extends AbstractBinaryOptimizationTarget> targetMeasureClass, int targetTag,
      Regularization regularization) {
    return new File(dataSetFolder,
        dataSetFolder.getName() + "." + fileNamePrefix + "." + targetMeasureClass.getSimpleName()
            + "." + targetTag + "." + regularization + THETA_FILE_NAME_SUFFIX);
  }

  public static void main(String[] args) {
    CommandLineParser commandLineParser = new DefaultParser();
    CommandLine commandLine = null;
    try {
      commandLine = commandLineParser.parse(COMMAND_LINE_OPTIONS, args);
    } catch (ParseException exp) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.setWidth(300);
      formatter.printHelp(RunMPGBinaryEvaluation.class.getSimpleName(), COMMAND_LINE_OPTIONS, true);
      return;
    }

    String dataSetName = commandLine.getOptionValue(OPTION_NAME_DATASET);
    String measure = commandLine.getOptionValue(OPTION_LONG_NAME_MEASURE);
    boolean skipIfThetaFileExists = commandLine.hasOption(OPTION_NAME_SKIP_IF_THETA_FILE_EXISTS);
    String[] regValueStrings = commandLine.getOptionValues(OPTION_NAME_REG);
    String[] fileNamePrefixes = commandLine.getOptionValues(OPTION_NAME_FILE_NAME_PREFIXES);
    String targetTagString = commandLine.getOptionValue(OPTION_NAME_TARGET);

    Class<? extends AbstractBinaryOptimizationTarget> targetMeasureClass = null;
    if (measure.equalsIgnoreCase(KEY_WORDS_F1)) {
      targetMeasureClass = BinaryF1.class;
    } else if (measure.equalsIgnoreCase(KEY_WORDS_PRECISION_AT_K)) {
      targetMeasureClass = PrecisionAtK.class;
    } else {
      System.err.println("Measure [" + measure + "] is not supported.");
      HelpFormatter formatter = new HelpFormatter();
      formatter.setWidth(300);
      formatter.printHelp(RunMPGBinaryEvaluation.class.getSimpleName(), COMMAND_LINE_OPTIONS, true);
      return;
    }

    double[] regularizationParameters = DEFAULT_REGULARIZATION_PARAMETERS;
    if (regValueStrings != null) {
      List<Double> _regularizationParameters = new ArrayList<>(regValueStrings.length);
      for (String regValueString : regValueStrings) {
        _regularizationParameters.add(Double.valueOf(regValueString));
      }
      if (!_regularizationParameters.isEmpty()) {
        regularizationParameters = Doubles.toArray(_regularizationParameters);
      }
    }

    if (fileNamePrefixes == null) {
      File dataSetFolder = new File(DATA_SETS_FOLDER, dataSetName);
      List<String> _fileNamePrefixes = new ArrayList<>();
      for (File trainingFiles : FileUtils.listFiles(dataSetFolder,
          new String[] {TRAINING_DATA_SET_FILE_NAME_EXTENSION}, false)) {
        _fileNamePrefixes.add(FilenameUtils.getBaseName(trainingFiles.getName()));
      }
      fileNamePrefixes = _fileNamePrefixes.toArray(new String[_fileNamePrefixes.size()]);
    }

    int targetTag = targetTagString == null ? 1 : Integer.parseInt(targetTagString);

    System.err.println("**** Data set:\t" + dataSetName);
    System.err.println("**** Measure:\t" + measure);
    System.err.println("**** File name prefixes:\t" + Arrays.toString(fileNamePrefixes));
    System.err
        .println("**** Regularization parameters:\t" + Arrays.toString(regularizationParameters));
    System.err.println("**** Skip if theta file exists:\t" + skipIfThetaFileExists);
    System.err.println("**** Target tag:\t" + targetTag);

    train(dataSetName, fileNamePrefixes, targetMeasureClass, targetTag, regularizationParameters,
        skipIfThetaFileExists);

    MPGConfig.SHOW_RUNNING_TRACING = false;
    validateAndTest(dataSetName, fileNamePrefixes, targetMeasureClass, targetTag,
        regularizationParameters);
  }
}
