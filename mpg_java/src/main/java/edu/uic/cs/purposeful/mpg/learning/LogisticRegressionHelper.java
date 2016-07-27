package edu.uic.cs.purposeful.mpg.learning;

import de.bwaldvogel.liblinear.Model;
import edu.uic.cs.purposeful.mpg.common.Regularization;

public interface LogisticRegressionHelper<Prediction, DataSet> {

  Model learnModel(DataSet dataSet, Regularization regularization);

  double[] learnWeights(DataSet dataSet, Regularization regularization);

  Prediction predict(DataSet dataSet, Model model);
}
