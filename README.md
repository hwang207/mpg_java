# Setup
Please refer to ["mpg_development_environment_setup.pdf"](https://github.com/hwang207/mpg_java/blob/master/mpg_development_environment_setup.pdf) in the root folder of the project.

# Examples
We have an unit-test that goes through training and test procedures:
* Binary F-1: [edu.uic.cs.purposeful.mpg.target.binary.f1.TestBinaryF1Classifier](https://github.com/hwang207/mpg_java/blob/master/mpg_java/src/test/java/edu/uic/cs/purposeful/mpg/target/binary/f1/TestBinaryF1Classifier.java)
* Dataset format in MPG is 'LIBSVM' format

# Configurations
* The configurations of MPG framework can be found in ['mpg_config.properties'](https://github.com/hwang207/mpg_java/blob/master/mpg_java/src/main/resources/edu/uic/cs/purposeful/mpg/mpg_config.properties)
* 'k_percent' for Precision@k is defined in ['mpg_precision_at_k_config.properties'](https://github.com/hwang207/mpg_java/blob/master/mpg_java/src/main/resources/edu/uic/cs/purposeful/mpg/target/binary/precision/mpg_precision_at_k_config.properties)

# Citation (BibTeX)
@inproceedings{wang2015adversarial,  
  title={Adversarial Prediction Games for Multivariate Losses},  
  author={Wang, Hong and Xing, Wei and Asif, Kaiser and Ziebart, Brian},  
  booktitle={Advances in Neural Information Processing Systems},  
  pages={2710--2718},  
  year={2015}  
}
