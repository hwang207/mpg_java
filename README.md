## Setup
There are two options to use MPG framework:

1. Check out the source into a Eclipse project. Please refer to [mpg_development_environment_setup.pdf](https://github.com/hwang207/mpg_java/blob/master/mpg_development_environment_setup.pdf) in the root folder of the project.

2. Use MPG as a Maven dependency by adding the following into your 'pom.xml' file:  
  *Note that you still need to install Gurobi 6.5 and/or lp_solve as instructed in [mpg_development_environment_setup.pdf](https://github.com/hwang207/mpg_java/blob/master/mpg_development_environment_setup.pdf).*
```xml
<repositories>
	<repository>
		<id>mpg_java_release_mvn-repo</id>
		<url>https://github.com/hwang207/mpg_java/raw/mvn-repo/</url>
		<releases>
			<enabled>true</enabled>
		</releases>
		<snapshots>
			<enabled>true</enabled>
		</snapshots>
	</repository>
</repositories>

<dependencies>
	<dependency>
		<groupId>edu.uic.cs.purposeful</groupId>
		<artifactId>purposeful_mpg</artifactId>
		<version>1.0.0</version>
	</dependency>
</dependencies>
```

## Examples
We have unit-tests that go through training and test procedures:
* F-1: [edu.uic.cs.purposeful.mpg.target.binary.f1.TestBinaryF1Classifier](https://github.com/hwang207/mpg_java/blob/master/mpg_java/src/test/java/edu/uic/cs/purposeful/mpg/target/binary/f1/TestBinaryF1Classifier.java)
* P@k: [edu.uic.cs.purposeful.mpg.target.binary.precision.TestPrecisionAtKClassifier](https://github.com/hwang207/mpg_java/blob/master/mpg_java/src/test/java/edu/uic/cs/purposeful/mpg/target/binary/precision/TestPrecisionAtKClassifier.java)
* Datasets are in 'LIBSVM' format

## Configurations
* The configurations of MPG framework can be found in [mpg_config.properties](https://github.com/hwang207/mpg_java/blob/master/mpg_java/src/main/resources/config/mpg_config.properties)
* 'k_percent' for Precision@k is defined in [mpg_precision_at_k_config.properties](https://github.com/hwang207/mpg_java/blob/master/mpg_java/src/main/resources/config/mpg_precision_at_k_config.properties)
* If you are using MPG as a Maven dependency, and want to change any configuration above, you need to create your configuration 'extension' file(s) to override the existing one(s), and put it/them into 'config' folder in your **classpath** (usually it is 'src/main/resources' or 'src/main/java' in a Maven project):  
  **config/mpg_config.properties.extension**  
  *(for example, it contains 'show_running_tracing=false' to override the default 'show_running_tracing=true')*  
  
  **config/mpg_precision_at_k_config.properties.extension**  
  *(for example, it contains 'k_percent=0.9' to override the default 'k_percent=0.5')*
  
## Citation (BibTeX)
@inproceedings{wang2015adversarial,  
  title={Adversarial Prediction Games for Multivariate Losses},  
  author={Wang, Hong and Xing, Wei and Asif, Kaiser and Ziebart, Brian},  
  booktitle={Advances in Neural Information Processing Systems},  
  pages={2710--2718},  
  year={2015}  
}
