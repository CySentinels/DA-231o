# DA-231o
Project repository for DA 231o Data Engineering at Scale (August 2024 Term) @ IISc BLR

# Project Purpose
The purpose of this project is to develop a reliable machine learning-based system for detecting phishing URLs by analyzing their structure, content, and behavior which can run on distributed system, in order to keep up with continuous increasing data load with number of new legitimate websites getting deployed every second and even more evolving threat actors with new and better ideas.

# Dataset
**Source**: [PhiUSIIL Phishing URL (Website)](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

**Summary**: PhiUSIIL Phishing URL Dataset is a substantial dataset comprising 134,850 legitimate and 100,945 phishing URLs. Most of the URLs we analyzed, while constructing the dataset, are the latest URLs. Features are extracted from the source code of the webpage and URL. Features such as CharContinuationRate, URLTitleMatchScore, URLCharProb, and TLDLegitimateProb are derived from existing features.

**Additional Info**:
1. Column "FILENAME" can be ignored.
2. Label 1 corresponds to a legitimate URL, label 0 to a phishing URL

# Setting Up Apache Spark with Hadoop on Windows

This guide helps you set up Apache Spark in a standalone mode on a Windows system and connect it with Hadoop.

---

## Prerequisites

- **Java**: Download and install the [JDK](https://www.oracle.com/java/technologies/javase-downloads.html) and set `JAVA_HOME`.
- **Hadoop**: Download the [Hadoop binaries](https://hadoop.apache.org/) and configure it.
- **Apache Spark**: Download the pre-built version for Hadoop from [Apache Spark](https://spark.apache.org/downloads.html).

---

## Steps

### 1. Set Up Java
- Set the `JAVA_HOME` environment variable and add it to `PATH`.
```cmd
set JAVA_HOME=C:\Program Files\Java\jdk-xx.x.x
set PATH=%JAVA_HOME%\bin;%PATH%
```
### 2. Set Up Hadoop
- **Extract Hadoop**: Extract the Hadoop binaries to a directory (e.g. 'C:\hadoop')
- **Configure Hadoop**:
  - In the etc/hadoop folder, configure the following files:
    - core-site.xml:
	```xml
	<configuration>
	  <property>
	    <name>fs.defaultFS</name>
	    <value>hdfs://localhost:9000</value>
	  </property>
	</configuration>
	```   
	
    - hdfs-site.xml:
	```xml
	<configuration>
	  <property>
		<name>dfs.replication</name>
		<value>1</value>
	  </property>
	  <property>
		<name>dfs.namenode.name.dir</name>
		<value>C:\hadoop\data\namenode</value>
	  </property>
	  <property>
		<name>dfs.datanode.data.dir</name>
		<value>C:\hadoop\data\datanode</value>
	  </property>
	</configuration>
	```
	
    - mapred-site.xml:
	```xml
	<configuration>
	  <property>
		<name>mapreduce.framework.name</name>
		<value>yarn</value>
	  </property>
	</configuration>
	```
	
    - yarn-site.xml:
	```xml
	<configuration>
	  <property>
		<name>yarn.nodemanager.aux-services</name>
		<value>mapreduce_shuffle</value>
	  </property>
	  <property>
		<name>yarn.nodemanager.auxservices.mapreduce.shuffle.class</name>
		<value>org.apache.hadoop.mapred.ShuffleHandler</value>
	  </property>
	</configuration>
	```
	
  - Format the Hadoop file system:
    ```cmd
    hdfs namenode -format
    ```
	
  - Start Hadoop services:
    ```cmd
    start-dfs.cmd
	start-yarn.cmd
    ```
	
### 3. Set Up Apache Spark
- **Extract Hadoop**: Extract the downloaded pre-built version for Hadoop from Apache Spark (e.g., C:\spark).
- Set the `SPARK_HOME` environment variable and add it to `PATH`.
```cmd
set SPARK_HOME=C:\spark
set PATH=%SPARK_HOME%\bin;%PATH%
```
- Verify Spark Installation: Open a command prompt and type:
```cmd
spark-shell
```
### 4. Configure Spark to Use Hadoop
- Edit **spark-env.cmd**:
  - Navigate to C:\spark\conf and rename spark-env.cmd.template to spark-env.cmd.
  - Add these following lines:
  ```cmd
  set HADOOP_HOME=E:\IISC\hadoop
  set SPARK_DIST_CLASSPATH=%HADOOP_HOME%\bin;%HADOOP_HOME%\lib;%HADOOP_HOME%\etc\hadoop
  ```
- **Add Hadoop Binary Path**: Ensure %HADOOP_HOME%\bin is in your system PATH for Spark to recognize Hadoop executables.
### 5. Test HDFS Connection in Spark
- Launch spark-shell and try accessing HDFS:
```scala
val rdd = sc.textFile("hdfs://localhost:9000/path/to/file")
rdd.collect().foreach(println)
```

# Project Steps
### 1. Exploratory Data Analysis (EDA)

**Dataset Overview**
1. Basic exploration: printSchema(), describe(), and dropDuplicates().
2. Dataset contains 235,795 rows, no missing or duplicate values.
   
**Key Groupings of Features**
1. URL Characteristics: Length, special characters, obfuscation metrics.
2. Legitimacy Indicators: HTTPS usage, TLD legitimacy, subdomains.
3. Web Page Content: Title, favicon, and descriptions.
4. Web Page Features: Redirects, popups, and social network links.

**Hypotheses and Findings**
1. URL Length: Longer URLs are more likely to be phishing.
2. TLDs: Suspicious TLDs are common in phishing URLs.
3. HTTPS: Both phishing and legitimate URLs use HTTPS, reducing its reliability as a single indicator.
4. Obfuscation: Phishing URLs frequently use obfuscation techniques.
   
For detailed visualizations and analysis, refer to 01-EDA.ipynb.

### 2. Model Training and Evaluation

**Models Trained:** Decision Tree, Random Forest, SVM, Naive Bayes.

**Steps:**

  **1. Data Preparation:** Categorical encoding, feature-target definition, train-test split.
  
  **2. Model Training:** Built initial models.
  
  **3. Hyperparameter Tuning:** Used GridSearchCV with cross-validation to optimize parameters.
  
  **4. Evaluation:** Assessed performance using accuracy, precision, recall, F1-score, ROC-AUC, and PR curves.

## Best Model:
 Random Forest achieved the highest performance.
 Saved as best_random_forest_model.model.zip.

## Refer to the below files for details analysis and documentation:
- **01_EDA.ipynb**: The Jupyter Notebook file of Exploratory Data Analysis on the collected data, for a combined look of code, markdown, and outputs in a single document.
- **02-ModelTraining.ipynb**: The Jupyter Notebook file of Model Training and Saving the Best Model to local system, for a combined look of code, markdown, and outputs in a single document.
- To run on the **Single-Node Spark Cluster with HDFS** (To Setup the same in Windows Environment steps are given above)
  - **eda.py**
  - **model_training.py**
- **requirements.txt**: contains list of required Python libraries

## Contributors
1. Shambo Samanta
2. Deepansh Sood
3. Sudipta Ghosh
4. Sourajit Bhar
