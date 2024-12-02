# DA-231o
Project repository for DA 231o Data Engineering at Scale (August 2024 Term) @ IISc BLR

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
