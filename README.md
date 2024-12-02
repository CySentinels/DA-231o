# DA-231o
Project repository for DA 231o Data Engineering at Scale (August 2024 Term) @ IISc BLR

---
To Setup the environment for working with Hadoop and Spark please follow the given steps:
	Genrally HDFS(Hadoop Distributed File System) and Spark are used for distributed system, but here as a basic tutorial the steps mentioned are for a stand alone Windows System. As it's easily avialble and cost effective for a tutorial purpose.

1. Install Java
	. Download and Install: Download the Java Development Kit (JDK)(https://www.oracle.com/java/technologies/javase-downloads.html) and install it.
	. Set JAVA_HOME Environment Variable:
		. Right-click on "This PC" or "My Computer" > Properties > Advanced System Settings > Environment Variables.
		. Add a new variable:
			|---------------------------------------------------------------------------------------|
			|	Variable name: JAVA_HOME															|
			|	Variable value: Path to JDK installation (e.g., C:\Program Files\Java\jdk-xx.x.x)	|
			|---------------------------------------------------------------------------------------|

		. Add %JAVA_HOME%\bin to the PATH variable.
2. Install Hadoop
	. Download Hadoop Binary: Get a pre-built Hadoop binary distribution for Windows from Apache Hadoop(https://hadoop.apache.org/).
	. Extract Hadoop: Extract the Hadoop binaries to a directory (e.g. 'C:\hadoop', for our case we used 'E:\IISC\hadoop').
	. Configure Hadoop:
		. In the etc/hadoop folder, configure the following files:
			. core-site.xml:
				|---------------------------------------------------------------------------------------|
				|	<configuration>																		|
				|		<property>																		|
				|			<name>fs.defaultFS</name>													|
				|			<value>hdfs://localhost:9000</value>										|
				|		</property>																		|
				|	</configuration>																	|
				|---------------------------------------------------------------------------------------|
			. hdfs-site.xml:
				|---------------------------------------------------------------------------------------|
				|	<configuration>																		|
				|		<property>																		|
				|			<name>dfs.replication</name>												|
				|			<value>1</value>															|
				|		</property>																		|
				|		<property>																		|
				|			<name>dfs.namenode.name.dir</name>											|
				|			<value>E:\IISC\hadoop\data\namenode</value>									|
				|		</property>																		|
				|		<property>																		|
				|			<name>dfs.datanode.data.dir</name>											|
				|			<value>E:\IISC\hadoop\data\datanode</value>									|
				|		</property>																		|
				|	</configuration>																	|
				|---------------------------------------------------------------------------------------|
			. mapred-site.xml
				|---------------------------------------------------------------------------------------|
				|	<configuration>																		|
				|		<property>																		|
				|			<name>mapreduce.framework.name</name>										|
				|			<value>yarn</value>															|
				|		</property>																		|
				|	</configuration>																	|
				|---------------------------------------------------------------------------------------|
			. yarn-site.xml
				|---------------------------------------------------------------------------------------|
				|	<configuration>																		|
				|		<property>																		|
				|			<name>yarn.nodemanager.aux-services</name>									|
				|			<value>mapreduce_shuffle</value>											|
				|		</property>																		|
				|		<property>																		|
				|			<name>yarn.nodemanager.auxservices.mapreduce.shuffle.class</name>			|
				|			<value>org.apache.hadoop.mapred.ShuffleHandler</value>						|
				|		</property>																		|
				|	</configuration>																	|
				|---------------------------------------------------------------------------------------|
		. Format the Hadoop file system:
			|---------------------------------------------------------------------------|
			|	hdfs namenode -format													|
			|---------------------------------------------------------------------------|
		. Start Hadoop services:
			|---------------------------------------------------------------------------|
			|	start-dfs.cmd															|
			|	start-yarn.cmd															|
			|---------------------------------------------------------------------------|
3. Install Apache Spark
	. Download Spark: Download the pre-built version for Hadoop from Apache Spark(https://spark.apache.org/downloads.html).
	. Extract Spark: Extract it to a directory (e.g., C:\spark).
	. Set SPARK_HOME Environment Variable:
		. Add a new variable:
			|-------------------------------------------------------------------------------------------------------|
			|	Variable name: SPARK_HOME																			|
			|	Variable value: Path to Spark installation (e.g. 'C:\spark', for our case we used 'E:\IISC\spark')	|
			|-------------------------------------------------------------------------------------------------------|
		. Add %SPARK_HOME%\bin to the PATH variable.
	. Verify Spark Installation: Open a command prompt and type:
		|---------------------------------------------------------------------------|
		|	spark-shell																|
		|---------------------------------------------------------------------------|
		This starts the Spark shell in standalone mode.
4. Configure Spark to Use Hadoop
	. Edit spark-env.cmd:
		. Navigate to E:\IISC\spark\conf and rename spark-env.cmd.template to spark-env.cmd.
		. Add the following lines:
			|---------------------------------------------------------------------------------------------------|
			|	set HADOOP_HOME=E:\IISC\hadoop																	|
			|	set SPARK_DIST_CLASSPATH=%HADOOP_HOME%\bin;%HADOOP_HOME%\lib;%HADOOP_HOME%\etc\hadoop			|
			|---------------------------------------------------------------------------------------------------|
	. Add Hadoop Binary Path: Ensure %HADOOP_HOME%\bin is in your system PATH for Spark to recognize Hadoop executables.
5. Test HDFS Connection in Spark
	. Launch spark-shell and try accessing HDFS:
		|---------------------------------------------------------------------------|
		|	val rdd = sc.textFile("hdfs://localhost:9000/path/to/file")				|
		|	rdd.collect().foreach(println)											|
		|---------------------------------------------------------------------------|
