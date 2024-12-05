from time import perf_counter
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import Evaluator
from pyspark.sql.types import *
from pyspark.sql.functions import col, abs
import pyspark.sql.functions as F
from pyspark.sql.window import Window as W
from pyspark.ml.feature import *
from pyspark.ml.linalg import Vectors, VectorUDT
import time
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, NaiveBayes, LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import RandomForestClassificationModel

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


start = perf_counter() # to record start time


# Create a Spark session
spark = SparkSession.builder.master("local").appName("PhishingAnalysis") \
                    .config('spark.ui.port', '4051').getOrCreate()

#Setting Log Level
sc = spark.sparkContext
sc.setLogLevel("FATAL") # Options: ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE, WARN

# Load the CSV file from HDFS
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("hdfs://localhost:9000/phishing/data/phishing_urls.csv")

# Defining Feature and Target Columns for Model Training
feature_columns = [
    "URLSimilarityIndex",
    "LineOfCode",
    "NoOfExternalRef",
    "NoOfSelfRef",
    "NoOfCSS",
    "NoOfImage",
    "HasSocialNet",
    "HasCopyrightInfo",
    "HasDescription",
    "NoOfJS"
]
# Define target column
target_column = 'label'

# Assembling Features and Transforming Dataset
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
# Transform the dataset
df = assembler.transform(df)
# View the dataset with features and target
df.select('features', target_column).show(5)

# Splitting the Data into Training and Testing Sets
# Splitting the data into 80% training and 20% testing
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1947)

# Print the number of records in each set
print(f"Training Data Count: {train_data.count()}")
print(f"Testing Data Count: {test_data.count()}")


# Label Distribution in Training and Testing Sets
# distribution of the label in the training set
train_data.groupBy("label").count().show()

# distribution of the label in the testing set
test_data.groupBy("label").count().show()

# Defining Evaluation Metric for Classification Model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Defining Random Forest Classifier Model
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Defining Hyperparameter Grid for Random Forest
paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [50, 100]) \
                              .addGrid(rf.maxDepth, [5, 10]) \
                              .build()

# Setting Up Cross-Validation for Random Forest
rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

# Training, Testing, and Evaluating the Random Forest Model
# Record start time for training
train_start_time = time.time()

# Train model
rf_cv_model = rf_cv.fit(train_data)

# Record end time for training
train_end_time = time.time()
train_duration = train_end_time - train_start_time
print(f"Training Duration: {train_duration:.2f} seconds")
# Record start time for testing
test_start_time = time.time()

# Test model
rf_predictions = rf_cv_model.bestModel.transform(test_data)
rf_accuracy = evaluator.evaluate(rf_predictions)

# Record end time for testing
test_end_time = time.time()
test_duration = test_end_time - test_start_time
print(f"Testing Duration: {test_duration:.2f} seconds")
# Output results
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Best Hyperparameters: ")
best_params = rf_cv_model.bestModel.extractParamMap()
for param, value in best_params.items():
    print(f"{param.name}: {value}")


# Defining Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

# Defining Hyperparameter Grid for Decision Tree Model
paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10]) \
                              .addGrid(dt.minInstancesPerNode, [1, 5]) \
                              .build()

# Setting Up Cross-Validation for Decision Tree Model
dt_cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)


# Decision Tree Model Trainin , Testing & Evaluating
# Start time for training
start_train_time = time.time()
# Train model
dt_cv_model = dt_cv.fit(train_data)
# End time for training
end_train_time = time.time()
train_duration = end_train_time - start_train_time
print(f"Training Duration: {train_duration:.2f} seconds")
# Start time for testing
start_test_time = time.time()
# Test model
dt_predictions = dt_cv_model.bestModel.transform(test_data)
dt_accuracy = evaluator.evaluate(dt_predictions)
# End time for testing
end_test_time = time.time()
test_duration = end_test_time - start_test_time
print(f"Testing Duration: {test_duration:.2f} seconds")
# Output result
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Best Decision Tree Hyperparameters: ")
best_params = dt_cv_model.bestModel.extractParamMap()
for param, value in best_params.items():
    print(f"{param.name}: {value}")


# Naive Bayes Model Definition
nb = NaiveBayes(labelCol="label", featuresCol="features")
# Defining Hyperparameter Grid for Naive Bayes Model
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.5, 1.0, 2.0]).build()
# Setting Up Cross-Validation for Naive Bayes Model
nb_cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
# Training, Testing, and Evaluating the Naive Bayes Model
# Start time for training
start_train_time = time.time()
# Train model
nb_cv_model = nb_cv.fit(train_data)
# End time for training
end_train_time = time.time()
train_duration = end_train_time - start_train_time
print(f"Training Duration: {train_duration:.2f} seconds")
# Start time for testing
start_test_time = time.time()
# Test model
nb_predictions = nb_cv_model.bestModel.transform(test_data)
nb_accuracy = evaluator.evaluate(nb_predictions)
# End time for testing
end_test_time = time.time()
test_duration = end_test_time - start_test_time
print(f"Testing Duration: {test_duration:.2f} seconds")
# Output results
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
# Output the best hyperparameters for Naive Bayes
print(f"Best Naive Bayes Hyperparameters: ")
best_params = nb_cv_model.bestModel.extractParamMap()
for param, value in best_params.items():
    print(f"{param.name}: {value}")

# Defining SVM Model
svm = LinearSVC(labelCol="label", featuresCol="features")
# Defining Hyperparameter Grid for SVM Model
paramGrid = ParamGridBuilder().addGrid(svm.maxIter, [50, 100]) \
                              .addGrid(svm.regParam, [0.01, 0.1]) \
                              .build()
# Setting Up Cross-Validation for SVM Model
svm_cv = CrossValidator(estimator=svm, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
# Training, Testing, and Evaluating the SVM Model
# Start time for training
start_train_time = time.time()
# Train model
svm_cv_model = svm_cv.fit(train_data)
# End time for training
end_train_time = time.time()
train_duration = end_train_time - start_train_time
print(f"Training Duration: {train_duration:.2f} seconds")
# Start time for testing
start_test_time = time.time()
# Test model
svm_predictions = svm_cv_model.bestModel.transform(test_data)
svm_accuracy = evaluator.evaluate(svm_predictions)
# End time for testing
end_test_time = time.time()
test_duration = end_test_time - start_test_time
print(f"Testing Duration: {test_duration:.2f} seconds")
# Output results
print(f"SVM Accuracy: {svm_accuracy:.4f}")
# Output the best hyperparameters for SVM
print(f"Best SVM Hyperparameters: ")
best_params = svm_cv_model.bestModel.extractParamMap()
for param, value in best_params.items():
    print(f"{param.name}: {value}")

# Model Comparison and Evaluation: Random Forest, Decision Tree, Naive Bayes, and SVM
# Define a function to evaluate metrics for each model
def evaluate_model(model_name, model, test_data):
    # Generate predictions
    predictions = model.transform(test_data)

    # Initialize evaluators for each metric
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    # Calculate metrics
    accuracy = accuracy_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)

    # Return results as a dictionary
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }

# Evaluate all models
results = []

# Random Forest
results.append(evaluate_model("Random Forest", rf_cv_model.bestModel, test_data))

# Decision Tree
results.append(evaluate_model("Decision Tree", dt_cv_model.bestModel, test_data))

# Naive Bayes
results.append(evaluate_model("Naive Bayes", nb_cv_model.bestModel, test_data))

# SVM
results.append(evaluate_model("SVM", svm_cv_model.bestModel, test_data))

# Display results in a sorted order based on F1-Score
results_sorted = sorted(results, key=lambda x: x["F1-Score"], reverse=True)

# Print results
print("\nComparison of Models:")
for result in results_sorted:
    print(f"Model: {result['Model']}")
    print(f"  Accuracy: {result['Accuracy']:.4f}")
    print(f"  Precision: {result['Precision']:.4f}")
    print(f"  Recall: {result['Recall']:.4f}")
    print(f"  F1-Score: {result['F1-Score']:.4f}")
    print()

# Identify the best model
best_model = results_sorted[0]
print(f"Best Model: {best_model['Model']}")
print(f"  Accuracy: {best_model['Accuracy']:.4f}")
print(f"  Precision: {best_model['Precision']:.4f}")
print(f"  Recall: {best_model['Recall']:.4f}")
print(f"  F1-Score: {best_model['F1-Score']:.4f}")


# Confusion Matrix Comparison for Phishing Detection Models
# Function to calculate confusion matrix
def get_confusion_matrix(model, test_data):
    predictions = model.transform(test_data)
    # Ensure the prediction column is in integer format
    confusion_matrix = predictions.groupBy("label", "prediction").count().toPandas()

    # Labels for confusion matrix (0 = Phishing, 1 = Legitimate)
    labels = [0, 1]

    # Initialize the matrix with zeros
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    # Populate the confusion matrix
    for _, row in confusion_matrix.iterrows():
        actual = int(row["label"])  # Convert label to integer
        predicted = int(row["prediction"])  # Convert prediction to integer
        matrix[actual, predicted] = row["count"]

    return matrix, labels

# Function to plot a confusion matrix heatmap
def plot_confusion_matrix(matrix, labels, model_name, ax):
    sns.heatmap(matrix, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix for {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# List to store confusion matrices for each model
confusion_matrices = []

# Generate confusion matrices for each model
models = [
    ("Random Forest", rf_cv_model.bestModel),
    ("Decision Tree", dt_cv_model.bestModel),
    ("Naive Bayes", nb_cv_model.bestModel),
    ("SVM", svm_cv_model.bestModel),
]

# Plot confusion matrices for each model
for idx, (model_name, model) in enumerate(models):
    matrix, labels = get_confusion_matrix(model, test_data)
    row = idx // 2  # Determine row (0 or 1)
    col = idx % 2   # Determine column (0 or 1)
    plot_confusion_matrix(matrix, labels, model_name, axes[row, col])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# Displaying Feature Importances from Best Model (Random Forest Model)
# Access feature importances from the trained Random Forest model
importances = rf_cv_model.bestModel.featureImportances.toArray()

# Map features to their importance scores
feature_importances = [(feature, importance) for feature, importance in zip(feature_columns, importances)]

# Sort and display feature importances
sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

# Saving the Best Random Forest Model in PySpark
best_model = rf_cv_model.bestModel


spark.stop()