from pyspark.sql import SparkSession
from dotenv import load_dotenv
from time import perf_counter

def create_spark_session():
    """Create a Spark Session"""
    _ = load_dotenv()
    return (
        SparkSession
        .builder
        .appName("SparkApp")
        .master("local[5]")
        .getOrCreate()
    )
spark = create_spark_session()
print('Session Started')

start = perf_counter()
spark.conf.set("spark.sql.caseSensitive", "true")
DATA_PATH = "data/phishing_urls.csv"
raw_sdf = spark.read.csv(DATA_PATH)
raw_sdf.show()

print(round((perf_counter() - start), 2), "seconds")

print('Code Executed Successfully')