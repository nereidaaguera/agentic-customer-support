from pyspark.sql import SparkSession

# create single Spark session to be used throughout the application
spark = SparkSession.builder.appName("telco_support_agent").getOrCreate()
