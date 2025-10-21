import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, abs as abs_diff
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

spark = SparkSession.builder.appName("Task4_FarePrediction_Assignment").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

MODEL_PATH = "models/ride_fare_predictor"
TRAINING_DATA_PATH = "training-dataset.csv"

if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] No model found. Training a new model using {TRAINING_DATA_PATH}...")
    train_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=False)
    train_df = train_df_raw.withColumn("distance_km", col("distance_km").cast(DoubleType())) \
                           .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))
    assembler = VectorAssembler(inputCols=["distance_km"], outputCol="features")
    train_data_with_features = assembler.transform(train_df)
    lr = LinearRegression(featuresCol="features", labelCol="fare_amount", predictionCol="prediction")
    model = lr.fit(train_data_with_features)
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Training Complete] Model saved to -> {MODEL_PATH}")
else:
    print(f"[Model Found] Using existing model from {MODEL_PATH}")

print("\n[Inference Phase] Starting real-time fare prediction stream...")

schema = StructType([
    StructField("trip_id", StringType()),
    StructField("driver_id", IntegerType()),
    StructField("distance_km", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("timestamp", StringType())
])

raw_stream = spark.readStream.format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

parsed_stream = raw_stream.select(from_json(col("value"), schema).alias("data")).select("data.*")

model = LinearRegressionModel.load(MODEL_PATH)

assembler_inference = VectorAssembler(inputCols=["distance_km"], outputCol="features")
stream_with_features = assembler_inference.transform(parsed_stream)

predictions = model.transform(stream_with_features)

predictions_with_deviation = predictions.withColumn(
    "deviation",
    abs_diff(col("fare_amount") - col("prediction"))
)

output_df = predictions_with_deviation.select(
    "trip_id", "driver_id", "distance_km", "fare_amount",
    col("prediction").alias("predicted_fare"), "deviation"
)

query = output_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
