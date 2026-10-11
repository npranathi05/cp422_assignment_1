from pyspark.sql import functions as F
from pyspark.sql.functions import col

try:
    df_clean
    print("df_clean already exists - using Part A's version")
except NameError:
    df = spark.read.csv(
        "/Volumes/workspace/default/cp422_a1/yellow_tripdata_2015-01.csv",
        header=True,
        inferSchema=True
    )
    df_clean = df.dropna(subset=["fare_amount", "trip_distance", "passenger_count"])
    df_clean = df_clean.filter((col("fare_amount") > 0) & (col("trip_distance") > 0))
    df_clean = df_clean.withColumn(
        "trip_duration",
        (col("tpep_dropoff_datetime").cast("long") - col("tpep_pickup_datetime").cast("long")) / 60
    )
    df_clean = df_clean.filter(col("trip_duration") > 0)
    df_clean = df_clean.withColumn("trip_speed", col("trip_distance") / (col("trip_duration") / 60))
    print("df_clean rebuilt")

geo = df_clean.filter(
    F.col("pickup_longitude").between(-74.30, -73.70) &
    F.col("pickup_latitude").between(40.50, 40.95) &
    F.col("dropoff_longitude").between(-74.30, -73.70) &
    F.col("dropoff_latitude").between(40.50, 40.95)
)

geo = (geo
    .withColumn("pickup_loc_id",
        F.concat_ws(",", F.round("pickup_latitude", 3), F.round("pickup_longitude", 3)))
    .withColumn("dropoff_loc_id",
        F.concat_ws(",", F.round("dropoff_latitude", 3), F.round("dropoff_longitude", 3))))

print("Rows with valid NYC coordinates:", geo.count())

import matplotlib.pyplot as plt

top_pickups = (geo.groupBy("pickup_loc_id").count()
                  .orderBy(F.desc("count")).limit(15).toPandas())
top_dropoffs = (geo.groupBy("dropoff_loc_id").count()
                   .orderBy(F.desc("count")).limit(15).toPandas())

print("Top 15 pickup locations")
display(top_pickups)
print("Top 15 drop-off locations")
display(top_dropoffs)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, data, c, color, title in [
    (axes[0], top_pickups, "pickup_loc_id", "steelblue", "Top 15 Pickup Locations"),
    (axes[1], top_dropoffs, "dropoff_loc_id", "darkorange", "Top 15 Drop-off Locations")]:
    d = data.sort_values("count")  # ascending so the largest bar is on top
    ax.barh(d[c], d["count"], color=color)
    ax.set_title(title + " by Trip Frequency (Jan 2015)")
    ax.set_xlabel("Number of Trips")
    ax.set_ylabel("Grid Cell (lat, lon)")
    ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

fare_by_zone = (geo.groupBy("pickup_loc_id")
    .agg(F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
         F.count("*").alias("trips"))
    .filter(F.col("trips") >= 500)
    .orderBy(F.desc("avg_fare")))

print("Top 15 highest average-fare pickup locations (>= 500 trips)")
top_fare = fare_by_zone.limit(15).toPandas()
display(top_fare)

d = top_fare.sort_values("avg_fare")
plt.figure(figsize=(10, 7))
plt.barh(d["pickup_loc_id"], d["avg_fare"], color="seagreen")
plt.title("Highest Average Fare by Pickup Location (min. 500 trips)")
plt.xlabel("Average Fare Amount (USD)")
plt.ylabel("Pickup Grid Cell (lat, lon)")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

import seaborn as sns
import pandas as pd

corr_df = df_clean.filter(
    (F.col("trip_duration") > 0) & (F.col("trip_duration") < 180) &
    (F.col("trip_distance") < 100) & (F.col("fare_amount") < 500)
)

cols = ["trip_duration", "trip_distance", "fare_amount"]
matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
for a in cols:
    for b in cols:
        matrix.loc[a, b] = 1.0 if a == b else corr_df.stat.corr(a, b, method="pearson")

print(matrix.round(4))

plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
            square=True, cbar_kws={"label": "Pearson r"})
plt.title("Pearson Correlation: Duration, Distance, Fare")
plt.tight_layout()
plt.show()
