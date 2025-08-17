# ============================
# Chicago Crime — Feature Engineering (Stage2, aligned to Stage1)
# - Uses Stage1 columns exactly (Beat, Date_TS, CrimeType, Hour, Arrest_Flag, Year, Month, Lat_Bin, Lng_Bin)
# - No data leakage: all windows use rowsBetween(-inf, -1) or lag()
# - Athena/QuickSight-friendly Parquet (snappy, partitioned by Year, Month)
# ============================

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# ------------------------------------------
#  Initialize Spark Session (Glue 4.0 safe)
# ------------------------------------------
spark = (
    SparkSession.builder
    .appName("CrimeFeatureEngineering_Stage2")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .getOrCreate()
)

# ------------------------------------------
#  Input / Output Paths
# ------------------------------------------
input_path  = "s3a://dataset-chicago/project-folder/Processed/Stage1/"
output_path = "s3a://dataset-chicago/project-folder/Processed/Final/"

# ------------------------------------------
#  Load Stage1 Preprocessed Data
# ------------------------------------------
df = spark.read.parquet(input_path).dropDuplicates()

# Ensure Stage1 columns exist (created during preprocessing)
# Stage1 guarantees these fields: Beat, Date_TS, CrimeType, Hour, Arrest_Flag, Year, Month, Lat_Bin, Lng_Bin,
# plus many others like IsWeekend, Domestic_Flag, FBI_Code, etc.
required_cols = ["Beat", "Date_TS", "CrimeType", "Hour", "Arrest_Flag", "Year", "Month", "Lat_Bin", "Lng_Bin"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required Stage1 columns: {missing}")

# Robust timestamp & calendar helpers
df = (
    df.withColumn("Date_TS", F.to_timestamp("Date_TS"))
      .withColumn("Date_Only", F.to_date("Date_TS"))
      .withColumn("Timestamp_Unix", F.col("Date_TS").cast("long"))
      # Use first-of-month date for safe ordering/lag (avoids lexicographic errors)
      .withColumn("YearMonthDate", F.trunc("Date_TS", "MONTH"))
)

# ------------------------------------------
#  7‑Day Rolling Crime Count (Beat, time-aware, past-only)
# ------------------------------------------
# Rolling window over last 7*86400 seconds, excluding current event
rolling_window = (
    Window.partitionBy("Beat")
    .orderBy("Timestamp_Unix")
    .rangeBetween(-7 * 86400, -1)
)
df = df.withColumn("Crime_Count_Last7Days", F.count(F.lit(1)).over(rolling_window))

# ------------------------------------------
#  CrimeType–Hour Density P(Hour | CrimeType), past-only
# ------------------------------------------
# Count of prior records for (CrimeType, Hour) divided by prior total for CrimeType
w_ct_total = (
    Window.partitionBy("CrimeType")
    .orderBy("Timestamp_Unix")
    .rowsBetween(Window.unboundedPreceding, -1)
)
w_ct_hour = (
    Window.partitionBy("CrimeType", "Hour")
    .orderBy("Timestamp_Unix")
    .rowsBetween(Window.unboundedPreceding, -1)
)
df = (
    df.withColumn("CrimeType_Total_Prior", F.count(F.lit(1)).over(w_ct_total))
      .withColumn("CrimeType_Hour_Prior",  F.count(F.lit(1)).over(w_ct_hour))
      .withColumn(
          "CrimeType_Hour_Density",
          F.when(F.col("CrimeType_Total_Prior") > 0,
                 F.col("CrimeType_Hour_Prior") / F.col("CrimeType_Total_Prior"))
           .otherwise(F.lit(None).cast("double"))
      )
      .drop("CrimeType_Total_Prior", "CrimeType_Hour_Prior")
)

# ------------------------------------------
#  Monthly Arrest Rate (previous month only, time-aware)
# ------------------------------------------
# Stage1 already created Arrest_Flag; compute per (Beat, YearMonthDate) and lag by 1 month
monthly = (
    df.groupBy("Beat", "YearMonthDate")
      .agg(
          F.count(F.lit(1)).alias("Monthly_Crime_Total"),
          F.sum("Arrest_Flag").alias("Monthly_Arrests")
      )
      .withColumn("Monthly_Arrest_Rate", F.col("Monthly_Arrests") / F.col("Monthly_Crime_Total"))
)
w_month = Window.partitionBy("Beat").orderBy("YearMonthDate")
monthly = monthly.withColumn("Prev_Month_Arrest_Rate", F.lag("Monthly_Arrest_Rate", 1).over(w_month))

df = df.join(
    monthly.select("Beat", "YearMonthDate", "Prev_Month_Arrest_Rate"),
    on=["Beat", "YearMonthDate"],
    how="left"
)

# ------------------------------------------
#  Top CrimeType per Beat — previous month (time-aware & leak-safe)
# ------------------------------------------
# Compute top CrimeType for each (Beat, YearMonthDate), then carry "previous month's top" to current events.
month_ct = (
    df.groupBy("Beat", "YearMonthDate", "CrimeType")
      .agg(F.count(F.lit(1)).alias("ct_count"))
)
rank_win = Window.partitionBy("Beat", "YearMonthDate").orderBy(F.desc("ct_count"), F.asc("CrimeType"))
top_ct = (
    month_ct.withColumn("r", F.row_number().over(rank_win))
            .filter(F.col("r") == 1)
            .select("Beat", "YearMonthDate", F.col("CrimeType").alias("TopCrimeType_Month"))
)
# Prev-month top for leak safety
w_prev_top = Window.partitionBy("Beat").orderBy("YearMonthDate")
top_ct = top_ct.withColumn("TopCrimeType_PrevMonth", F.lag("TopCrimeType_Month", 1).over(w_prev_top))

df = df.join(
    top_ct.select("Beat", "YearMonthDate", "TopCrimeType_PrevMonth"),
    on=["Beat", "YearMonthDate"],
    how="left"
)

# ------------------------------------------
#  Daily Crime Count (Beat, Lat/Lng bins) + PrevDay & Outlier flag (3σ, past-only)
# ------------------------------------------
daily = (
    df.groupBy("Beat", "Lat_Bin", "Lng_Bin", "Date_Only")
      .agg(F.count(F.lit(1)).alias("Daily_Crime_Count"))
)

# Prev day count per Beat (optionally include spatial bins in partition if you want per-cell prev day)
w_prev_day = (
    Window.partitionBy("Beat")
    .orderBy("Date_Only")
    .rowsBetween(-1, -1)
)
daily = daily.withColumn("PrevDay_CrimeCount", F.first("Daily_Crime_Count").over(w_prev_day))

# Historical stats (mean/std) up to previous day for outlier detection
w_hist = (
    Window.partitionBy("Beat")
    .orderBy("Date_Only")
    .rowsBetween(Window.unboundedPreceding, -1)
)
daily = (
    daily.withColumn("hist_avg",    F.avg("Daily_Crime_Count").over(w_hist))
         .withColumn("hist_stddev", F.stddev("Daily_Crime_Count").over(w_hist))
         .withColumn(
             "Is_CrimeOutlier",
             F.when(
                 (F.col("hist_avg").isNotNull()) & (F.col("hist_stddev").isNotNull()) &
                 (F.col("Daily_Crime_Count") > F.col("hist_avg") + 3 * F.col("hist_stddev")),
                 F.lit(True)
             ).otherwise(F.lit(False))
         )
)

df = df.join(
    daily.select("Beat", "Date_Only", "Lat_Bin", "Lng_Bin", "PrevDay_CrimeCount", "Is_CrimeOutlier"),
    on=["Beat", "Date_Only", "Lat_Bin", "Lng_Bin"],
    how="left"
)

# ------------------------------------------
#  Final Cleanup & Save
# ------------------------------------------
# Fill count-like nulls conservatively; keep rates/densities as null if unseen historically
df = df.fillna(0, subset=["Crime_Count_Last7Days", "PrevDay_CrimeCount"])

# Write Athena-optimized partitions (Stage1 used same partitioning)
(
    df.write.mode("overwrite")
      .partitionBy("Year", "Month")
      .parquet(output_path)
)

print(" Saved feature-engineered dataset to:", output_path)
print(" Final row count:", df.count())

spark.stop()
