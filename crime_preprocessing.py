# ============================================================
#  Chicago Crime — Stage 1 Preprocessing (PySpark, S3-ready)
# - No Python UDFs (vectorized Spark only)
# - No data leakage in rolling/window features
# - Safe column guards; robust timestamp parsing
# - Athena/QuickSight-friendly Parquet (snappy, partitioned)
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, upper, trim, coalesce, to_timestamp, year, month, hour, dayofweek,
    count, mean, round as sround, concat_ws, instr, regexp_replace
)
from pyspark.sql.window import Window

# ========== 1) Spark ==========
spark = SparkSession.builder \
    .appName("CrimeDataPreprocessing") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

# ---- Paths (edit if needed) ----
input_path  = "s3a://dataset-chicago/project-folder/Dataset/dataset.csv"    # use s3a
output_path = "s3a://dataset-chicago/project-folder/Processed/Stage1/"

# ========== 2) Load ==========
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Utility: guard for optional columns
def has(cname: str) -> bool:
    return cname in df.columns

# ========== 3) Drop sparse/irrelevant ==========
sparse_cols = ["Ward", "Community Area"]
drop_cols   = ["ID", "Case Number", "Block", "IUCR", "X Coordinate", "Y Coordinate"]

df = df.drop(*[c for c in sparse_cols if has(c)])
df = df.drop(*[c for c in drop_cols   if has(c)])
df = df.dropDuplicates()

# ========== 4) Clean Description ==========
if has("Description"):
    df = df.withColumn("Description", upper(trim(col("Description"))))

# ========== 5) Timestamp parse + time parts ==========
if not has("Date"):
    raise ValueError("Missing required column: 'Date'")

df = df.withColumn(
    "Date_TS",
    coalesce(
        to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a"),
        to_timestamp(col("Date"), "yyyy-MM-dd HH:mm:ss")
    )
)
df = df.filter(col("Date_TS").isNotNull())

df = df.withColumn("Year",    year("Date_TS")) \
       .withColumn("Month",   month("Date_TS")) \
       .withColumn("Hour",    hour("Date_TS")) \
       .withColumn("Weekday", dayofweek("Date_TS")) \
       .withColumn("IsWeekend", when(col("Weekday").isin(1,7), lit(1)).otherwise(lit(0)))

# ========== 6) Season & TimeSlot ==========
df = df.withColumn(
    "Season",
    when(col("Month").isin(12,1,2),  lit("Winter"))
    .when(col("Month").isin(3,4,5),  lit("Spring"))
    .when(col("Month").isin(6,7,8),  lit("Summer"))
    .otherwise(lit("Fall"))
)

df = df.withColumn(
    "TimeSlot",
    when((col("Hour") <= 5) | (col("Hour") >= 23), lit("Night"))
    .when(col("Hour").between(6,11),               lit("Morning"))
    .when(col("Hour").between(12,17),              lit("Afternoon"))
    .otherwise(lit("Evening"))
)

# ========== 7) YearGroup & Special Eras ==========
df = df.withColumn(
    "YearGroup",
    when(col("Year").between(2001,2005), lit("2001–2005"))
    .when(col("Year").between(2006,2010), lit("2006–2010"))
    .when(col("Year").between(2011,2015), lit("2011–2015"))
    .when(col("Year").between(2016,2020), lit("2016–2020"))
    .otherwise(lit("2021–2025"))
)

df = df.withColumn("Is_Covid_Era",  when(col("Year").isin(2020, 2021), lit(1)).otherwise(lit(0))) \
       .withColumn("Is_Post_Reform", when(col("Year") >= 2016,          lit(1)).otherwise(lit(0)))

# ========== 8) PrimaryType & CrimeType ==========
if has("Primary Type"):
    df = df.withColumn("PrimaryType", upper(trim(col("Primary Type"))))
elif has("PrimaryType"):
    df = df.withColumn("PrimaryType", upper(trim(col("PrimaryType"))))
else:
    raise ValueError("Missing required column: 'Primary Type' (or 'PrimaryType').")

top_crimes = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'ASSAULT', 'NARCOTICS',
              'BURGLARY', 'MOTOR VEHICLE THEFT', 'ROBBERY', 'DECEPTIVE PRACTICE']

df = df.withColumn(
    "CrimeType",
    when(col("PrimaryType").isin(*top_crimes), col("PrimaryType")).otherwise(lit("OTHER"))
)

# ========== 9) CrimeType frequency (prior-only) ==========
crime_w = Window.partitionBy("CrimeType") \
                .orderBy(col("Date_TS").cast("long")) \
                .rowsBetween(Window.unboundedPreceding, -1)
df = df.withColumn("CrimeType_Count", count(lit(1)).over(crime_w))

# ========== 10) Description_Grouped (SAFE rules only) ==========
if has("Description"):
    desc_u = upper(trim(col("Description")))
    df = df.withColumn(
        "Description_Grouped",
        when(instr(desc_u, "UNDER") > 0,    lit("UNDER_500"))
        .when(instr(desc_u, "OVER") > 0,    lit("OVER_500"))
        .when(instr(desc_u, "DOMESTIC") > 0,lit("DOMESTIC"))
        .otherwise(lit("OTHER"))
    )
else:
    df = df.withColumn("Description_Grouped", lit("OTHER"))

# ========== 11) Location grouping ==========
loc_src = "Location Description" if has("Location Description") else ("Location_Description" if has("Location_Description") else None)
if loc_src:
    df = df.withColumn("Location_Description", coalesce(upper(trim(col(loc_src))), lit("UNKNOWN")))
    if loc_src != "Location_Description":
        df = df.drop(loc_src)
else:
    df = df.withColumn("Location_Description", lit("UNKNOWN"))

top_locs = [
    "STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "PARKING LOT/GARAGE(NON.RESID.)",
    "GAS STATION", "SCHOOL - PUBLIC", "RESTAURANT", "CTA PLATFORM", "ALLEY"
]
df = df.withColumn(
    "Location_Grouped",
    when(col("Location_Description").isin(*top_locs), col("Location_Description")).otherwise(lit("OTHER"))
)

# ========== 12) Binary flags & context ==========
def to_int_flag(src_col: str, out_col: str):
    global df
    if has(src_col):
        df = df.withColumn(src_col, regexp_replace(upper(trim(col(src_col))), "TRUE", "1"))
        df = df.withColumn(src_col, regexp_replace(upper(trim(col(src_col))), "FALSE", "0"))
        df = df.withColumn(out_col, col(src_col).cast("int"))
    else:
        df = df.withColumn(out_col, lit(0))

to_int_flag("Arrest",   "Arrest_Flag")
to_int_flag("Domestic", "Domestic_Flag")

df = df.withColumn("Crime_Context", concat_ws("_", col("CrimeType"), col("Domestic_Flag")))

# ========== 13) Arrest rate by (Beat, CrimeType) — prior-only ==========
if not has("Beat"):
    raise ValueError("Missing required column: 'Beat'.")

arrest_w = Window.partitionBy("Beat", "CrimeType") \
                 .orderBy(col("Date_TS").cast("long")) \
                 .rowsBetween(Window.unboundedPreceding, -1)

df = df.withColumn(
    "Arrest_Rate_Percent",
    sround(mean(col("Arrest_Flag")).over(arrest_w) * 100.0, 2)
)

# ========== 14) Domestic rate by Beat — prior-only ==========
dom_w = Window.partitionBy("Beat") \
              .orderBy(col("Date_TS").cast("long")) \
              .rowsBetween(Window.unboundedPreceding, -1)

df = df.withColumn(
    "Domestic_Crime_Rate_Percent",
    sround(mean(col("Domestic_Flag")).over(dom_w) * 100.0, 2)
)

# ========== 15) FBI code features (safe .like prefix checks) ==========
if has("FBI Code"):
    df = df.withColumn("FBI_Code", upper(trim(col("FBI Code"))))
elif has("FBI_Code"):
    df = df.withColumn("FBI_Code", upper(trim(col("FBI_Code"))))
else:
    df = df.withColumn("FBI_Code", lit("UNKNOWN"))

df = df.withColumn(
    "FBI_Category",
    when(col("FBI_Code").like("06%"),   lit("THEFT"))
    .when(col("FBI_Code").like("08B%"), lit("ASSAULT"))
    .when(col("FBI_Code").like("14%"),  lit("VANDALISM"))
    .when(col("FBI_Code").like("18%"),  lit("DRUG VIOLATION"))
    .when(col("FBI_Code").like("26%"),  lit("WEAPONS"))
    .when(col("FBI_Code").like("01A%"), lit("HOMICIDE"))
    .otherwise(lit("OTHER"))
)

fbi_w = Window.partitionBy("FBI_Code") \
              .orderBy(col("Date_TS").cast("long")) \
              .rowsBetween(Window.unboundedPreceding, -1)

df = df.withColumn("FBI_Code_Count", count(lit(1)).over(fbi_w)) \
       .withColumn("FBI_Arrest_Rate", sround(mean(col("Arrest_Flag")).over(fbi_w) * 100.0, 2))

# ========== 16) Geo bins ==========
if has("Latitude") and has("Longitude"):
    df = df.filter(col("Latitude").isNotNull() & col("Longitude").isNotNull())
    df = df.withColumn("Lat_Bin", sround(col("Latitude"), 2)) \
           .withColumn("Lng_Bin", sround(col("Longitude"), 2))
else:
    df = df.withColumn("Lat_Bin", lit(None).cast("double")) \
           .withColumn("Lng_Bin", lit(None).cast("double"))

# ========== 17) Null handling for counts ==========
count_cols = [c for c in ["CrimeType_Count", "FBI_Code_Count"] if has(c)]
if count_cols:
    df = df.fillna(0, subset=count_cols)

# Defensive dedup
df = df.dropDuplicates()

# ========== 18) Save to Parquet (partitioned) ==========
df.write.mode("overwrite").partitionBy("Year", "Month").parquet(output_path)

print(" Preprocessing complete. Final row count:", df.count())
spark.stop()
