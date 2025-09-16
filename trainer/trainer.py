"""
Train an ALS model with PySpark and export item (movie) factors + mapping table.

Outputs to `--out` directory:
 - movie_factors.npy    # NumPy array shape (N_movies, rank)
 - movie_id_index.csv   # columns: movieId,index,original_title
 - optionally: als_model (Spark model)
"""

import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
import numpy as np
import pandas as pd


def train(movies_csv, ratings_csv, out_dir, rank=20, regParam=0.1, maxIter=10):
    spark = SparkSession.builder.appName("movie-als-training").getOrCreate()

    # Load movies metadata
    movies_raw = spark.read.option("header", True).csv(movies_csv)

    # Filter rows where `id` is only digits before casting
    movies = (
        movies_raw
        .filter(col("id").rlike("^[0-9]+$"))  # keep only valid numeric IDs
        .select(
            col("id").cast("int").alias("movieId"),
            col("original_title")
        )
        .dropna()
    )

    # Load ratings and cast types
    ratings = (
        spark.read.option("header", True).csv(ratings_csv)
        .select(
            col("userId").cast("int"),
            col("movieId").cast("int"),
            col("rating").cast("double"),
        )
        .dropna()
    )

    # Keep only ratings where movieId exists in movies
    movies_ids = movies.select("movieId").distinct()
    ratings = ratings.join(movies_ids, on="movieId", how="inner")

    # Repartition for faster work locally
    ratings = ratings.repartition(8)

    # Train ALS
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        maxIter=maxIter,
        regParam=regParam,
        coldStartStrategy="drop",
        nonnegative=True,
    )

    model = als.fit(ratings)

    # Extract item factors
    item_factors = model.itemFactors  # columns: id, features

    # Convert to pandas and save as numpy array
    item_pd = item_factors.toPandas()
    # Ensure id column is int
    item_pd["id"] = item_pd["id"].astype(int)

    # Join titles so we can build an ordered mapping
    movies_pd = movies.toPandas()
    movies_pd["movieId"] = movies_pd["movieId"].astype(int)

    merged = pd.merge(movies_pd, item_pd, left_on="movieId", right_on="id", how="inner")

    # Create deterministic index (0..N-1) so numpy array rows align with this csv
    merged = merged.sort_values("movieId").reset_index(drop=True)
    merged["index"] = np.arange(len(merged))

    features = np.vstack(merged["features"].tolist())

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "movie_factors.npy"), features)

    # Save mapping
    mapping_df = merged[["movieId", "index", "original_title"]]
    mapping_df.to_csv(os.path.join(out_dir, "movie_id_index.csv"), index=False)

    # Save Spark model (optional)
    try:
        model_path = os.path.join(out_dir, "als_model")
        model.write().overwrite().save(model_path)
    except Exception as e:
        print("Warning: could not save Spark model:", e)

    spark.stop()
    print("Training finished. Artifacts saved to:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies", required=True)
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    train(args.movies, args.ratings, args.out, rank=args.rank, regParam=args.reg, maxIter=args.iters)
