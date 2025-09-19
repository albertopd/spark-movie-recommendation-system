
"""
trainer.py
-----------
Train an ALS (Alternating Least Squares) collaborative filtering model using PySpark on the MovieLens dataset.
Exports movie factor vectors and a mapping table for use in the recommendation web app.

Outputs to the specified directory:
 - movie_factors.npy: NumPy array of movie factor vectors (N_movies, rank)
 - movie_id_index.csv: CSV mapping of movieId, index, and title
 - als_model/: (optional) Exported Spark ALS model
"""

import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
import numpy as np
import pandas as pd


MOVIE_FACTORS_NPY = "movie_factors.npy"
MOVIE_INDEX_CSV = "movie_index_map.csv"


def train(movies_csv, ratings_csv, out_dir, rank=10, regParam=0.1, maxIter=10):
    """
    Train an ALS model on the given ratings and movies data, and export model artifacts.

    Args:
        movies_csv (str): Path to movies.csv file (must have columns: movieId, title)
        ratings_csv (str): Path to ratings.csv file (must have columns: userId, movieId, rating)
        out_dir (str): Output directory for artifacts
        rank (int): Number of latent factors for ALS
        regParam (float): Regularization parameter for ALS
        maxIter (int): Number of ALS training iterations

    Outputs:
        - movie_factors.npy: Numpy array of movie factor vectors
        - movie_id_index.csv: Mapping of movieId, index, and title
        - als_model/: (optional) Exported Spark ALS model
    """
    spark = SparkSession.builder \
        .appName("movie-als-training") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
        .getOrCreate()

    # Load movies metadata
    movies = (
        spark.read.option("header", True).csv(movies_csv)
        .select(
            col("movieId").cast("int"),
            col("title").cast("string")
        )
        .dropDuplicates(["movieId"])
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
    ratings = ratings.repartition(200)

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
        checkpointInterval=10
    )

    model = als.fit(ratings)

    # Extract item factors and convert to pandas
    item_factors = model.itemFactors  # columns: id, features
    item_factors_pd = item_factors.toPandas()

    # Join titles so we can build an ordered mapping
    movies_pd = movies.toPandas()
    merged = pd.merge(movies_pd, item_factors_pd, left_on="movieId", right_on="id", how="inner")

    # Create deterministic index (0..N-1) so numpy array rows align with this csv
    merged = merged.sort_values("movieId").reset_index(drop=True)
    merged["index"] = np.arange(len(merged))

    features = np.vstack(merged["features"].tolist())

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, MOVIE_FACTORS_NPY), features)

    # Save mapping
    mapping_df = merged[["title", "index"]]
    mapping_df.to_csv(os.path.join(out_dir, MOVIE_INDEX_CSV), index=False)

    # Save Spark model
    try:
        model_path = os.path.join(out_dir, "als_model")
        model.write().overwrite().save(model_path)
    except Exception as e:
        print("Warning: could not save Spark model:", e)

    spark.stop()
    print("Training finished. Artifacts saved to:", out_dir)


if __name__ == "__main__":
    """
    Command-line interface for training the ALS model.
    Example usage:
        python trainer.py --movies data/movies.csv --ratings data/ratings.csv --out artifacts/
    """
    parser = argparse.ArgumentParser(description="Train ALS model and export artifacts for movie recommendation system.")
    parser.add_argument("--movies", required=True, help="Path to movies.csv")
    parser.add_argument("--ratings", required=True, help="Path to ratings.csv")
    parser.add_argument("--out", default="artifacts", help="Output directory for artifacts")
    parser.add_argument("--rank", type=int, default=20, help="ALS rank (number of latent factors)")
    parser.add_argument("--reg", type=float, default=0.1, help="ALS regularization parameter")
    parser.add_argument("--iters", type=int, default=10, help="ALS max iterations")
    args = parser.parse_args()

    train(args.movies, args.ratings, args.out, rank=args.rank, regParam=args.reg, maxIter=args.iters)
