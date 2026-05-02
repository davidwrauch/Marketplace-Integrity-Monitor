from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


FEATURE_COLUMNS = [
    "stars",
    "review_length",
    "word_count",
    "useful",
    "funny",
    "cool",
    "is_extreme_rating",
    "rating_deviation_from_business_average",
    "reviewer_average_rating",
    "reviewer_review_count",
    "reviewer_extreme_rating_share",
    "reviewer_five_star_share",
    "reviewer_one_star_share",
    "uppercase_share",
    "repeated_character_flag",
    "avg_word_length",
    "generic_praise_flag",
    "html_break_count",
    "text_formatting_anomaly_flag",
]


def build_review_features(reviews: DataFrame, businesses: DataFrame) -> DataFrame:
    business_average = reviews.groupBy("business_id").agg(
        F.avg("stars").alias("business_average_stars")
    )
    reviewer_stats = reviews.groupBy("user_id").agg(
        F.avg("stars").alias("reviewer_average_rating"),
        F.count("*").cast("double").alias("reviewer_review_count"),
        F.avg(F.when(F.col("stars").isin(1.0, 5.0), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
            "reviewer_extreme_rating_share"
        ),
        F.avg(F.when(F.col("stars") == 5.0, F.lit(1.0)).otherwise(F.lit(0.0))).alias(
            "reviewer_five_star_share"
        ),
        F.avg(F.when(F.col("stars") == 1.0, F.lit(1.0)).otherwise(F.lit(0.0))).alias(
            "reviewer_one_star_share"
        ),
    )

    text_col = F.coalesce(F.col("text"), F.lit(""))
    html_break_pattern = r"(?i)<br\s*/?>"
    cleaned_text = F.regexp_replace(text_col, html_break_pattern, "\n")
    normalized_text = F.trim(F.regexp_replace(cleaned_text, r"\s+", " "))
    lower_text = F.lower(normalized_text)
    uppercase_count = F.length(F.regexp_replace(cleaned_text, r"[^A-Z]", ""))
    alpha_count = F.length(F.regexp_replace(cleaned_text, r"[^A-Za-z]", ""))
    html_break_count = (
        F.length(text_col) - F.length(F.regexp_replace(text_col, html_break_pattern, ""))
    ) / F.lit(4.0)
    sentence_split_text = F.regexp_replace(normalized_text, r"[.!?]+", "|")

    return (
        reviews.join(business_average, "business_id", "left")
        .join(reviewer_stats, "user_id", "left")
        .join(businesses, "business_id", "left")
        .withColumn("review_date", F.to_date("review_timestamp"))
        .withColumn("review_month", F.date_trunc("month", F.col("review_timestamp")).cast("date"))
        .withColumn("clean_review_text", cleaned_text)
        .withColumn("review_length", F.length(cleaned_text).cast("double"))
        .withColumn(
            "word_count",
            F.when(F.length(normalized_text) == 0, F.lit(0.0)).otherwise(
                F.size(F.split(normalized_text, " ")).cast("double")
            ),
        )
        .withColumn("avg_word_length", F.col("review_length") / F.greatest(F.col("word_count"), F.lit(1.0)))
        .withColumn("sentence_split_text", sentence_split_text)
        .withColumn(
            "sentence_word_counts",
            F.expr(
                "transform(filter(split(sentence_split_text, '\\\\|'), x -> length(trim(x)) > 0), "
                "x -> cast(size(split(trim(x), ' ')) as double))"
            ),
        )
        .withColumn("sentence_count", F.size(F.col("sentence_word_counts")).cast("double"))
        .withColumn(
            "sentence_count",
            F.when(F.col("word_count") == 0, F.lit(0.0)).otherwise(F.greatest(F.col("sentence_count"), F.lit(1.0))),
        )
        .withColumn("avg_sentence_length", F.col("word_count") / F.greatest(F.col("sentence_count"), F.lit(1.0)))
        .withColumn(
            "sentence_length_variability",
            F.when(F.col("sentence_count") < 2.0, F.lit(0.0)).otherwise(
                F.least(
                    F.sqrt(
                        F.expr(
                            "aggregate(sentence_word_counts, cast(0.0 as double), "
                            "(acc, x) -> acc + pow(x - avg_sentence_length, 2D))"
                        ) / F.greatest(F.col("sentence_count"), F.lit(1.0))
                    ) / F.greatest(F.col("avg_sentence_length"), F.lit(1.0)),
                    F.lit(1.0),
                )
            ),
        )
        .withColumn("uppercase_share", uppercase_count.cast("double") / F.greatest(alpha_count.cast("double"), F.lit(1.0)))
        .withColumn(
            "repeated_character_flag",
            F.when(cleaned_text.rlike(r"(.)\1{4,}"), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "generic_praise_flag",
            F.when(
                lower_text.rlike(r"\b(great product|love it|works great|highly recommend|excellent product|perfect|five stars)\b"),
                F.lit(1.0),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("html_break_count", F.greatest(html_break_count.cast("double"), F.lit(0.0)))
        .withColumn(
            "text_formatting_anomaly_flag",
            F.when(
                (F.col("html_break_count") >= 4.0)
                | ((F.col("avg_word_length") >= 18.0) & (F.col("review_length") >= 60.0))
                | (cleaned_text.rlike(r"\n\s*\n\s*\n")),
                F.lit(1.0),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("no_space_ratio", F.least(F.col("avg_word_length") / F.lit(25.0), F.lit(1.0)))
        .withColumn(
            "first_person_flag",
            F.when(lower_text.rlike(r"\b(i|me|my|mine|we|our|ours)\b"), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "concrete_detail_flag",
            F.when(
                lower_text.rlike(
                    r"\b(skin|hair|face|scent|smell|bottle|package|packaging|color|size|"
                    r"shade|brush|cream|lotion|shampoo|makeup|mascara|used|using|"
                    r"week|weeks|month|months|morning|night|dry|oily|sensitive)\b"
                ),
                F.lit(1.0),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "balanced_pros_cons_flag",
            F.when(lower_text.rlike(r"\b(but|however|although|pros|cons|downside|except)\b"), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "specific_experience_flag",
            F.when(lower_text.rlike(r"\b(used|using|bought|purchased|tried|after|before|for my|on my|arrived)\b"), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "repeated_promotional_language_flag",
            F.when(
                lower_text.rlike(r"\b(amazing product|must buy|best product|buy it|worth every penny|five stars|highly recommend)\b"),
                F.lit(1.0),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "synthetic_templated_language_score",
            F.greatest(
                F.lit(0.0),
                F.least(
                    F.lit(1.0),
                    0.20 * F.col("generic_praise_flag")
                    + 0.15 * F.when((F.col("stars") == 5.0) & (F.col("word_count") <= 12.0), F.lit(1.0)).otherwise(F.lit(0.0))
                    + 0.15 * F.col("repeated_promotional_language_flag")
                    + 0.15 * (F.lit(1.0) - F.col("concrete_detail_flag"))
                    + 0.10 * F.when((F.col("sentence_count") >= 2.0) & (F.col("sentence_length_variability") <= 0.10), F.lit(1.0)).otherwise(F.lit(0.0))
                    + 0.10 * F.col("text_formatting_anomaly_flag")
                    + 0.05 * F.col("repeated_character_flag")
                    + 0.05 * F.col("no_space_ratio")
                    - 0.10 * F.col("first_person_flag")
                    - 0.15 * F.col("concrete_detail_flag")
                    - 0.10 * F.col("balanced_pros_cons_flag")
                    - 0.10 * F.col("specific_experience_flag"),
                ),
            ),
        )
        .withColumn(
            "is_extreme_rating",
            F.when(F.col("stars").isin(1.0, 5.0), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "rating_deviation_from_product_average",
            (F.col("stars") - F.col("business_average_stars")).cast("double"),
        )
        .withColumn(
            "rating_deviation_from_business_average",
            F.abs(F.col("rating_deviation_from_product_average")).cast("double"),
        )
        .fillna(
            {
                "useful": 0.0,
                "funny": 0.0,
                "cool": 0.0,
                "review_length": 0.0,
                "word_count": 0.0,
                "avg_word_length": 0.0,
                "uppercase_share": 0.0,
                "repeated_character_flag": 0.0,
                "generic_praise_flag": 0.0,
                "html_break_count": 0.0,
                "text_formatting_anomaly_flag": 0.0,
                "no_space_ratio": 0.0,
                "is_extreme_rating": 0.0,
                "rating_deviation_from_product_average": 0.0,
                "rating_deviation_from_business_average": 0.0,
                "reviewer_average_rating": 0.0,
                "reviewer_review_count": 0.0,
                "reviewer_extreme_rating_share": 0.0,
                "reviewer_five_star_share": 0.0,
                "reviewer_one_star_share": 0.0,
                "sentence_count": 0.0,
                "avg_sentence_length": 0.0,
                "sentence_length_variability": 0.0,
                "first_person_flag": 0.0,
                "concrete_detail_flag": 0.0,
                "balanced_pros_cons_flag": 0.0,
                "specific_experience_flag": 0.0,
                "repeated_promotional_language_flag": 0.0,
                "synthetic_templated_language_score": 0.0,
            }
        )
        .drop("sentence_split_text", "sentence_word_counts")
    )