from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark(app_name: str = "marketplace-integrity-monitor") -> SparkSession:
    """Create a local Spark session with settings suited for JSON and Parquet batch work."""
    try:
        spark = (
            SparkSession.builder.appName(app_name)
            .master("local[*]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.driver.memory", "6g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.shuffle.partitions", "40")
            .config("spark.sql.sources.commitProtocolClass", "org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol")
            .config("spark.sql.parquet.output.committer.class", "org.apache.parquet.hadoop.ParquetOutputCommitter")
            .config("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
            .getOrCreate()
        )
        spark.range(1).count()
        return spark
    except Exception as exc:
        raise RuntimeError(
            "Spark could not initialize. Confirm Java is installed and JAVA_HOME is set, "
            "then retry from the project root."
        ) from exc

