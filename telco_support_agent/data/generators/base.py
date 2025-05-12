"""Base generator class for data generation.

This module contains the BaseGenerator class which is inherited by all
entity-specific generators.
"""
import random
from datetime import datetime, timedelta
from typing import Any

import pyspark.sql.functions as functions
from faker import Faker
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from telco_support_agent.utils.spark_utils import spark


class BaseGenerator:
    """Base class for all data generators.

    Provides common functionality for data generation, validation, and storage.

    Attributes:
        config: Config dict.
        fake: Faker instance for generating realistic data.
        random: Random instance with configured seed.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the base generator.

        Args:
            config: Config dict.
        """
        self.config = config

        # Set up Faker with the configured seed
        self.fake = Faker()
        Faker.seed(config["seed"])

        # Set up random with the configured seed
        self.random = random
        self.random.seed(config["seed"])

    def generate_id(self, prefix: str, start_num: int, count: int) -> list[str]:
        """Generate IDs with specified prefix and padding.

        Args:
            prefix: Prefix for the ID (e.g., "CUS", "BILL").
            start_num: Starting number.
            count: Number of IDs to generate.

        Returns:
            List of generated IDs.
        """
        if prefix in ["CUS"]:
            # Customers use 5-digit padding
            return [f"{prefix}-{start_num + i:05d}" for i in range(count)]
        elif prefix in ["SUB"]:
            # Subscriptions use 8-digit padding
            return [f"{prefix}-{start_num + i:08d}" for i in range(count)]
        elif prefix in ["BILL"]:
            # Billing uses 10-digit padding
            return [f"{prefix}-{start_num + i:010d}" for i in range(count)]
        elif prefix in ["USG"]:
            # Usage uses 12-digit padding
            return [f"{prefix}-{start_num + i:012d}" for i in range(count)]
        else:
            # Default to 4-digit padding (Plans, Devices, Promotions, KB articles, Tickets)
            return [f"{prefix}-{start_num + i:04d}" for i in range(count)]

    def random_date(self, start_date: str, end_date: str) -> datetime:
        """Generate a random date between start_date and end_date.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            Random datetime between start_date and end_date.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        delta = end - start
        random_days = self.random.randint(0, delta.days)

        return start + timedelta(days=random_days)

    def select_weighted(self, distribution: dict[str, float]) -> str:
        """Select a random key from a dict based on weighted probabilities.

        Args:
            distribution: dict mapping keys to their probability weights.

        Returns:
            Randomly selected key.
        """
        # Ensure probabilities sum to 1
        total = sum(distribution.values())
        normalized_dist = {k: v / total for k, v in distribution.items()}

        # Generate a random number between 0 and 1
        r = self.random.random()

        # Select key based on cumulative probability
        cumulative: float = 0.0  # Explicitly use float type
        for key, prob in normalized_dist.items():
            cumulative += prob
            if r <= cumulative:
                return key

        # Fallback to the last key (should rarely happen due to floating point precision)
        return list(distribution.keys())[-1]

    def save_to_delta(
        self,
        df: DataFrame,
        table_name: str,
        mode: str = "overwrite",
        partition_by: list[str] | None = None,
    ) -> None:
        """Save DataFrame to Delta table.

        Args:
            df: DataFrame to save.
            table_name: Name of the Delta table.
            mode: Write mode (overwrite, append, etc.).
            partition_by: Columns to partition by.
        """
        writer = df.write.format("delta").mode(mode)

        if partition_by:
            writer = writer.partitionBy(*partition_by)

        writer.saveAsTable(table_name)

        print(f"Saved {df.count()} records to {table_name}")

    def create_dataframe_from_schema(
        self, schema: StructType, data: list[tuple]
    ) -> DataFrame:
        """Create a DataFrame from a schema and data.

        Args:
            schema: StructType schema for the DataFrame.
            data: List of tuples containing the data.

        Returns:
            DataFrame created from the schema and data.
        """
        return spark.createDataFrame(data, schema)

    def apply_distribution_udf(self, distribution: dict[str, float]) -> Any:
        """Create a UDF to apply a distribution.

        Args:
            distribution: dict mapping values to probabilities.

        Returns:
            PySpark UDF that returns values according to the distribution.
        """
        # Convert distribution to list of tuples for easier sampling
        items = list(distribution.items())
        values = [item[0] for item in items]
        weights = [item[1] for item in items]

        def sample_from_distribution() -> str:
            return str(self.random.choices(values, weights=weights, k=1)[0])

        return functions.udf(sample_from_distribution)
