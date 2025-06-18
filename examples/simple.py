import pandas as pd

from cachetto import cached


@cached
def load_data(rows: int = 100) -> pd.DataFrame:
    """Simulate loading data."""
    print(f"Loading '{rows}' rows of data...")
    import time

    time.sleep(1)
    return pd.DataFrame(
        {
            "id": range(rows * 10),
            "value": range(rows * 10, rows * 20),
            "filter": [rows] * (rows * 10),
        }
    )


@cached
def process_data(df: pd.DataFrame, multiplier: int = 1) -> pd.DataFrame:
    """Simulate data processing."""
    print(f"Processing data with multiplier={multiplier}")
    result = df.copy()
    result["value"] = result["value"] * multiplier
    return result


class DataProcessor:
    @cached
    def transform(self, data: pd.DataFrame, operation: str = "sum") -> pd.DataFrame:
        """Transform data based on operation."""
        print(f"Transforming data with operation={operation}")
        if operation == "sum":
            return data.groupby("filter").sum().reset_index()
        return data


# Test usage
print("Testing cached decorator:")

# First call - will execute and cache
df1 = load_data(100)
print(f"Result shape: {df1.shape}")

# Second call - will load from cache
df2 = load_data(100)
print(f"Result shape: {df2.shape}")

df3 = load_data(5)
print(f"Result shape: {df3.shape}")

# Test with class method
processor = DataProcessor()
result = processor.transform(df1, "sum")
print(f"Transformed result shape: {result.shape}")
