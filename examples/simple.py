import pandas as pd

from dfcache import dfcache


@dfcache()
def load_data(file_id: int, filter_col: str = "default") -> pd.DataFrame:
    """Simulate loading data."""
    print(f"Loading data for file_id={file_id}, filter_col={filter_col}")
    return pd.DataFrame(
        {
            "id": range(file_id * 10),
            "value": range(file_id * 10, file_id * 20),
            "filter": [filter_col] * (file_id * 10),
        }
    )


@dfcache(cache_dir="custom_cache")
def process_data(df: pd.DataFrame, multiplier: int = 1) -> pd.DataFrame:
    """Simulate data processing."""
    print(f"Processing data with multiplier={multiplier}")
    result = df.copy()
    result["value"] = result["value"] * multiplier
    return result


class DataProcessor:
    @dfcache
    def transform(self, data: pd.DataFrame, operation: str = "sum") -> pd.DataFrame:
        """Transform data based on operation."""
        print(f"Transforming data with operation={operation}")
        if operation == "sum":
            return data.groupby("filter").sum().reset_index()
        return data


# Test usage
print("Testing dfcache decorator:")

# First call - will execute and cache
df1 = load_data(5, "test")
print(f"Result shape: {df1.shape}")

# Second call - will load from cache
df2 = load_data(5, "test")
print(f"Result shape: {df2.shape}")

# Test with class method
processor = DataProcessor()
result = processor.transform(df1, "sum")
print(f"Transformed result shape: {result.shape}")
