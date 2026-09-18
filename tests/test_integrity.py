import pandas as pd
from src.integrity import DataSchema, IntegrityGuard

def test_hash_is_deterministic(sample_df):
    guard = IntegrityGuard()
    assert guard.get_data_hash(sample_df) == guard.get_data_hash(sample_df.copy())

def test_schema_validation_reports_missing_column(sample_df):
    schema = DataSchema(columns={"age": "int64"}, required_columns=["age", "missing"])
    report = IntegrityGuard().validate_schema(sample_df, schema)
    assert report.schema_valid is False
    assert report.row_count == len(sample_df)

def test_schema_range_warning(sample_df):
    schema = DataSchema(
        columns={"age": str(sample_df.age.dtype)},
        required_columns=["age"],
        value_ranges={"age": (0, 10)},
    )
    report = IntegrityGuard().validate_schema(sample_df, schema)
    assert any("Range violation" in warning for warning in report.warnings)
