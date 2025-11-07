"""
Tests for data pipeline
"""

import pytest
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineering


def test_data_ingestion():
    """Test data ingestion"""
    ingestor = DataIngestion(data_dir="tests/test_data/raw")
    os.makedirs("tests/test_data/raw", exist_ok=True)
    
    # Test sample data creation
    sample_df = ingestor._create_sample_data()
    assert not sample_df.empty
    assert len(sample_df) > 0
    assert 'pts' in sample_df.columns
    assert 'reb' in sample_df.columns


def test_feature_engineering():
    """Test feature engineering"""
    fe = FeatureEngineering(
        data_dir="tests/test_data/raw",
        output_dir="tests/test_data/processed"
    )
    os.makedirs("tests/test_data/processed", exist_ok=True)
    
    # Create sample data
    import numpy as np
    sample_data = pd.DataFrame({
        'player.id': np.random.randint(1, 10, 100),
        'game.id': range(100),
        'min': np.random.uniform(10, 48, 100),
        'pts': np.random.randint(0, 50, 100),
        'reb': np.random.randint(0, 20, 100),
        'ast': np.random.randint(0, 15, 100),
        'stl': np.random.randint(0, 5, 100),
        'blk': np.random.randint(0, 5, 100),
        'turnover': np.random.randint(0, 8, 100),
        'pf': np.random.randint(0, 6, 100),
        'fgm': np.random.randint(0, 20, 100),
        'fga': np.random.randint(0, 25, 100),
        'fg3m': np.random.randint(0, 10, 100),
        'fg3a': np.random.randint(0, 15, 100),
        'ftm': np.random.randint(0, 10, 100),
        'fta': np.random.randint(0, 12, 100),
    })
    
    # Save sample data
    os.makedirs("tests/test_data/raw", exist_ok=True)
    sample_data.to_csv("tests/test_data/raw/test_stats.csv", index=False)
    
    # Test feature engineering
    df_processed = fe.process_pipeline(filename="test_stats.csv")
    
    assert not df_processed.empty
    assert 'fatigue_risk' in df_processed.columns
    assert 'avg_minutes_last_5' in df_processed.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

