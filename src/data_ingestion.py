"""
Data Ingestion Module
Fetches NBA player statistics from balldontlie.io API
"""

import requests
import pandas as pd
import os
from datetime import datetime
from typing import List, Optional
import time


class DataIngestion:
    """Handles data collection from balldontlie.io API"""
    
    BASE_URL = "https://www.balldontlie.io/api/v1"
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data ingestion
        
        Args:
            data_dir: Directory to save raw data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_player_stats(self, 
                          player_ids: Optional[List[int]] = None,
                          seasons: Optional[List[int]] = None,
                          per_page: int = 100) -> pd.DataFrame:
        """
        Fetch player statistics from API
        
        Args:
            player_ids: List of player IDs to fetch (None = all players)
            seasons: List of seasons to fetch (e.g., [2023, 2024])
            per_page: Number of records per page
            
        Returns:
            DataFrame with player statistics
        """
        all_data = []
        page = 1
        
        # Build URL
        url = f"{self.BASE_URL}/stats"
        params = {"per_page": per_page}
        
        if player_ids:
            for pid in player_ids:
                params[f"player_ids[]"] = pid
        if seasons:
            for season in seasons:
                params[f"seasons[]"] = season
        
        print(f"Fetching data from {self.BASE_URL}/stats...")
        
        while True:
            params["page"] = page
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("data"):
                    break
                
                all_data.extend(data["data"])
                
                # Check if there are more pages
                meta = data.get("meta", {})
                if page >= meta.get("total_pages", 1):
                    break
                
                page += 1
                time.sleep(0.5)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                break
        
        if not all_data:
            print("No data fetched. Creating sample data for testing...")
            return self._create_sample_data()
        
        # Normalize JSON data to DataFrame
        df = pd.json_normalize(all_data)
        return df
    
    def fetch_players(self) -> pd.DataFrame:
        """
        Fetch list of all players
        
        Returns:
            DataFrame with player information
        """
        url = f"{self.BASE_URL}/players"
        all_players = []
        page = 1
        
        print("Fetching player list...")
        
        while True:
            try:
                response = requests.get(url, params={"per_page": 100, "page": page}, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("data"):
                    break
                
                all_players.extend(data["data"])
                
                meta = data.get("meta", {})
                if page >= meta.get("total_pages", 1):
                    break
                
                page += 1
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching players: {e}")
                break
        
        if not all_players:
            return pd.DataFrame()
        
        return pd.json_normalize(all_players)
    
    def save_data(self, df: pd.DataFrame, filename: Optional[str] = None):
        """
        Save DataFrame to CSV
        
        Args:
            df: DataFrame to save
            filename: Optional filename (default: stats_YYYYMMDD.csv)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"stats_{timestamp}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample data for testing when API is unavailable
        
        Returns:
            Sample DataFrame with player statistics
        """
        import numpy as np
        
        n_samples = 1000
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        sample_data = {
            'id': range(1, n_samples + 1),
            'game.id': np.random.randint(1, 100, n_samples),
            'player.id': np.random.randint(1, 50, n_samples),
            'min': np.random.uniform(10, 48, n_samples).round(1),
            'pts': np.random.randint(0, 50, n_samples),
            'reb': np.random.randint(0, 20, n_samples),
            'ast': np.random.randint(0, 15, n_samples),
            'stl': np.random.randint(0, 5, n_samples),
            'blk': np.random.randint(0, 5, n_samples),
            'turnover': np.random.randint(0, 8, n_samples),
            'pf': np.random.randint(0, 6, n_samples),
            'fgm': np.random.randint(0, 20, n_samples),
            'fga': np.random.randint(0, 25, n_samples),
            'fg3m': np.random.randint(0, 10, n_samples),
            'fg3a': np.random.randint(0, 15, n_samples),
            'ftm': np.random.randint(0, 10, n_samples),
            'fta': np.random.randint(0, 12, n_samples),
        }
        
        df = pd.DataFrame(sample_data)
        return df


def main():
    """Main function for data ingestion"""
    ingestor = DataIngestion()
    
    # Fetch recent season data
    print("Fetching NBA player statistics...")
    df_stats = ingestor.fetch_player_stats(seasons=[2023, 2024])
    
    if not df_stats.empty:
        print(f"Fetched {len(df_stats)} records")
        ingestor.save_data(df_stats)
    else:
        print("Using sample data for development")
        df_stats = ingestor._create_sample_data()
        ingestor.save_data(df_stats, "sample_stats.csv")


if __name__ == "__main__":
    main()

