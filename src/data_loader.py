"""
Data loading and preprocessing module for Juror-AI Comparison Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

def load_data(filepath: str = "data/datacsv.csv") -> pd.DataFrame:
    """
    Load the juror-AI comparison dataset from CSV.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded {len(df)} rows from {filepath}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File {filepath} not found")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def validate_data_structure(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate the dataset structure and report basic statistics.
    
    Args:
        df: The loaded dataset
        
    Returns:
        dict: Validation report with statistics
    """
    
    # Expected columns based on actual data structure
    expected_cols = {
        'Scenario': 'object',
        'Sex': 'object', 
        'Ethnicity': 'object',
        'Political Affiliation': 'object',
        'Economic Status of Household': 'object',
        'Educational Attainment': 'object',
        'Human rating': 'float64',
        'gpt-4.1_Scale Score': 'float64',
        'claude-sonnet-4-20250514_Scale Score': 'float64',
        'gemini-2.5-flash_Scale Score': 'float64'
    }
    
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_columns': [],
        'null_counts': {},
        'scenario_counts': {},
        'value_ranges': {}
    }
    
    # Check for missing expected columns
    for col in expected_cols.keys():
        if col not in df.columns:
            report['missing_columns'].append(col)
    
    # Count nulls per column
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            report['null_counts'][col] = null_count
    
    # Count scenarios
    if 'Scenario' in df.columns:
        report['scenario_counts'] = df['Scenario'].value_counts().to_dict()
    
    # Check rating value ranges
    rating_cols = ['Human rating', 'gpt-4.1_Scale Score', 
                   'claude-sonnet-4-20250514_Scale Score', 'gemini-2.5-flash_Scale Score']
    
    for col in rating_cols:
        if col in df.columns:
            report['value_ranges'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
    
    return report

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset for analysis.
    
    Args:
        df: Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Rename columns for easier analysis
    column_mapping = {
        'Human rating': 'Human',
        'gpt-4.1_Scale Score': 'OpenAI', 
        'claude-sonnet-4-20250514_Scale Score': 'Claude',
        'gemini-2.5-flash_Scale Score': 'Gemini',
        'Sex': 'Gender',
        'Political Affiliation': 'Party'
    }
    
    df_clean = df_clean.rename(columns=column_mapping)
    
    # Ensure categorical ordering for scenarios
    scenario_categories = [
        "Explicit Name", 
        "Blank Space", 
        "Other Person", 
        "Inferentially Incriminating"
    ]
    
    if 'Scenario' in df_clean.columns:
        df_clean['Scenario'] = pd.Categorical(
            df_clean['Scenario'], 
            categories=scenario_categories, 
            ordered=True
        )
    
    # Keep all demographic data (no cleaning for full dataset)
    
    # Add a unique identifier for each row (juror response)
    df_clean['JurorID'] = range(1, len(df_clean) + 1)
    
    # Drop rows with missing rating data
    rating_cols = ['Human', 'OpenAI', 'Claude', 'Gemini']
    before_count = len(df_clean)
    df_clean = df_clean.dropna(subset=rating_cols)
    after_count = len(df_clean)
    
    if before_count != after_count:
        print(f"⚠️  Dropped {before_count - after_count} rows with missing rating data")
    
    print(f"✅ Final dataset: {len(df_clean):,} responses (full dataset preserved)")
    
    return df_clean

def melt_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform data from wide to long format for easier analysis and plotting.
    
    Args:
        df: Preprocessed dataset in wide format
        
    Returns:
        pd.DataFrame: Dataset in long format
    """
    
    # Identify demographic and ID columns
    id_vars = ['JurorID', 'Scenario', 'Gender', 'Ethnicity', 'Party', 
               'Economic Status of Household', 'Educational Attainment']
    
    # AI model columns (excluding Human which will be used as baseline)
    value_vars = ['OpenAI', 'Claude', 'Gemini']
    
    # Melt the dataframe
    df_long = pd.melt(
        df, 
        id_vars=id_vars + ['Human'],  # Include Human as separate column for comparison
        value_vars=value_vars,
        var_name='AI_Model',
        value_name='AI_Rating'
    )
    
    return df_long

def get_sample_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Get a random sample of the data for spot-checking.
    
    Args:
        df: Dataset to sample from
        n: Number of rows to sample
        
    Returns:
        pd.DataFrame: Random sample
    """
    return df.sample(n=min(n, len(df)), random_state=42)

def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a comprehensive summary of the dataset.
    
    Args:
        df: Dataset to summarize
    """
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    
    validation_report = validate_data_structure(df)
    
    print(f"📋 Total Rows: {validation_report['total_rows']:,}")
    print(f"📋 Total Columns: {validation_report['total_columns']}")
    
    if validation_report['missing_columns']:
        print(f"⚠️  Missing Expected Columns: {validation_report['missing_columns']}")
    else:
        print("✅ All expected columns present")
    
    if validation_report['null_counts']:
        print(f"\n🕳️  Null Values Found:")
        for col, count in validation_report['null_counts'].items():
            print(f"   • {col}: {count}")
    else:
        print("✅ No null values found")
    
    print(f"\n📈 Scenario Distribution:")
    for scenario, count in validation_report['scenario_counts'].items():
        print(f"   • {scenario}: {count}")
    
    print(f"\n🎯 Rating Ranges:")
    for col, stats in validation_report['value_ranges'].items():
        print(f"   • {col}: {stats['min']:.1f} - {stats['max']:.1f} (μ={stats['mean']:.1f}, σ={stats['std']:.1f})")
    
    print("="*60) 