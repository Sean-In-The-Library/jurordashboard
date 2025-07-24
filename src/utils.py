"""
Utility functions for Juror-AI Comparison Analysis
"""

import pandas as pd
import numpy as np
import base64
import io
from typing import Dict, List, Optional, Tuple

def create_download_link(df: pd.DataFrame, filename: str, text: str = "Download CSV") -> str:
    """
    Generate a download link for a DataFrame as CSV.
    
    Args:
        df: DataFrame to download
        filename: Name of the file
        text: Link text to display
        
    Returns:
        str: HTML string for download link
    """
    
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def format_metrics_for_display(metrics_df: pd.DataFrame, precision: int = 3) -> pd.DataFrame:
    """
    Format metrics DataFrame for better display in UI.
    
    Args:
        metrics_df: DataFrame with metrics
        precision: Number of decimal places
        
    Returns:
        pd.DataFrame: Formatted DataFrame
    """
    
    display_df = metrics_df.copy()
    
    # Round numeric columns
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        display_df[col] = display_df[col].round(precision)
    
    return display_df

def get_top_performers(metrics_dict: Dict[str, pd.DataFrame], metric: str = 'MAE', 
                      top_n: int = 3) -> Dict[str, List[str]]:
    """
    Get top performing models across different analysis dimensions.
    
    Args:
        metrics_dict: Dictionary of metrics from compute_all_metrics
        metric: Metric to use for ranking (lower is better for MAE/MSE)
        top_n: Number of top performers to return
        
    Returns:
        dict: Top performers for each analysis dimension
    """
    
    top_performers = {}
    
    for analysis_type, df in metrics_dict.items():
        if metric in df.columns:
            # For MAE/MSE, lower is better
            if metric in ['MAE', 'MSE']:
                sorted_df = df.sort_values(metric)
            else:
                # For other metrics, higher might be better
                sorted_df = df.sort_values(metric, ascending=False)
            
            if 'Model' in sorted_df.columns:
                top_models = sorted_df['Model'].head(top_n).tolist()
            else:
                top_models = sorted_df.index[:top_n].tolist()
            
            top_performers[analysis_type] = top_models
    
    return top_performers

def validate_streamlit_inputs(scenarios: List[str], parties: List[str], 
                             models: List[str], df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate user inputs from Streamlit interface.
    Human ratings are treated as baseline (not selectable).
    
    Args:
        scenarios: Selected scenarios
        parties: Selected parties
        models: Selected AI models (excluding Human baseline)
        df: Original dataset
        
    Returns:
        tuple: (is_valid, error_message)
    """
    
    if not scenarios:
        return False, "Please select at least one scenario."
    
    if not models:
        return False, "Please select at least one AI model to compare against Human baseline."
    
    # Ensure Human is not included in models (it's the baseline)
    if 'Human' in models:
        return False, "Human ratings are the baseline - please select only AI models to compare."
    
    # Check if selections exist in data
    available_scenarios = df['Scenario'].unique().tolist()
    invalid_scenarios = [s for s in scenarios if s not in available_scenarios]
    if invalid_scenarios:
        return False, f"Invalid scenarios: {invalid_scenarios}"
    
    if 'Party' in df.columns:
        available_parties = df['Party'].unique().tolist()
        invalid_parties = [p for p in parties if p not in available_parties]
        if invalid_parties:
            return False, f"Invalid parties: {invalid_parties}"
    
    available_models = [col for col in ['OpenAI', 'Claude', 'Gemini'] if col in df.columns]
    invalid_models = [m for m in models if m not in available_models]
    if invalid_models:
        return False, f"Invalid AI models: {invalid_models}"
    
    return True, ""

def filter_dataframe(df: pd.DataFrame, scenarios: List[str] = None, 
                    parties: List[str] = None, **kwargs) -> pd.DataFrame:
    """
    Filter DataFrame based on multiple criteria.
    
    Args:
        df: DataFrame to filter
        scenarios: List of scenarios to include
        parties: List of parties to include
        **kwargs: Additional filtering criteria
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    
    filtered_df = df.copy()
    
    if scenarios:
        filtered_df = filtered_df[filtered_df['Scenario'].isin(scenarios)]
    
    if parties and 'Party' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Party'].isin(parties)]
    
    # Additional filtering
    for column, values in kwargs.items():
        if column in filtered_df.columns and values:
            if isinstance(values, list):
                filtered_df = filtered_df[filtered_df[column].isin(values)]
            else:
                filtered_df = filtered_df[filtered_df[column] == values]
    
    return filtered_df

def calculate_significance_test(df: pd.DataFrame, model1: str, model2: str) -> Dict[str, float]:
    """
    Calculate statistical significance test between two models.
    
    Args:
        df: DataFrame with model ratings
        model1: First model column name
        model2: Second model column name
        
    Returns:
        dict: Test statistics
    """
    
    from scipy import stats
    
    if model1 not in df.columns or model2 not in df.columns or 'Human' not in df.columns:
        return {}
    
    # Calculate errors for both models
    errors1 = np.abs(df[model1] - df['Human'])
    errors2 = np.abs(df[model2] - df['Human'])
    
    # Remove NaN values
    mask = ~(np.isnan(errors1) | np.isnan(errors2))
    errors1_clean = errors1[mask]
    errors2_clean = errors2[mask]
    
    if len(errors1_clean) < 10 or len(errors2_clean) < 10:
        return {'error': 'Insufficient data for significance testing'}
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(errors1_clean, errors2_clean)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((errors1_clean.var() + errors2_clean.var()) / 2)
    cohens_d = (errors1_clean.mean() - errors2_clean.mean()) / pooled_std
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'sample_size': len(errors1_clean)
    }

def create_performance_summary(df: pd.DataFrame, models: List[str] = None) -> str:
    """
    Create a text summary of AI model alignment with human baseline.
    
    Args:
        df: Dataset with ratings
        models: List of AI models to analyze (Human is baseline)
        
    Returns:
        str: Text summary of AI-Human alignment
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    # Ensure Human is not included (it's the baseline)
    models = [m for m in models if m != 'Human']
    
    summary_parts = []
    
    # Overall performance vs Human baseline
    mae_scores = {}
    for model in models:
        if model in df.columns and 'Human' in df.columns:
            mae = np.mean(np.abs(df[model] - df['Human']))
            mae_scores[model] = mae
    
    if mae_scores:
        best_model = min(mae_scores.keys(), key=mae_scores.get)
        worst_model = max(mae_scores.keys(), key=mae_scores.get)
        
        summary_parts.append(f"**AI-Human Alignment Summary:**")
        summary_parts.append(f"• Best alignment: **{best_model}** (MAE: {mae_scores[best_model]:.3f} vs Human baseline)")
        summary_parts.append(f"• Poorest alignment: **{worst_model}** (MAE: {mae_scores[worst_model]:.3f} vs Human baseline)")
        
        # Performance gap
        performance_gap = mae_scores[worst_model] - mae_scores[best_model]
        summary_parts.append(f"• Alignment gap: {performance_gap:.3f} MAE units")
    
    # Scenario-specific insights vs Human baseline
    if 'Scenario' in df.columns:
        summary_parts.append(f"\n**Scenario-Based Alignment:**")
        
        for scenario in df['Scenario'].unique():
            scenario_data = df[df['Scenario'] == scenario]
            scenario_mae = {}
            
            for model in models:
                if model in scenario_data.columns and 'Human' in scenario_data.columns:
                    mae = np.mean(np.abs(scenario_data[model] - scenario_data['Human']))
                    scenario_mae[model] = mae
            
            if scenario_mae:
                best_in_scenario = min(scenario_mae.keys(), key=scenario_mae.get)
                summary_parts.append(f"• {scenario}: **{best_in_scenario}** best aligns with Human ratings (MAE: {scenario_mae[best_in_scenario]:.3f})")
    
    return "\n".join(summary_parts)

def export_results_to_csv(metrics_dict: Dict[str, pd.DataFrame], filename_prefix: str = "juror_ai_analysis") -> List[str]:
    """
    Export all metrics to separate CSV files.
    
    Args:
        metrics_dict: Dictionary of metrics DataFrames
        filename_prefix: Prefix for output filenames
        
    Returns:
        list: List of created filenames
    """
    
    created_files = []
    
    for analysis_type, df in metrics_dict.items():
        filename = f"{filename_prefix}_{analysis_type}.csv"
        df.to_csv(f"results/{filename}")
        created_files.append(filename)
    
    return created_files

def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: Dataset to check
        
    Returns:
        dict: Data quality report
    """
    
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_data': {},
        'outliers': {},
        'data_types': {},
        'quality_score': 0.0,
        'issues': []
    }
    
    # Missing data analysis
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        if missing_count > 0:
            report['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            if missing_pct > 10:
                report['issues'].append(f"High missing data in {col}: {missing_pct:.1f}%")
    
    # Outlier detection for rating columns
    rating_cols = ['Human', 'OpenAI', 'Claude', 'Gemini']
    for col in rating_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            report['outliers'][col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df.dropna(subset=[col]))) * 100
            }
    
    # Data types
    for col in df.columns:
        report['data_types'][col] = str(df[col].dtype)
    
    # Calculate quality score (0-100)
    total_issues = len(report['issues'])
    missing_severity = sum([info['percentage'] for info in report['missing_data'].values()])
    outlier_severity = sum([info['percentage'] for info in report['outliers'].values()])
    
    quality_score = max(0, 100 - (total_issues * 10) - (missing_severity / 4) - (outlier_severity / 8))
    report['quality_score'] = round(quality_score, 1)
    
    return report 

def filter_demographics_for_visualization(df: pd.DataFrame, demo_col: str, excluded_groups: List[str] = None) -> pd.DataFrame:
    """
    Filter demographic data for visualization only (not analysis).
    This allows users to exclude small/outlier groups from charts while keeping full dataset.
    
    Args:
        df: Full dataset
        demo_col: Demographic column to filter
        excluded_groups: List of demographic groups to exclude from visualization
        
    Returns:
        pd.DataFrame: Filtered dataframe for visualization
    """
    if excluded_groups is None or not excluded_groups:
        return df
    
    if demo_col not in df.columns:
        return df
    
    # Filter out excluded groups for visualization only
    return df[~df[demo_col].isin(excluded_groups)]

def get_demographic_groups(df: pd.DataFrame, demo_col: str) -> List[str]:
    """
    Get all unique groups in a demographic column.
    
    Args:
        df: Dataset
        demo_col: Demographic column name
        
    Returns:
        List[str]: Sorted list of unique demographic groups
    """
    if demo_col not in df.columns:
        return []
    
    groups = df[demo_col].dropna().unique()
    return sorted(groups)

def get_small_demographic_groups(df: pd.DataFrame, demo_col: str, min_size: int = 50) -> List[str]:
    """
    Identify demographic groups with small sample sizes.
    
    Args:
        df: Dataset
        demo_col: Demographic column name
        min_size: Minimum group size to not be considered "small"
        
    Returns:
        List[str]: Groups with sample size < min_size
    """
    if demo_col not in df.columns:
        return []
    
    group_counts = df[demo_col].value_counts()
    small_groups = group_counts[group_counts < min_size].index.tolist()
    return small_groups 