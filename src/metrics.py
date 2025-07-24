"""
Statistical metrics module for Juror-AI Comparison Analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple

def compute_metrics(y_true: np.array, y_pred: np.array) -> Dict[str, float]:
    """
    Compute statistical metrics comparing predicted vs actual values.
    
    Args:
        y_true: Actual values (human ratings)
        y_pred: Predicted values (AI model ratings)
        
    Returns:
        dict: Dictionary with MAE, MSE, and MPE (Mean Prediction Error)
    """
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'MAE': np.nan, 'MSE': np.nan, 'MPE': np.nan, 'Count': 0}
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    mpe = np.mean(y_pred_clean - y_true_clean)  # Mean Signed Error (bias)
    
    return {
        'MAE': mae,
        'MSE': mse, 
        'MPE': mpe,
        'Count': len(y_true_clean)
    }

def compute_overall_metrics(df: pd.DataFrame, models: List[str] = None) -> pd.DataFrame:
    """
    Compute overall metrics for each AI model vs human ratings (baseline).
    Human ratings are treated as ground truth.
    
    Args:
        df: Dataset with Human and AI model columns
        models: List of AI model column names (excluding Human)
        
    Returns:
        pd.DataFrame: Metrics showing how well each AI model aligns with Human ratings
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    # Remove 'Human' from models list if accidentally included
    models = [m for m in models if m != 'Human']
    
    results = []
    
    for model in models:
        if model in df.columns and 'Human' in df.columns:
            # Human is ground truth (y_true), AI model is prediction (y_pred)
            metrics = compute_metrics(df['Human'].values, df[model].values)
            metrics['Model'] = model
            results.append(metrics)
    
    return pd.DataFrame(results).set_index('Model')

def compute_scenario_metrics(df: pd.DataFrame, models: List[str] = None) -> pd.DataFrame:
    """
    Compute metrics for each AI model by scenario vs human baseline.
    Human ratings are treated as ground truth for each scenario.
    
    Args:
        df: Dataset with Human, AI model, and Scenario columns
        models: List of AI model column names (excluding Human)
        
    Returns:
        pd.DataFrame: Metrics showing AI model alignment with Human ratings by scenario
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    # Remove 'Human' from models list if accidentally included
    models = [m for m in models if m != 'Human']
    
    results = []
    
    for scenario in df['Scenario'].unique():
        scenario_data = df[df['Scenario'] == scenario]
        
        for model in models:
            if model in scenario_data.columns and 'Human' in scenario_data.columns:
                # Human is ground truth, AI model is prediction
                metrics = compute_metrics(
                    scenario_data['Human'].values, 
                    scenario_data[model].values
                )
                metrics['Model'] = model
                metrics['Scenario'] = scenario
                results.append(metrics)
    
    return pd.DataFrame(results)

def compute_demographic_metrics(df: pd.DataFrame, demographic_col: str, 
                               models: List[str] = None) -> pd.DataFrame:
    """
    Compute metrics for each AI model by demographic group vs human baseline.
    Human ratings are treated as ground truth within each demographic group.
    
    Args:
        df: Dataset with Human, AI model, and demographic columns
        demographic_col: Name of the demographic column to group by
        models: List of AI model column names (excluding Human)
        
    Returns:
        pd.DataFrame: Metrics showing AI model alignment with Human ratings by demographic
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    # Remove 'Human' from models list if accidentally included
    models = [m for m in models if m != 'Human']
    
    if demographic_col not in df.columns:
        print(f"⚠️  Column '{demographic_col}' not found in dataset")
        return pd.DataFrame()
    
    results = []
    
    for demographic_value in df[demographic_col].unique():
        if pd.isna(demographic_value):
            continue
            
        demo_data = df[df[demographic_col] == demographic_value]
        
        for model in models:
            if model in demo_data.columns and 'Human' in demo_data.columns:
                # Human is ground truth, AI model is prediction
                metrics = compute_metrics(
                    demo_data['Human'].values, 
                    demo_data[model].values
                )
                metrics['Model'] = model
                metrics[demographic_col] = demographic_value
                results.append(metrics)
    
    return pd.DataFrame(results)

def compute_all_metrics(df: pd.DataFrame, models: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Compute comprehensive metrics across all dimensions.
    All AI models are compared against Human ratings as ground truth.
    
    Args:
        df: Dataset with all required columns
        models: List of AI model column names (excluding Human baseline)
        
    Returns:
        dict: Dictionary containing different metric breakdowns showing AI alignment with Human ratings
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    # Ensure Human is not included in models list (it's the baseline)
    models = [m for m in models if m != 'Human']
    
    results = {}
    
    # Overall metrics
    results['overall'] = compute_overall_metrics(df, models)
    
    # Scenario-level metrics
    if 'Scenario' in df.columns:
        results['by_scenario'] = compute_scenario_metrics(df, models)
    
    # Demographic metrics - include all available demographic columns
    demographic_cols = ['Party', 'Gender', 'Ethnicity', 'Educational Attainment', 
                       'Economic Status of Household']
    
    for demo_col in demographic_cols:
        if demo_col in df.columns:
            key = f'by_{demo_col.lower().replace(" ", "_")}'
            results[key] = compute_demographic_metrics(df, demo_col, models)
    
    return results

def get_error_distribution(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Get the distribution of errors (AI - Human) for a specific model.
    
    Args:
        df: Dataset with Human and AI model columns
        model: Name of the AI model column
        
    Returns:
        pd.DataFrame: Error distribution data
    """
    
    if model not in df.columns or 'Human' not in df.columns:
        return pd.DataFrame()
    
    errors = df[model] - df['Human']
    
    # Create bins for error distribution
    error_data = df.copy()
    error_data['Error'] = errors
    error_data['AbsError'] = np.abs(errors)
    error_data['Model'] = model
    
    return error_data[['JurorID', 'Scenario', 'Human', model, 'Error', 'AbsError', 'Model']]

def compare_models_pairwise(df: pd.DataFrame, models: List[str] = None) -> pd.DataFrame:
    """
    Compare models pairwise to see which performs better in different scenarios.
    
    Args:
        df: Dataset with all model columns
        models: List of AI model column names
        
    Returns:
        pd.DataFrame: Pairwise comparison results
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    results = []
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            if model1 in df.columns and model2 in df.columns and 'Human' in df.columns:
                
                # Calculate errors for both models
                error1 = np.abs(df[model1] - df['Human'])
                error2 = np.abs(df[model2] - df['Human'])
                
                # Count wins (lower error = win)
                model1_wins = (error1 < error2).sum()
                model2_wins = (error2 < error1).sum()
                ties = (error1 == error2).sum()
                
                results.append({
                    'Model_1': model1,
                    'Model_2': model2,
                    f'{model1}_Wins': model1_wins,
                    f'{model2}_Wins': model2_wins,
                    'Ties': ties,
                    'Total_Comparisons': len(df),
                    f'{model1}_Win_Rate': model1_wins / len(df),
                    f'{model2}_Win_Rate': model2_wins / len(df)
                })
    
    return pd.DataFrame(results)

def create_metrics_summary_table(metrics_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary table of all metrics for easy reporting.
    
    Args:
        metrics_dict: Dictionary of metric DataFrames from compute_all_metrics
        
    Returns:
        pd.DataFrame: Summary table
    """
    
    if 'overall' not in metrics_dict:
        return pd.DataFrame()
    
    summary = metrics_dict['overall'].copy()
    summary['Analysis_Level'] = 'Overall'
    
    # Add best/worst scenario performance for each model
    if 'by_scenario' in metrics_dict:
        scenario_metrics = metrics_dict['by_scenario']
        
        for model in summary.index:
            model_scenarios = scenario_metrics[scenario_metrics['Model'] == model]
            
            if not model_scenarios.empty:
                best_scenario = model_scenarios.loc[model_scenarios['MAE'].idxmin(), 'Scenario']
                worst_scenario = model_scenarios.loc[model_scenarios['MAE'].idxmax(), 'Scenario']
                best_mae = model_scenarios['MAE'].min()
                worst_mae = model_scenarios['MAE'].max()
                
                summary.loc[model, 'Best_Scenario'] = best_scenario
                summary.loc[model, 'Best_Scenario_MAE'] = best_mae
                summary.loc[model, 'Worst_Scenario'] = worst_scenario
                summary.loc[model, 'Worst_Scenario_MAE'] = worst_mae
    
    return summary

def print_metrics_report(metrics_dict: Dict[str, pd.DataFrame]) -> None:
    """
    Print a comprehensive metrics report.
    
    Args:
        metrics_dict: Dictionary of metric DataFrames from compute_all_metrics
    """
    
    print("\n" + "="*80)
    print("📊 JUROR-AI COMPARISON METRICS REPORT")
    print("="*80)
    
    # Overall metrics
    if 'overall' in metrics_dict:
        print("\n🎯 OVERALL PERFORMANCE")
        print("-" * 40)
        overall = metrics_dict['overall']
        
        for model in overall.index:
            mae = overall.loc[model, 'MAE']
            mse = overall.loc[model, 'MSE']
            mpe = overall.loc[model, 'MPE']
            count = overall.loc[model, 'Count']
            
            print(f"{model:>8}: MAE={mae:.3f}, MSE={mse:.3f}, MPE={mpe:+.3f} (n={count:,})")
        
        # Rank models by MAE
        best_model = overall['MAE'].idxmin()
        print(f"\n🏆 Best Overall: {best_model} (MAE = {overall.loc[best_model, 'MAE']:.3f})")
    
    # Scenario breakdown
    if 'by_scenario' in metrics_dict:
        print("\n📈 PERFORMANCE BY SCENARIO")
        print("-" * 40)
        scenario_metrics = metrics_dict['by_scenario']
        
        for scenario in scenario_metrics['Scenario'].unique():
            print(f"\n{scenario}:")
            scenario_data = scenario_metrics[scenario_metrics['Scenario'] == scenario]
            
            for _, row in scenario_data.iterrows():
                print(f"  {row['Model']:>8}: MAE={row['MAE']:.3f}")
    
    print("\n" + "="*80) 