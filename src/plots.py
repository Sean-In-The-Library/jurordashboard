"""
Plotting and visualization module for Juror-AI Comparison Analysis
Modern 2025 styling with consistent color palette
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Set global Plotly theme
pio.templates.default = "plotly_white"

# Modern 2025 Color Palette
PALETTE = {
    'Human': '#37474F',      # Dark Blue Grey
    'OpenAI': '#26A69A',     # Teal
    'Claude': '#FF7043',     # Deep Orange
    'Gemini': '#42A5F5'      # Blue
}

# Extended color palette for demographics
EXTENDED_PALETTE = [
    '#26A69A', '#FF7043', '#42A5F5', '#66BB6A', 
    '#FFA726', '#AB47BC', '#29B6F6', '#EF5350',
    '#7E57C2', '#26C6DA', '#FF8A65', '#9CCC65'
]

def setup_plot_style():
    """Set up consistent plot styling for all visualizations."""
    
    # Plotly template customization
    pio.templates["custom"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Arial, sans-serif", size=12),
            title=dict(font=dict(size=18, color='#2C3E50')),
            paper_bgcolor='white',
            plot_bgcolor='white',
            colorway=list(PALETTE.values()),
            xaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title=dict(font=dict(size=14, color='#34495E'))
            ),
            yaxis=dict(
                gridcolor='#E8E8E8',
                linecolor='#CCCCCC',
                title=dict(font=dict(size=14, color='#34495E'))
            ),
            legend=dict(
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#CCCCCC',
                borderwidth=1
            )
        )
    )
    pio.templates.default = "custom"

def plot_overall_mae(metrics_df: pd.DataFrame, title: str = "AI Model Alignment with Human Ratings (MAE)") -> go.Figure:
    """
    Create a bar chart showing how well each AI model aligns with human ratings.
    Lower MAE = Better alignment with human judgment.
    
    Args:
        metrics_df: DataFrame with MAE metrics by AI model (vs Human baseline)
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Bar chart showing AI-Human alignment
    """
    
    setup_plot_style()
    
    # Get colors for each model
    colors = [PALETTE.get(model, EXTENDED_PALETTE[i % len(EXTENDED_PALETTE)]) 
              for i, model in enumerate(metrics_df.index)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics_df.index,
            y=metrics_df['MAE'],
            marker_color=colors,
            text=[f"{mae:.3f}" for mae in metrics_df['MAE']],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>MAE: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="AI Model",
        yaxis_title="Mean Absolute Error vs Human Ratings (Lower = Better Alignment)",
        showlegend=False,
        height=400,
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    return fig

def plot_scenario_heatmap(df: pd.DataFrame, models: List[str] = None, 
                         title: str = "AI-Human Alignment by Scenario (MAE)") -> go.Figure:
    """
    Create a heatmap showing how well AI models align with human ratings across scenarios.
    Lower values (darker) = Better alignment with human judgment.
    
    Args:
        df: Dataset with scenario and model metrics
        models: List of AI models to include (Human is baseline, not included)
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Heatmap showing AI-Human alignment by scenario
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    # Ensure Human is not included (it's the baseline)
    models = [m for m in models if m != 'Human']
    
    setup_plot_style()
    
    # Compute MAE for each scenario-model combination
    heatmap_data = []
    scenarios = df['Scenario'].unique()
    
    for scenario in scenarios:
        scenario_data = df[df['Scenario'] == scenario]
        row = []
        for model in models:
            if model in scenario_data.columns and 'Human' in scenario_data.columns:
                mae = np.mean(np.abs(scenario_data[model] - scenario_data['Human']))
                row.append(mae)
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=models,
        y=scenarios,
        colorscale='Viridis',
        hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>MAE: %{z:.3f}<extra></extra>',
        colorbar=dict(title="MAE")
    ))
    
    # Add text annotations
    for i, scenario in enumerate(scenarios):
        for j, model in enumerate(models):
            if not np.isnan(heatmap_data[i][j]):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{heatmap_data[i][j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if heatmap_data[i][j] > np.nanmax(heatmap_data) * 0.5 else "black")
                )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="AI Model",
        yaxis_title="Scenario",
        height=500,
        margin=dict(t=80, b=60, l=120, r=60)
    )
    
    return fig

def plot_error_distribution(df: pd.DataFrame, models: List[str] = None,
                           title: str = "Error Distribution by Model") -> go.Figure:
    """
    Create violin plots showing error distributions for each model.
    
    Args:
        df: Dataset with Human and AI model ratings
        models: List of models to include
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Violin plot
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    setup_plot_style()
    
    fig = go.Figure()
    
    for i, model in enumerate(models):
        if model in df.columns and 'Human' in df.columns:
            errors = df[model] - df['Human']
            
            fig.add_trace(go.Violin(
                y=errors,
                name=model,
                box_visible=True,
                meanline_visible=True,
                fillcolor=PALETTE.get(model, EXTENDED_PALETTE[i]),
                opacity=0.7,
                line_color='black',
                hovertemplate=f'<b>{model}</b><br>Error: %{{y:.2f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        yaxis_title="Error (AI - Human)",
        xaxis_title="AI Model",
        height=500,
        margin=dict(t=80, b=60, l=60, r=60),
        showlegend=False
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)
    
    return fig

def plot_demographic_comparison(df: pd.DataFrame, demographic_col: str, 
                               models: List[str] = None,
                               title: str = None) -> go.Figure:
    """
    Create a grouped bar chart comparing model performance across demographic groups.
    
    Args:
        df: Dataset with demographics and ratings
        demographic_col: Column name for demographic grouping
        models: List of models to include
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Grouped bar chart
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    if title is None:
        title = f"Model Performance by {demographic_col}"
    
    setup_plot_style()
    
    # Calculate MAE for each demographic group and model
    demographics = df[demographic_col].unique()
    
    fig = go.Figure()
    
    for i, model in enumerate(models):
        if model in df.columns and 'Human' in df.columns:
            mae_values = []
            
            for demo in demographics:
                demo_data = df[df[demographic_col] == demo]
                if len(demo_data) > 0:
                    mae = np.mean(np.abs(demo_data[model] - demo_data['Human']))
                    mae_values.append(mae)
                else:
                    mae_values.append(np.nan)
            
            fig.add_trace(go.Bar(
                name=model,
                x=demographics,
                y=mae_values,
                marker_color=PALETTE.get(model, EXTENDED_PALETTE[i]),
                hovertemplate=f'<b>{model}</b><br>%{{x}}<br>MAE: %{{y:.3f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=demographic_col,
        yaxis_title="Mean Absolute Error (MAE)",
        barmode='group',
        height=500,
        margin=dict(t=80, b=60, l=60, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_scatter_matrix(df: pd.DataFrame, models: List[str] = None,
                       sample_size: int = 500) -> go.Figure:
    """
    Create a scatter plot matrix comparing Human vs AI ratings.
    
    Args:
        df: Dataset with ratings
        models: List of models to include
        sample_size: Number of points to sample for performance
        
    Returns:
        plotly.graph_objects.Figure: Scatter matrix
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    setup_plot_style()
    
    # Sample data for performance if dataset is large
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    n_models = len(models)
    fig = make_subplots(
        rows=1, cols=n_models,
        subplot_titles=[f"Human vs {model}" for model in models],
        horizontal_spacing=0.1
    )
    
    for i, model in enumerate(models):
        if model in df_sample.columns and 'Human' in df_sample.columns:
            
            fig.add_trace(
                go.Scatter(
                    x=df_sample['Human'],
                    y=df_sample[model],
                    mode='markers',
                    name=model,
                    marker=dict(
                        color=PALETTE.get(model, EXTENDED_PALETTE[i]),
                        size=6,
                        opacity=0.6
                    ),
                    hovertemplate=f'<b>{model}</b><br>Human: %{{x}}<br>AI: %{{y}}<extra></extra>'
                ),
                row=1, col=i+1
            )
            
            # Add diagonal line (perfect agreement)
            min_val = min(df_sample['Human'].min(), df_sample[model].min())
            max_val = max(df_sample['Human'].max(), df_sample[model].max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect Agreement',
                    showlegend=(i == 0)
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title=dict(text="Human vs AI Ratings Comparison", x=0.5),
        height=400,
        margin=dict(t=100, b=60, l=60, r=60),
        showlegend=False
    )
    
    # Update axes labels
    for i in range(n_models):
        fig.update_xaxes(title_text="Human Rating", row=1, col=i+1)
        if i == 0:
            fig.update_yaxes(title_text="AI Rating", row=1, col=i+1)
    
    return fig

def plot_scenario_trends(df: pd.DataFrame, models: List[str] = None,
                        title: str = "Performance Trends Across Scenarios") -> go.Figure:
    """
    Create a line plot showing how model performance varies across scenarios.
    
    Args:
        df: Dataset with scenarios and ratings
        models: List of models to include
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Line plot
    """
    
    if models is None:
        models = ['OpenAI', 'Claude', 'Gemini']
    
    setup_plot_style()
    
    scenarios = sorted(df['Scenario'].unique())
    
    fig = go.Figure()
    
    for i, model in enumerate(models):
        if model in df.columns and 'Human' in df.columns:
            mae_values = []
            
            for scenario in scenarios:
                scenario_data = df[df['Scenario'] == scenario]
                mae = np.mean(np.abs(scenario_data[model] - scenario_data['Human']))
                mae_values.append(mae)
            
            fig.add_trace(go.Scatter(
                x=scenarios,
                y=mae_values,
                mode='lines+markers',
                name=model,
                line=dict(color=PALETTE.get(model, EXTENDED_PALETTE[i]), width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>{model}</b><br>%{{x}}<br>MAE: %{{y:.3f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Scenario",
        yaxis_title="Mean Absolute Error (MAE)",
        height=500,
        margin=dict(t=80, b=80, l=60, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_dashboard_summary_stats(df: pd.DataFrame) -> Dict[str, any]:
    """
    Create summary statistics for dashboard display.
    All AI models are compared against Human ratings as baseline.
    
    Args:
        df: Dataset with all ratings
        
    Returns:
        dict: Summary statistics showing AI alignment with Human baseline
    """
    
    models = ['OpenAI', 'Claude', 'Gemini']  # AI models only, Human is baseline
    
    stats = {
        'total_responses': len(df),
        'total_scenarios': df['Scenario'].nunique(),
        'model_performance': {},
        'best_model': None,
        'scenario_counts': df['Scenario'].value_counts().to_dict()
    }
    
    # Calculate overall MAE for each AI model vs Human baseline
    mae_scores = {}
    for model in models:
        if model in df.columns and 'Human' in df.columns:
            mae = np.mean(np.abs(df[model] - df['Human']))
            mae_scores[model] = mae
            stats['model_performance'][model] = {
                'mae': mae,
                'mse': np.mean((df[model] - df['Human']) ** 2),
                'correlation': df[model].corr(df['Human'])
            }
    
    # Find best aligning model (lowest MAE vs Human)
    if mae_scores:
        stats['best_model'] = min(mae_scores.keys(), key=mae_scores.get)
    
    return stats 