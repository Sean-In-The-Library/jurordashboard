"""
Streamlit Dashboard for Juror-AI Comparison Analysis
Interactive exploration of AI model performance vs human juror ratings
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Custom module imports
from src.data_loader import load_data, preprocess_data, get_sample_data
from src.metrics import compute_all_metrics, compute_overall_metrics
from src.plots import (
    plot_overall_mae, plot_scenario_heatmap, plot_error_distribution,
    plot_demographic_comparison, plot_scatter_matrix, plot_scenario_trends,
    create_dashboard_summary_stats, PALETTE, EXTENDED_PALETTE
)
from src.utils import (
    validate_streamlit_inputs, filter_dataframe, format_metrics_for_display,
    create_download_link, create_performance_summary,
    filter_demographics_for_visualization, get_demographic_groups, get_small_demographic_groups
)

# Page configuration
st.set_page_config(
    page_title="Juror-AI Comparison Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #2C3E50;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid #3498DB;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .stSelectbox label {
        font-weight: 600;
        color: #34495E;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and preprocess data with caching for performance."""
    try:
        raw_df = load_data("data/datacsv.csv")
        processed_df = preprocess_data(raw_df)
        return processed_df, None
    except Exception as e:
        return None, str(e)

def main():
    """Main dashboard application."""
    
    # Header
    st.title("⚖️ Juror-AI Comparison Dashboard")
    st.markdown("""
    **Interactive Analysis of AI Model Alignment with Human Juror Ratings**
    
    Explore how well AI models (OpenAI GPT-4.1, Claude Sonnet 4, and Gemini 2.5) align with **human juror 
    ratings (ground truth baseline)** across different legal scenarios and demographic groups.
    
    📊 **Analysis Framework**: Human ratings = Ground Truth | AI ratings = Predictions to evaluate
    """)
    
    # Load data
    with st.spinner("🔄 Loading and processing data..."):
        df, error = load_and_process_data()
    
    if error:
        st.error(f"❌ Error loading data: {error}")
        st.stop()
    
    if df is None:
        st.error("❌ Failed to load data. Please check the data file.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🎛️ Analysis Configuration")
    
    # Core Analysis Settings
    st.sidebar.subheader("📊 Core Settings")
    
    # AI Model selection (Human is baseline, not selectable)
    available_models = ['OpenAI', 'Claude', 'Gemini']
    selected_models = st.sidebar.multiselect(
        "🤖 AI Models to Analyze",
        available_models,
        default=available_models,
        help="Select AI models to compare against Human baseline"
    )
    
    # Scenario selection
    available_scenarios = sorted(df['Scenario'].unique())
    selected_scenarios = st.sidebar.multiselect(
        "📋 Legal Scenarios",
        available_scenarios,
        default=available_scenarios,
        help="Select legal scenarios to include"
    )
    
    # Data Filtering Options
    st.sidebar.subheader("🎯 Population Filters")
    st.sidebar.markdown("*Filter the dataset by demographic characteristics*")
    
    # Political affiliation selection
    if 'Party' in df.columns:
        available_parties = sorted(df['Party'].unique())
        selected_parties = st.sidebar.multiselect(
            "🏛️ Political Affiliation",
            available_parties,
            default=available_parties
        )
    else:
        selected_parties = []
    
    # Simplified demographic filters with show/hide toggle
    show_more_filters = st.sidebar.checkbox("🔧 Show Additional Filters", value=False)
    
    if show_more_filters:
        # Gender filter
        if 'Gender' in df.columns:
            available_genders = sorted(df['Gender'].unique())
            selected_genders = st.sidebar.multiselect(
                "👤 Gender",
                available_genders,
                default=available_genders
            )
        else:
            selected_genders = []
        
        # Ethnicity filter
        if 'Ethnicity' in df.columns:
            available_ethnicity = sorted(df['Ethnicity'].unique())
            selected_ethnicity = st.sidebar.multiselect(
                "🌍 Ethnicity",
                available_ethnicity,
                default=available_ethnicity
            )
        else:
            selected_ethnicity = []
        
        # Education filter
        if 'Educational Attainment' in df.columns:
            available_education = sorted(df['Educational Attainment'].unique())
            selected_education = st.sidebar.multiselect(
                "🎓 Education Level",
                available_education,
                default=available_education
            )
        else:
            selected_education = []
        
        # Economic Status filter
        if 'Economic Status of Household' in df.columns:
            available_economic = sorted(df['Economic Status of Household'].unique())
            selected_economic = st.sidebar.multiselect(
                "💰 Economic Status",
                available_economic,
                default=available_economic
            )
        else:
            selected_economic = []
    else:
        # Default to all when filters are hidden
        selected_genders = df['Gender'].unique().tolist() if 'Gender' in df.columns else []
        selected_ethnicity = df['Ethnicity'].unique().tolist() if 'Ethnicity' in df.columns else []
        selected_education = df['Educational Attainment'].unique().tolist() if 'Educational Attainment' in df.columns else []
        selected_economic = df['Economic Status of Household'].unique().tolist() if 'Economic Status of Household' in df.columns else []
    
    # Filter Summary
    st.sidebar.subheader("📋 Current Selection")
    
    # Calculate filtered count
    temp_filtered = filter_dataframe(
        df,
        scenarios=selected_scenarios,
        parties=selected_parties,
        Gender=selected_genders if selected_genders else None,
        Ethnicity=selected_ethnicity if selected_ethnicity else None,
        **{'Educational Attainment': selected_education} if selected_education else {},
        **{'Economic Status of Household': selected_economic} if selected_economic else {}
    )
    
    filter_pct = (len(temp_filtered) / len(df)) * 100
    st.sidebar.metric(
        "📊 Responses Included", 
        f"{len(temp_filtered):,}", 
        f"{filter_pct:.1f}% of total"
    )
    
    # Quick summary of active filters
    active_filters = []
    if len(selected_scenarios) < len(available_scenarios):
        active_filters.append(f"{len(selected_scenarios)}/{len(available_scenarios)} scenarios")
    if len(selected_parties) < len(available_parties):
        active_filters.append(f"{len(selected_parties)}/{len(available_parties)} parties")
    if show_more_filters:
        if len(selected_genders) < len(available_genders):
            active_filters.append(f"{len(selected_genders)}/{len(available_genders)} genders")
        if len(selected_ethnicity) < len(available_ethnicity):
            active_filters.append(f"{len(selected_ethnicity)}/{len(available_ethnicity)} ethnicities")
    
    if active_filters:
        st.sidebar.markdown("**Active Filters:**")
        for filt in active_filters[:3]:  # Show max 3 to avoid clutter
            st.sidebar.markdown(f"• {filt}")
        if len(active_filters) > 3:
            st.sidebar.markdown(f"• +{len(active_filters)-3} more...")
    else:
        st.sidebar.success("✅ All data included")
    
    # Reset button
    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()
    
    # Validate inputs
    is_valid, error_msg = validate_streamlit_inputs(
        selected_scenarios, selected_parties, selected_models, df
    )
    
    if not is_valid:
        st.sidebar.error(f"⚠️ {error_msg}")
        st.stop()
    
    # Filter data based on selections
    filter_kwargs = {}
    if selected_genders:
        filter_kwargs['Gender'] = selected_genders
    if selected_ethnicity:
        filter_kwargs['Ethnicity'] = selected_ethnicity
    if selected_education:
        filter_kwargs['Educational Attainment'] = selected_education
    if selected_economic:
        filter_kwargs['Economic Status of Household'] = selected_economic
    
    filtered_df = filter_dataframe(
        df,
        scenarios=selected_scenarios,
        parties=selected_parties,
        **filter_kwargs
    )
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches the selected filters. Please adjust your selections.")
        st.stop()
    
    # Dashboard summary stats
    summary_stats = create_dashboard_summary_stats(filtered_df)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Responses", 
            f"{summary_stats['total_responses']:,}",
            f"{len(df):,} total"
        )
    
    with col2:
        st.metric(
            "📈 Scenarios", 
            summary_stats['total_scenarios'],
            f"{len(available_scenarios)} available"
        )
    
    with col3:
        if summary_stats['best_model']:
            best_mae = summary_stats['model_performance'][summary_stats['best_model']]['mae']
            st.metric(
                "🏆 Best AI Alignment", 
                summary_stats['best_model'],
                f"MAE vs Human: {best_mae:.3f}"
            )
        else:
            st.metric("🏆 Best AI Alignment", "N/A")
    
    with col4:
        if summary_stats['best_model']:
            best_corr = summary_stats['model_performance'][summary_stats['best_model']]['correlation']
            st.metric(
                "🔗 Highest Correlation", 
                f"{best_corr:.3f}",
                "with Human baseline"
            )
        else:
            st.metric("🔗 Correlation", "N/A")
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "📈 By Scenario", "👥 By Demographics", "📊 Demo Comparison", "⚖️ Bias Analysis", "🔍 Deep Dive", "📋 Raw Data"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("📊 AI-Human Alignment Analysis")
        st.info("💡 **How to use**: This shows overall performance of AI models compared to human ratings. Lower MAE = better alignment with human judgment.")
        
        # Compute metrics for filtered data
        with st.spinner("Computing AI-Human alignment metrics..."):
            metrics = compute_overall_metrics(filtered_df, selected_models)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Alignment Metrics (vs Human Baseline)")
            formatted_metrics = format_metrics_for_display(metrics)
            st.dataframe(formatted_metrics, use_container_width=True)
            
            # Performance summary
            st.subheader("📝 AI-Human Alignment Summary")
            performance_text = create_performance_summary(filtered_df, selected_models)
            st.markdown(performance_text)
            
            # Demographic alignment highlights
            st.subheader("👥 Demographic Alignment Highlights")
            from src.metrics import compute_demographic_metrics
            
            demo_highlights = []
            demographic_cols = ['Party', 'Gender', 'Ethnicity', 'Educational Attainment', 'Economic Status of Household']
            
            for demo_col in demographic_cols:
                if demo_col in filtered_df.columns and len(filtered_df[demo_col].unique()) > 1:
                    demo_metrics = compute_demographic_metrics(filtered_df, demo_col, selected_models)
                    if not demo_metrics.empty:
                        best_combo = demo_metrics.loc[demo_metrics['MAE'].idxmin()]
                        demo_highlights.append(
                            f"• **{demo_col}**: {best_combo['Model']} best aligns with {best_combo[demo_col]} humans (MAE: {best_combo['MAE']:.3f})"
                        )
            
            if demo_highlights:
                st.markdown('\n'.join(demo_highlights))
            else:
                st.markdown("*Demographic alignment analysis available in Demographics tab*")
        
        with col2:
            st.subheader("📊 Visual Comparison")
            if not metrics.empty:
                fig_overall = plot_overall_mae(metrics)
                st.plotly_chart(fig_overall, use_container_width=True, key="overview_mae_chart")
            else:
                st.warning("No metrics to display")
    
    # Tab 2: Scenario Analysis
    with tab2:
        st.header("📈 Scenario-Based Analysis")
        st.info("💡 **How to use**: Compare AI performance across different legal scenarios. See which scenarios are easier/harder for AI to match human judgment.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔥 Performance Heatmap")
            fig_heatmap = plot_scenario_heatmap(filtered_df, selected_models)
            st.plotly_chart(fig_heatmap, use_container_width=True, key="scenario_heatmap_chart")
        
        with col2:
            st.subheader("📈 Scenario Trends")
            fig_trends = plot_scenario_trends(filtered_df, selected_models)
            st.plotly_chart(fig_trends, use_container_width=True, key="scenario_trends_chart")
        
        # Scenario counts
        st.subheader("📊 Scenario Distribution")
        scenario_counts = filtered_df['Scenario'].value_counts()
        st.bar_chart(scenario_counts)
    
    # Tab 3: Demographics
    with tab3:
        st.header("👥 Demographic Analysis")
        st.info("💡 **How to use**: Choose a demographic category below to see how AI alignment varies across different population groups.")
        
        # Demographic selector
        demo_options = {
            'Party': '🏛️ Political Affiliation',
            'Gender': '👤 Gender', 
            'Ethnicity': '🌍 Ethnicity',
            'Educational Attainment': '🎓 Educational Attainment',
            'Economic Status of Household': '💰 Economic Status'
        }
        available_demos = {k: v for k, v in demo_options.items() if k in filtered_df.columns}
        
        # Demographic category selector
        st.subheader("👥 Select Demographic Category for Analysis")
        
        if available_demos:
            selected_demo_key = st.selectbox(
                "Choose demographic dimension to analyze:",
                list(available_demos.keys()),
                format_func=lambda x: available_demos[x],
                help="Select which demographic factor to analyze",
                key="demo_selector"
            )
            
            if selected_demo_key:
                # Add visualization filtering controls
                all_groups = get_demographic_groups(filtered_df, selected_demo_key)
                small_groups = get_small_demographic_groups(filtered_df, selected_demo_key, min_size=50)
                
                st.markdown("---")
                st.markdown("**🎛️ Visualization Controls** (affects charts only, not analysis)")
                
                excluded_groups = st.multiselect(
                    f"Exclude groups from visualization:",
                    options=all_groups,
                    default=[],
                    help=f"Select groups to hide from charts. Small groups (n<50): {', '.join(small_groups) if small_groups else 'None'}",
                    key=f"exclude_{selected_demo_key}"
                )
                
                # Quick toggle for small groups
                if small_groups:
                    if st.checkbox(
                        f"Exclude small groups (n<50): {', '.join(small_groups)}",
                        key=f"exclude_small_{selected_demo_key}",
                        help="Quickly exclude all groups with less than 50 responses"
                    ):
                        excluded_groups.extend([g for g in small_groups if g not in excluded_groups])
                
                # Apply visualization filtering
                viz_df = filter_demographics_for_visualization(filtered_df, selected_demo_key, excluded_groups)
                
                # Show data summary
                if excluded_groups:
                    st.info(f"📊 Showing {len(viz_df):,} responses ({len(excluded_groups)} groups excluded for visualization)")
                else:
                    st.info(f"📊 Showing all {len(viz_df):,} responses")
                
                st.markdown("---")
                
                # Compute metrics for selected demographic (using full data for analysis)
                with st.spinner("Computing demographic metrics..."):
                    from src.metrics import compute_demographic_metrics
                    demo_metrics = compute_demographic_metrics(filtered_df, selected_demo_key, selected_models)
                
                # Create two columns for visualization and stats
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"📊 {available_demos[selected_demo_key]} Performance Analysis")
                    fig_demo = plot_demographic_comparison(viz_df, selected_demo_key, selected_models)
                    st.plotly_chart(fig_demo, use_container_width=True, key=f"demo_chart_{selected_demo_key}")
                
                with col2:
                    st.subheader(f"📈 {available_demos[selected_demo_key]} Distribution")
                    demo_counts = viz_df[selected_demo_key].value_counts()
                    st.dataframe(demo_counts.reset_index(), use_container_width=True)
                    
                    # Show performance metrics table
                    if not demo_metrics.empty:
                        st.subheader("🎯 Performance Metrics")
                        formatted_demo_metrics = format_metrics_for_display(demo_metrics)
                        st.dataframe(formatted_demo_metrics, use_container_width=True)
                        
                        # Highlight best performing combination
                        best_combo = demo_metrics.loc[demo_metrics['MAE'].idxmin()]
                        st.success(f"🏆 **Best Performance**: {best_combo['Model']} with {best_combo[selected_demo_key]} (MAE: {best_combo['MAE']:.3f})")
                    else:
                        st.warning("No metrics available for this demographic.")
                
                # Summary insights for selected demographic
                st.subheader(f"💡 {available_demos[selected_demo_key]} Insights")
                
                if not demo_metrics.empty:
                    # Calculate insights
                    unique_groups = demo_metrics[selected_demo_key].unique()
                    model_performance = {}
                    
                    for model in selected_models:
                        model_data = demo_metrics[demo_metrics['Model'] == model]
                        if not model_data.empty:
                            best_group = model_data.loc[model_data['MAE'].idxmin(), selected_demo_key]
                            worst_group = model_data.loc[model_data['MAE'].idxmax(), selected_demo_key]
                            model_performance[model] = {
                                'best': best_group,
                                'worst': worst_group,
                                'best_mae': model_data['MAE'].min(),
                                'worst_mae': model_data['MAE'].max()
                            }
                    
                    # Display insights
                    insights_cols = st.columns(len(selected_models))
                    for i, model in enumerate(selected_models):
                        if model in model_performance:
                            with insights_cols[i]:
                                st.markdown(f"**{model}**")
                                perf = model_performance[model]
                                st.markdown(f"✅ Best: {perf['best']} ({perf['best_mae']:.3f})")
                                st.markdown(f"❌ Worst: {perf['worst']} ({perf['worst_mae']:.3f})")
                else:
                    st.info("Select a demographic category to see detailed insights.")
        
        else:
            st.warning("No demographic data available for analysis.")
    
    # Tab 4: Comprehensive Demographic Comparison
    with tab4:
        st.header("📊 Comprehensive Demographic Analysis")
        st.info("💡 **How to use**: Compare AI performance vs Human baseline across ALL demographic categories. Each chart shows category-specific human consistency as the baseline.")
        
        # Get all demographic columns
        all_demo_cols = {
            'Party': '🏛️ Political Affiliation',
            'Gender': '👤 Gender/Sex', 
            'Ethnicity': '🌍 Ethnicity',
            'Educational Attainment': '🎓 Educational Attainment',
            'Economic Status of Household': '💰 Economic Status'
        }
        
        available_demo_cols = {k: v for k, v in all_demo_cols.items() if k in filtered_df.columns}
        
        if not available_demo_cols:
            st.warning("No demographic data available for analysis.")
        else:
            st.markdown(f"**Analyzing {len(available_demo_cols)} demographic categories with {len(filtered_df):,} responses**")
            
            # Create comprehensive analysis for each demographic
            for demo_col, demo_name in available_demo_cols.items():
                st.subheader(f"{demo_name} Analysis")
                
                # Add visualization controls for this demographic
                all_groups = get_demographic_groups(filtered_df, demo_col)
                small_groups = get_small_demographic_groups(filtered_df, demo_col, min_size=50)
                
                with st.expander(f"🎛️ Visualization Controls for {demo_name}", expanded=False):
                    excluded_groups = st.multiselect(
                        f"Exclude groups from {demo_name} visualization:",
                        options=all_groups,
                        default=[],
                        help=f"Groups to hide from chart. Small groups (n<50): {', '.join(small_groups) if small_groups else 'None'}",
                        key=f"exclude_demo_comp_{demo_col}"
                    )
                    
                    if small_groups:
                        if st.checkbox(
                            f"Exclude small groups: {', '.join(small_groups)}",
                            key=f"exclude_small_demo_comp_{demo_col}",
                            help="Exclude all groups with less than 50 responses"
                        ):
                            excluded_groups.extend([g for g in small_groups if g not in excluded_groups])
                
                # Apply visualization filtering
                demo_viz_df = filter_demographics_for_visualization(filtered_df, demo_col, excluded_groups)
                
                # Calculate human baseline variability within this demographic
                human_baselines = {}
                ai_performance = {model: {} for model in selected_models}
                
                demo_groups = demo_viz_df[demo_col].unique()
                demo_groups = [g for g in demo_groups if pd.notna(g)]
                
                for group in demo_groups:
                    group_data = demo_viz_df[demo_viz_df[demo_col] == group]
                    
                    if len(group_data) > 1:
                        # Human baseline: standard deviation within this group (consistency measure)
                        human_std = group_data['Human'].std()
                        human_mean = group_data['Human'].mean()
                        human_baselines[group] = {
                            'std': human_std,
                            'mean': human_mean,
                            'count': len(group_data)
                        }
                        
                        # AI performance vs human for this group
                        for model in selected_models:
                            if model in group_data.columns:
                                mae = np.mean(np.abs(group_data[model] - group_data['Human']))
                                mpe = np.mean(group_data[model] - group_data['Human'])
                                ai_performance[model][group] = {'MAE': mae, 'MPE': mpe}
                
                if human_baselines:
                    # Create comparison chart
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create comparison plot
                        fig = go.Figure()
                        
                        # Human baseline (standard deviation - consistency measure)
                        groups = list(human_baselines.keys())
                        human_stds = [human_baselines[g]['std'] for g in groups]
                        
                        fig.add_trace(go.Scatter(
                            x=groups,
                            y=human_stds,
                            mode='lines+markers',
                            name='Human Consistency (Lower = More Consistent)',
                            line=dict(color='#37474F', width=3, dash='dash'),
                            marker=dict(size=8),
                            hovertemplate='<b>Human Consistency</b><br>%{x}<br>Std Dev: %{y:.3f}<extra></extra>'
                        ))
                        
                        # AI MAE for each model
                        for i, model in enumerate(selected_models):
                            if model in ai_performance:
                                model_maes = [ai_performance[model].get(g, {}).get('MAE', np.nan) for g in groups]
                                model_maes = [mae for mae in model_maes if not np.isnan(mae)]
                                
                                if model_maes:
                                    fig.add_trace(go.Scatter(
                                        x=groups[:len(model_maes)],
                                        y=model_maes,
                                        mode='lines+markers',
                                        name=f'{model} MAE vs Human',
                                        line=dict(color=PALETTE.get(model, EXTENDED_PALETTE[i]), width=3),
                                        marker=dict(size=8),
                                        hovertemplate=f'<b>{model}</b><br>%{{x}}<br>MAE: %{{y:.3f}}<extra></extra>'
                                    ))
                        
                        fig.update_layout(
                            title=f"{demo_name}: AI Performance vs Human Consistency",
                            xaxis_title=demo_col,
                            yaxis_title="Error/Inconsistency (Lower = Better)",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        # Rotate x-axis labels if there are many categories
                        if len(groups) > 3:
                            fig.update_xaxes(tickangle=45)
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"demo_comparison_{demo_col}")
                    
                    with col2:
                        st.markdown("**📈 Group Statistics**")
                        
                        # Show statistics table
                        stats_data = []
                        for group in groups:
                            if group in human_baselines:
                                hb = human_baselines[group]
                                best_ai_mae = min([ai_performance[model].get(group, {}).get('MAE', float('inf')) 
                                                 for model in selected_models 
                                                 if model in ai_performance])
                                
                                stats_data.append({
                                    'Group': group,
                                    'Human Std': f"{hb['std']:.3f}",
                                    'Best AI MAE': f"{best_ai_mae:.3f}" if best_ai_mae != float('inf') else 'N/A',
                                    'Count': hb['count']
                                })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Insights
                        if human_baselines:
                            most_consistent_group = min(human_baselines.keys(), 
                                                      key=lambda x: human_baselines[x]['std'])
                            least_consistent_group = max(human_baselines.keys(), 
                                                       key=lambda x: human_baselines[x]['std'])
                            
                            st.markdown("**💡 Key Insights:**")
                            st.markdown(f"• Most consistent humans: **{most_consistent_group}**")
                            st.markdown(f"• Least consistent humans: **{least_consistent_group}**")
                            
                            # Find which AI model performs best for this demographic
                            best_overall_performance = {}
                            for model in selected_models:
                                if model in ai_performance:
                                    avg_mae = np.mean([ai_performance[model].get(g, {}).get('MAE', np.nan) 
                                                     for g in groups if not np.isnan(ai_performance[model].get(g, {}).get('MAE', np.nan))])
                                    if not np.isnan(avg_mae):
                                        best_overall_performance[model] = avg_mae
                            
                            if best_overall_performance:
                                best_model = min(best_overall_performance.keys(), key=best_overall_performance.get)
                                st.markdown(f"• Best AI for {demo_name}: **{best_model}**")
                else:
                    st.warning(f"Not enough data for {demo_name} analysis")
                
                st.markdown("---")
    
    # Tab 5: Bias Analysis
    with tab5:
        st.header("⚖️ AI Bias Analysis Across Demographics")
        st.info("💡 **How to use**: Detect systematic bias in AI models across demographic groups. Positive values = AI rates higher than humans for that group. Negative values = AI rates lower. Zero line = no bias (human baseline).")
        
        # Get all demographic columns
        all_demo_cols = {
            'Party': '🏛️ Political Affiliation',
            'Gender': '👤 Gender/Sex', 
            'Ethnicity': '🌍 Ethnicity',
            'Educational Attainment': '🎓 Educational Attainment',
            'Economic Status of Household': '💰 Economic Status'
        }
        
        available_demo_cols = {k: v for k, v in all_demo_cols.items() if k in filtered_df.columns}
        
        if not available_demo_cols:
            st.warning("No demographic data available for bias analysis.")
        else:
            st.markdown(f"**Analyzing bias patterns across {len(available_demo_cols)} demographic categories with {len(filtered_df):,} responses**")
            
            # Create bias analysis for each demographic
            for demo_col, demo_name in available_demo_cols.items():
                st.subheader(f"{demo_name} Bias Analysis")
                
                # Add visualization controls for this demographic
                all_groups = get_demographic_groups(filtered_df, demo_col)
                small_groups = get_small_demographic_groups(filtered_df, demo_col, min_size=50)
                
                with st.expander(f"🎛️ Visualization Controls for {demo_name}", expanded=False):
                    excluded_groups = st.multiselect(
                        f"Exclude groups from {demo_name} bias visualization:",
                        options=all_groups,
                        default=[],
                        help=f"Groups to hide from chart. Small groups (n<50): {', '.join(small_groups) if small_groups else 'None'}",
                        key=f"exclude_bias_{demo_col}"
                    )
                    
                    if small_groups:
                        if st.checkbox(
                            f"Exclude small groups: {', '.join(small_groups)}",
                            key=f"exclude_small_bias_{demo_col}",
                            help="Exclude all groups with less than 50 responses"
                        ):
                            excluded_groups.extend([g for g in small_groups if g not in excluded_groups])
                
                # Apply visualization filtering
                bias_viz_df = filter_demographics_for_visualization(filtered_df, demo_col, excluded_groups)
                
                # Calculate bias (MPE) for each demographic group
                ai_bias = {model: {} for model in selected_models}
                group_stats = {}
                
                demo_groups = bias_viz_df[demo_col].unique()
                demo_groups = [g for g in demo_groups if pd.notna(g)]
                
                for group in demo_groups:
                    group_data = bias_viz_df[bias_viz_df[demo_col] == group]
                    
                    if len(group_data) > 1:
                        # Group statistics
                        human_mean = group_data['Human'].mean()
                        group_stats[group] = {
                            'human_mean': human_mean,
                            'count': len(group_data)
                        }
                        
                        # AI bias (MPE) vs human for this group
                        for model in selected_models:
                            if model in group_data.columns:
                                # MPE = Mean Prediction Error (signed bias)
                                mpe = np.mean(group_data[model] - group_data['Human'])
                                ai_bias[model][group] = mpe
                
                if group_stats:
                    # Create bias comparison chart
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create bias plot
                        fig = go.Figure()
                        
                        # Zero line (no bias baseline)
                        groups = list(group_stats.keys())
                        zero_line = [0] * len(groups)
                        
                        fig.add_trace(go.Scatter(
                            x=groups,
                            y=zero_line,
                            mode='lines',
                            name='No Bias (Human Baseline)',
                            line=dict(color='#37474F', width=3, dash='dash'),
                            hovertemplate='<b>No Bias Baseline</b><br>%{x}<br>Bias: 0.000<extra></extra>'
                        ))
                        
                        # AI bias for each model
                        for i, model in enumerate(selected_models):
                            if model in ai_bias:
                                model_bias = [ai_bias[model].get(g, np.nan) for g in groups]
                                model_bias = [bias for bias in model_bias if not np.isnan(bias)]
                                
                                if model_bias:
                                    fig.add_trace(go.Scatter(
                                        x=groups[:len(model_bias)],
                                        y=model_bias,
                                        mode='lines+markers',
                                        name=f'{model} Bias (MPE)',
                                        line=dict(color=PALETTE.get(model, EXTENDED_PALETTE[i]), width=3),
                                        marker=dict(size=8),
                                        hovertemplate=f'<b>{model} Bias</b><br>%{{x}}<br>MPE: %{{y:.3f}}<extra></extra>'
                                    ))
                        
                        fig.update_layout(
                            title=f"{demo_name}: AI Bias Patterns (MPE vs Human Baseline)",
                            xaxis_title=demo_col,
                            yaxis_title="Bias (+ = AI rates higher than humans, - = AI rates lower)",
                            yaxis=dict(range=[-3, 3]),  # Fixed Y-axis range for better visibility
                            height=400,
                            hovermode='x unified'
                        )
                        
                        # Add horizontal shaded regions for bias interpretation (adjusted for new range)
                        fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="green", opacity=0.1, annotation_text="Low Bias Zone")
                        fig.add_hrect(y0=0.5, y1=3, fillcolor="orange", opacity=0.1, annotation_text="High Positive Bias")
                        fig.add_hrect(y0=-3, y1=-0.5, fillcolor="red", opacity=0.1, annotation_text="High Negative Bias")
                        
                        # Rotate x-axis labels if there are many categories
                        if len(groups) > 3:
                            fig.update_xaxes(tickangle=45)
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"bias_analysis_{demo_col}")
                    
                    with col2:
                        st.markdown("**⚖️ Bias Statistics**")
                        
                        # Show bias statistics table
                        bias_data = []
                        for group in groups:
                            if group in group_stats:
                                gs = group_stats[group]
                                
                                # Find most biased model for this group
                                group_biases = {model: abs(ai_bias[model].get(group, 0)) 
                                              for model in selected_models if model in ai_bias}
                                most_biased_model = max(group_biases.keys(), key=group_biases.get) if group_biases else 'N/A'
                                max_bias = max(group_biases.values()) if group_biases else 0
                                
                                bias_data.append({
                                    'Group': group,
                                    'Human Avg': f"{gs['human_mean']:.2f}",
                                    'Most Biased': most_biased_model,
                                    'Max Bias': f"{max_bias:.3f}",
                                    'Count': gs['count']
                                })
                        
                        bias_df = pd.DataFrame(bias_data)
                        st.dataframe(bias_df, use_container_width=True)
                        
                        # Bias insights
                        if ai_bias:
                            st.markdown("**🎯 Bias Insights:**")
                            
                            # Find most systematically biased model overall
                            model_avg_bias = {}
                            for model in selected_models:
                                if model in ai_bias:
                                    all_biases = [abs(ai_bias[model].get(g, 0)) for g in groups]
                                    model_avg_bias[model] = np.mean(all_biases)
                            
                            if model_avg_bias:
                                most_biased_overall = max(model_avg_bias.keys(), key=model_avg_bias.get)
                                least_biased_overall = min(model_avg_bias.keys(), key=model_avg_bias.get)
                                
                                st.markdown(f"• Most biased: **{most_biased_overall}**")
                                st.markdown(f"• Least biased: **{least_biased_overall}**")
                            
                            # Find which demographic groups face most bias
                            group_total_bias = {}
                            for group in groups:
                                total_bias = sum([abs(ai_bias[model].get(group, 0)) for model in selected_models if model in ai_bias])
                                group_total_bias[group] = total_bias
                            
                            if group_total_bias:
                                most_biased_group = max(group_total_bias.keys(), key=group_total_bias.get)
                                least_biased_group = min(group_total_bias.keys(), key=group_total_bias.get)
                                
                                st.markdown(f"• Most bias against: **{most_biased_group}**")
                                st.markdown(f"• Least bias against: **{least_biased_group}**")
                            
                            # Bias direction analysis
                            st.markdown("**🔍 Bias Directions:**")
                            for model in selected_models:
                                if model in ai_bias:
                                    positive_bias_groups = [g for g in groups if ai_bias[model].get(g, 0) > 0.5]
                                    negative_bias_groups = [g for g in groups if ai_bias[model].get(g, 0) < -0.5]
                                    
                                    if positive_bias_groups:
                                        st.markdown(f"• **{model}** rates higher: {', '.join(positive_bias_groups)}")
                                    if negative_bias_groups:
                                        st.markdown(f"• **{model}** rates lower: {', '.join(negative_bias_groups)}")
                else:
                    st.warning(f"Not enough data for {demo_name} bias analysis")
                
                st.markdown("---")
    
    # Tab 6: Deep Dive
    with tab6:
        st.header("🔍 Deep Dive Analysis")
        st.info("💡 **How to use**: Explore detailed error patterns and correlations between AI models and human ratings.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Error Distribution")
            fig_errors = plot_error_distribution(filtered_df, selected_models)
            st.plotly_chart(fig_errors, use_container_width=True, key="error_distribution_chart")
        
        with col2:
            st.subheader("🎯 Human vs AI Scatter")
            fig_scatter = plot_scatter_matrix(filtered_df, selected_models)
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_matrix_chart")
        
        # Statistical insights
        st.subheader("📈 Statistical Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("**Key Findings:**")
            
            # Calculate some insights
            for model in selected_models:
                if model in filtered_df.columns:
                    correlation = filtered_df[model].corr(filtered_df['Human'])
                    mae = np.mean(np.abs(filtered_df[model] - filtered_df['Human']))
                    
                    st.markdown(f"• **{model}**: Correlation = {correlation:.3f}, MAE = {mae:.3f}")
        
        with insights_col2:
            st.markdown("**Distribution Stats:**")
            
            # Show rating distributions
            for model in ['Human'] + selected_models:
                if model in filtered_df.columns:
                    mean_rating = filtered_df[model].mean()
                    std_rating = filtered_df[model].std()
                    st.markdown(f"• **{model}**: μ = {mean_rating:.2f}, σ = {std_rating:.2f}")
    
    # Tab 7: Raw Data
    with tab7:
        st.header("📋 Raw Data Explorer")
        st.info("💡 **How to use**: Browse and download the actual data being analyzed. Use filters in the sidebar to subset the data.")
        
        # Show filtered data
        st.subheader("🔍 Filtered Dataset")
        st.markdown(f"**Showing {len(filtered_df):,} rows matching your filters**")
        
        # Columns to display
        display_columns = ['JurorID', 'Scenario', 'Human'] + selected_models
        
        # Add all available demographic columns
        demographic_columns = ['Party', 'Gender', 'Ethnicity', 'Educational Attainment', 'Economic Status of Household']
        for demo_col in demographic_columns:
            if demo_col in filtered_df.columns:
                display_columns.append(demo_col)
        
        available_display_cols = [col for col in display_columns if col in filtered_df.columns]
        
        # Display data
        st.dataframe(
            filtered_df[available_display_cols],
            use_container_width=True,
            height=400
        )
        
        # Download options
        st.subheader("💾 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download filtered data
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_data,
                file_name=f"juror_ai_filtered_data_{len(filtered_df)}_rows.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download metrics
            if not metrics.empty:
                metrics_csv = metrics.to_csv()
                st.download_button(
                    label="📊 Download Metrics (CSV)",
                    data=metrics_csv,
                    file_name="juror_ai_metrics.csv",
                    mime="text/csv"
                )
        
        # Sample data for quick inspection
        st.subheader("🎲 Random Sample")
        sample_size = st.slider("Sample size:", 5, min(50, len(filtered_df)), 10)
        sample_data = get_sample_data(filtered_df, sample_size)
        st.dataframe(sample_data[available_display_cols], use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Juror-AI Comparison Dashboard** | "
        "Built with Streamlit | "
        f"Data: {len(df):,} total responses | "
        f"Filtered: {len(filtered_df):,} responses"
    )

if __name__ == "__main__":
    main() 