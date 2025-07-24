# ⚖️ Juror-AI Comparison Analysis

**Comprehensive analysis comparing human juror ratings with AI model predictions across legal scenarios**

## 👨‍⚖️ Authors

**Sean A. Harrington** and **Hayley Stillwell**  
University of Oklahoma College of Law

This research project examines the alignment between human juror decision-making and artificial intelligence model predictions across various legal scenarios and demographic factors.

## 📋 Overview

This project provides an interactive analysis platform to evaluate how well AI models (OpenAI GPT-4.1, Claude Sonnet 4, and Gemini 2.5) align with human juror decision-making across different legal scenarios and demographic groups.

### 🎯 Key Features

- **Comprehensive Statistical Analysis**: MAE, MSE, MPE metrics across multiple dimensions
- **Interactive Streamlit Dashboard**: Real-time filtering and visualization
- **Modern 2025 Styling**: Consistent color palette and responsive design
- **Multi-dimensional Analysis**: Scenarios, demographics, and model comparisons
- **Export Capabilities**: Download filtered data and metrics in CSV format

### 📊 Dataset Information

- **Total Responses**: 1,198 juror ratings
- **Scenarios**: 4 different legal scenarios
  - Explicit Name
  - Blank Space  
  - Other Person
  - Inferentially Incriminating
- **AI Models**: OpenAI GPT-4.1, Claude Sonnet 4, Gemini 2.5
- **Demographics**: Political affiliation, gender, ethnicity, education, economic status

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone or download the project**
```bash
git clone <repository-url>
cd PaperData
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify data file**
Ensure `data/datacsv.csv` contains your dataset (1,198 rows expected)

### Running the Analysis

#### Option 1: Interactive Dashboard (Recommended)
```bash
streamlit run app/streamlit_app.py
```
This opens a web browser with the interactive dashboard at `http://localhost:8501`

#### Option 2: Jupyter Notebook Analysis
```bash
jupyter notebook notebooks/analysis.ipynb
```
Run all cells for comprehensive analysis and visualization generation

## 📁 Project Structure

```
PaperData/
├── data/                          # Data files
│   └── datacsv.csv               # Main dataset
├── notebooks/                     # Analysis notebooks
│   └── analysis.ipynb            # Complete analysis workflow
├── src/                          # Source modules
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── metrics.py               # Statistical metrics computation
│   ├── plots.py                 # Visualization functions
│   └── utils.py                 # Utility functions
├── app/                         # Streamlit dashboard
│   └── streamlit_app.py         # Main dashboard application
├── results/                     # Generated outputs
│   └── figs/                    # Saved visualizations
├── requirements.txt             # Python dependencies
├── plan.md                      # Original project plan
└── README.md                    # This file
```

## 🎨 Dashboard Features

### 📊 Overview Tab
- Overall performance metrics (MAE, MSE, MPE)
- Visual performance comparison
- Performance summary text

### 📈 Scenarios Tab  
- Scenario-based performance heatmap
- Trend analysis across scenarios
- Scenario distribution charts

### 👥 Demographics Tab
- **Comprehensive Analysis**: All 5 demographic categories in dedicated tabs
- **Performance Metrics**: MAE comparison across all demographic groups
- **Interactive Visualizations**: Charts for Political Affiliation, Gender, Ethnicity, Education, Economic Status
- **Distribution Analysis**: Population breakdowns for each category

### 🔍 Deep Dive Tab
- Error distribution analysis
- Human vs AI scatter plots
- Statistical insights and correlations

### 📋 Data Explorer Tab
- Interactive data table
- Download filtered datasets
- Random sampling for inspection

## 🎛️ Interactive Filters

The dashboard provides comprehensive filtering options:

- **Scenarios**: Select which legal scenarios to analyze
- **AI Models**: Choose which models to compare
- **Political Affiliation**: Filter by political party
- **Demographics**: Comprehensive filters for Gender, Ethnicity, Education, Economic Status

## 📊 Analysis Capabilities

### Statistical Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference
- **MSE (Mean Squared Error)**: Squared error metric
- **MPE (Mean Prediction Error)**: Bias measurement
- **Correlation**: Linear relationship strength

### Visualization Types
- Bar charts for overall performance
- Heatmaps for scenario-model combinations
- Violin plots for error distributions
- Scatter plots for correlation analysis
- Line plots for trend analysis

### Demographic Analysis
- **Political Affiliation**: Performance variations across party lines
- **Gender**: AI-human alignment differences between male/female responses
- **Ethnicity**: Cultural and ethnic pattern analysis
- **Educational Attainment**: Impact of education level on AI alignment
- **Economic Status**: Performance across household income brackets

## 🔧 Technical Implementation

### Data Processing Pipeline
1. **Loading**: CSV data validation and structure checking
2. **Cleaning**: Missing value handling and outlier detection
3. **Transformation**: Column renaming and categorical ordering
4. **Preprocessing**: Format optimization for analysis

### Modern Styling
- **Color Palette**: 
  - Human: #37474F (Dark Blue Grey)
  - OpenAI: #26A69A (Teal)
  - Claude: #FF7043 (Deep Orange)  
  - Gemini: #42A5F5 (Blue)
- **Plotly Theme**: Custom template with consistent styling
- **Responsive Design**: Works on desktop and mobile

### Performance Optimizations
- **Caching**: Streamlit data caching for faster loading
- **Sampling**: Large dataset handling with smart sampling
- **Efficient Filtering**: Optimized DataFrame operations

## 🧪 Testing and Validation

### Data Quality Checks
- Missing value detection and reporting
- Outlier identification using IQR method
- Data type validation
- Schema compliance verification

### Spot-Check Features
- Random sampling for manual verification
- Statistical sanity checks
- Cross-validation of metrics
- Visual inspection tools

## 📈 Key Insights

The analysis reveals:

1. **Model Performance**: Comparative evaluation of AI alignment with human judgment
2. **Scenario Sensitivity**: Different models excel in different legal contexts
3. **Demographic Patterns**: Political and educational factors influence alignment
4. **Statistical Significance**: Robust metrics for model comparison

## 🔮 Future Enhancements

Potential extensions:
- Model ensemble recommendations
- Temporal analysis if data includes timestamps
- Advanced statistical tests (ANOVA, post-hoc)
- Machine learning prediction models
- Real-time data integration

## 📚 Dependencies

Key libraries:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **plotly**: Interactive visualizations
- **streamlit**: Web dashboard framework
- **scikit-learn**: Statistical metrics
- **seaborn**: Statistical plotting

## 🤝 Contributing

To contribute to this project:
1. Follow the existing code structure
2. Maintain the established styling conventions
3. Add comprehensive documentation
4. Test new features thoroughly
5. Update this README for significant changes

## 📝 License

This project is for research and educational purposes. Please ensure compliance with data usage agreements and institutional policies.

## 📞 Support

For questions or issues:
1. Check the Jupyter notebook for detailed analysis examples
2. Review the source code documentation
3. Test with sample data to verify setup
4. Ensure all dependencies are correctly installed

---

**Built with modern data science practices and 2025 design standards** 🚀 