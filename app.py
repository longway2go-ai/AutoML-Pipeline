import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score  # Add these for imbalanced data

# ML Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, SGDRegressor

# For handling imbalanced datasets
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. SMOTE will not be available.")
    print("Install with: pip install imbalanced-learn")

# Collections for counting
from collections import Counter

import time
import random
import io

# Page configuration
st.set_page_config(
    page_title="AutoML Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 2rem;
        color: #667eea;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    .step-header {
        font-size: 1.5rem;
        color: #764ba2;
        margin: 1rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea20, #764ba220);
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .fact-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        font-style: italic;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .step-completed {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .step-current {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    .step-pending {
        background: #f0f0f0;
        padding: 1rem;
        border-radius: 10px;
        color: #666;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .navigation-buttons {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Random facts for entertainment during processing
RANDOM_FACTS = [
    "🧠 Did you know? Machine Learning algorithms can process data 1000x faster than humans!",
    "🌟 Fun fact: Random Forest gets its name because it creates a 'forest' of decision trees!",
    "🚀 Amazing: Netflix saves $1 billion per year using machine learning recommendations!",
    "🎯 Cool fact: Support Vector Machines can work in infinite dimensions!",
    "🔮 Interesting: Ensemble methods combine multiple models to make better predictions!",
    "💡 Did you know? Cross-validation helps prevent overfitting by testing on unseen data!",
    "🎲 Fun fact: Gradient boosting builds models sequentially, learning from previous mistakes!",
    "🌊 Amazing: Feature scaling can improve model performance by up to 40%!",
    "🔍 Cool fact: Data preprocessing takes 80% of a data scientist's time!",
    "🎪 Interesting: AdaBoost was one of the first successful ensemble methods!",
    "🎨 Did you know? RandomSearchCV is often better than GridSearch for hyperparameter tuning!",
    "⚡ Fun fact: Hyperparameter optimization can improve model performance by 15-30%!"
]

MODEL_FACTS = {
    'RandomForestClassifier': [
        "Random Forest is resistant to overfitting due to averaging multiple decision trees",
        "It can handle missing values and maintains accuracy even with missing data",
        "Random Forest provides feature importance rankings automatically"
    ],
    'GradientBoostingClassifier': [
        "Gradient Boosting builds models sequentially, each correcting the previous one's errors",
        "It's highly effective for structured data and often wins Kaggle competitions",
        "XGBoost and LightGBM are popular implementations of gradient boosting"
    ],
    'LogisticRegression': [
        "Logistic Regression uses the sigmoid function to map predictions to probabilities",
        "Despite its name, it's a classification algorithm, not regression",
        "It's highly interpretable and works well with linearly separable data"
    ],
    'SVC': [
        "Support Vector Machines find the optimal hyperplane that maximizes margin between classes",
        "SVMs can handle non-linear relationships using kernel tricks",
        "They work well with high-dimensional data and are memory efficient"
    ],
    'RandomForestRegressor': [
        "Random Forest Regression combines predictions from multiple decision trees",
        "It provides built-in cross-validation through out-of-bag error estimation",
        "The algorithm is robust to outliers and handles non-linear relationships well"
    ],
    'GradientBoostingRegressor': [
        "Gradient Boosting Regression minimizes loss function using gradient descent",
        "It can capture complex patterns and interactions in the data",
        "The learning rate parameter controls the contribution of each tree"
    ]
}

# Define steps
STEPS = [
    "📁 Data Upload",
    "⚙️ Data Preprocessing",
    "📊 Exploratory Data Analysis", 
    "🚀 Model Training",
    "📈 Model Evaluation", 
    "🏆 Final Results"
]

def show_progress_sidebar():
    """Show progress in sidebar"""
    st.sidebar.markdown("## 📋 Pipeline Progress")
    
    for i, step in enumerate(STEPS):
        if i < st.session_state.current_step:
            st.sidebar.markdown(f'<div class="step-completed">✅ {step}</div>', unsafe_allow_html=True)
        elif i == st.session_state.current_step:
            st.sidebar.markdown(f'<div class="step-current">🔄 {step}</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f'<div class="step-pending">⏳ {step}</div>', unsafe_allow_html=True)
    
    # Show random fact
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎲 Random ML Fact")
    fact = random.choice(RANDOM_FACTS)
    st.sidebar.info(fact)

def show_navigation_buttons():
    """Show navigation buttons"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_step > 0:
            if st.button("⬅️ Previous", type="secondary", use_container_width=True):
                st.session_state.current_step -= 1
                st.rerun()
    
    with col3:
        if st.session_state.current_step < len(STEPS) - 1:
            # Check if current step is completed
            can_proceed = False
            if st.session_state.current_step == 0 and st.session_state.df is not None:
                can_proceed = True
            elif st.session_state.current_step == 1 and st.session_state.processed_df is not None:
                can_proceed = True
            elif st.session_state.current_step == 2 and st.session_state.target_col is not None:
                can_proceed = True
            elif st.session_state.current_step == 3 and st.session_state.results is not None:
                can_proceed = True
            elif st.session_state.current_step == 4 and st.session_state.best_model is not None:
                can_proceed = True
            
            if can_proceed:
                if st.button("Next ➡️", type="primary", use_container_width=True):
                    st.session_state.current_step += 1
                    st.rerun()

def load_data(uploaded_file):
    """Load CSV or Excel file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or Excel file")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def step_data_upload():
    """Step 1: Data Upload"""
    st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the AutoML Pipeline! 🎉
    
    Upload your dataset and let our AI-powered system handle the entire machine learning pipeline for you.
    
    **Supported formats:**
    - 📄 CSV files (.csv)
    - 📊 Excel files (.xlsx, .xls)
    
    **What happens next:**
    1. **EDA**: Comprehensive data analysis and visualization
    2. **Preprocessing**: Automatic data cleaning and preparation
    3. **Model Training**: Multiple ML models with hyperparameter tuning
    4. **Evaluation**: Performance comparison and best model selection
    5. **Results**: Detailed insights and model information
    """)
    
    uploaded_file = st.file_uploader(
        "Choose your dataset file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset to begin the automated ML pipeline"
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading your data..."):
            df = load_data(uploaded_file)
        
        if df is not None:
            st.session_state.df = df
            st.success(f"✅ Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns!")
            
            # Quick preview
            st.subheader("📋 Quick Data Preview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{df.shape[0]}</h3><p>Rows</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Columns</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>{df.isnull().sum().sum()}</h3><p>Missing Values</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><h3>{df.duplicated().sum()}</h3><p>Duplicates</p></div>', unsafe_allow_html=True)
            
            st.dataframe(df.head(), use_container_width=True)
            
            # Target column selection
            st.subheader("🎯 Select Target Column")
            target_col = st.selectbox(
                "Choose the column you want to predict:",
                ["Select a column..."] + df.columns.tolist(),
                help="This is the variable you want to predict (dependent variable)"
            )
            
            if target_col != "Select a column...":
                st.session_state.target_col = target_col
                
                # Determine problem type
                unique_values = df[target_col].nunique()
                total_values = len(df[target_col])
                
                if df[target_col].dtype in ['int64', 'float64']:
                    if unique_values > 10 and unique_values > total_values * 0.05:
                        problem_type = 'regression'
                    else:
                        problem_type = 'classification'
                else:
                    problem_type = 'classification'
                
                st.session_state.problem_type = problem_type
                st.info(f"🔍 Detected problem type: **{problem_type.title()}**")

def step_preprocessing():
    """Step 2: Data Preprocessing"""
    st.markdown('<div class="section-header">⚙️ Data Preprocessing</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.error("❌ No data loaded. Please go back to the Data Upload step.")
        return
    
    st.markdown("""
    ### 🔧 Automated Data Preprocessing
    
    Our preprocessing pipeline will automatically handle:
    - **Missing Values**: Fill with appropriate strategies (mean for numeric, mode for categorical)
    - **Data Type Optimization**: Convert string numbers to numeric types
    - **Categorical Encoding**: Label encode categorical variables for ML compatibility
    - **Duplicate Removal**: Remove duplicate rows to prevent data leakage
    
    This ensures your data is clean and ready for analysis and modeling.
    """)
    
    if st.button("🚀 Start Preprocessing", type="primary", use_container_width=True):
        df = st.session_state.df.copy()
        preprocessing_steps = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Handle missing values
        status_text.text("🔧 Handling missing values...")
        progress_bar.progress(0.2)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Fill numeric missing values with mean
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='mean')
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            preprocessing_steps.append(f"✅ Filled {len(numeric_cols)} numeric columns with mean values")
        
        # Fill categorical missing values with mode
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
            preprocessing_steps.append(f"✅ Filled {len(categorical_cols)} categorical columns with mode values")
        
        time.sleep(1)
        
        # Step 2: Data type optimization
        status_text.text("🎯 Optimizing data types...")
        progress_bar.progress(0.4)
        
        # Convert object columns that look like numbers
        for col in categorical_cols:
            try:
                if df[col].astype(str).str.replace('.', '').str.replace('-', '').str.isdigit().all():
                    df[col] = pd.to_numeric(df[col])
                    preprocessing_steps.append(f"✅ Converted '{col}' from object to numeric")
            except:
                pass
        
        time.sleep(1)
        
        # Step 3: Encode categorical variables
        status_text.text("🏷️ Encoding categorical variables...")
        progress_bar.progress(0.6)
        
        label_encoders = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            preprocessing_steps.append(f"✅ Label encoded categorical column: '{col}'")
        
        time.sleep(1)
        
        # Step 4: Remove duplicates
        status_text.text("🗑️ Removing duplicates...")
        progress_bar.progress(0.8)
        
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            preprocessing_steps.append(f"✅ Removed {removed_duplicates} duplicate rows")
        
        time.sleep(1)
        
        # Step 5: Final validation
        status_text.text("✨ Finalizing preprocessing...")
        progress_bar.progress(1.0)
        
        st.session_state.processed_df = df
        
        # Display results
        st.success("🎉 Preprocessing completed successfully!")
        
        # Show preprocessing summary
        st.markdown('<div class="step-header">📋 Preprocessing Summary</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before Preprocessing")
            original_df = st.session_state.df
            st.write(f"**Rows**: {original_df.shape[0]}")
            st.write(f"**Columns**: {original_df.shape[1]}")
            st.write(f"**Missing Values**: {original_df.isnull().sum().sum()}")
            st.write(f"**Duplicates**: {original_df.duplicated().sum()}")
            st.write(f"**Data Types**: {original_df.dtypes.value_counts().to_dict()}")
        
        with col2:
            st.subheader("After Preprocessing")
            st.write(f"**Rows**: {df.shape[0]}")
            st.write(f"**Columns**: {df.shape[1]}")
            st.write(f"**Missing Values**: {df.isnull().sum().sum()}")
            st.write(f"**Duplicates**: {df.duplicated().sum()}")
            st.write(f"**Data Types**: {df.dtypes.value_counts().to_dict()}")
        
        # Show steps taken
        st.subheader("🔧 Steps Performed")
        for step in preprocessing_steps:
            st.write(step)
        
        # Show processed data preview
        st.subheader("👀 Processed Data Preview")
        st.dataframe(df.head(), use_container_width=True)

def step_eda():
    """Step 3: Enhanced Exploratory Data Analysis"""
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_df is None:
        st.error("❌ No processed data available. Please complete the preprocessing step.")
        return
    
    df = st.session_state.processed_df
    
    st.markdown("""
    ### 🔍 Comprehensive Data Analysis
    
    Now that your data is clean and processed, let's explore it with advanced visualizations:
    - **Distribution Analysis**: Histograms, box plots, and pie charts
    - **Relationship Analysis**: Correlation matrices and scatter plots  
    - **Target Analysis**: Deep dive into your prediction variable
    - **Feature Insights**: Statistical summaries and patterns
    """)
    
    # Target column selection for EDA
    if st.session_state.target_col is None:
        st.subheader("🎯 Select Target Column for Analysis")
        target_col = st.selectbox(
            "Choose the column you want to predict:",
            ["Select a column..."] + df.columns.tolist(),
            help="This is the variable you want to predict (dependent variable)"
        )
        
        if target_col != "Select a column...":
            st.session_state.target_col = target_col
            
            # Determine problem type
            unique_values = df[target_col].nunique()
            total_values = len(df[target_col])
            
            if df[target_col].dtype in ['int64', 'float64']:
                if unique_values > 10 and unique_values > total_values * 0.05:
                    problem_type = 'regression'
                else:
                    problem_type = 'classification'
            else:
                problem_type = 'classification'
            
            st.session_state.problem_type = problem_type
            st.info(f"🔍 Detected problem type: **{problem_type.title()}**")
            st.rerun()
    
    if st.session_state.target_col is None:
        st.warning("⚠️ Please select a target column to continue with EDA.")
        return
    
    target_col = st.session_state.target_col
    
    # Dataset Overview
    st.markdown('<div class="step-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h2>{df.shape[0]}</h2><p>Total Rows</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h2>{df.shape[1]}</h2><p>Total Columns</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h2>{df.select_dtypes(include=[np.number]).shape[1]}</h2><p>Numeric Columns</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h2>{df.select_dtypes(include=["object"]).shape[1]}</h2><p>Categorical Columns</p></div>', unsafe_allow_html=True)
    
    # Data Types and Info
    st.markdown('<div class="step-header">📋 Data Types & Statistical Summary</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Unique Values': df.nunique()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Target Variable Analysis
    st.markdown('<div class="step-header">🎯 Target Variable Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.problem_type == 'classification':
            # Pie chart for classification
            value_counts = df[target_col].value_counts()
            fig = px.pie(
                values=value_counts.values, 
                names=value_counts.index,
                title=f'Target Distribution - {target_col}'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Histogram for regression
            fig = px.histogram(
                df, x=target_col, 
                title=f'Target Distribution - {target_col}', 
                nbins=30,
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Target Statistics")
        if st.session_state.problem_type == 'classification':
            value_counts = df[target_col].value_counts()
            st.write("**Class Distribution:**")
            for class_val, count in value_counts.items():
                percentage = (count / len(df)) * 100
                st.write(f"- **{class_val}**: {count} ({percentage:.1f}%)")
            
            # Class balance check
            balance_ratio = value_counts.min() / value_counts.max()
            if balance_ratio < 0.5:
                st.warning(f"⚠️ Imbalanced classes detected (ratio: {balance_ratio:.2f})")
            else:
                st.success(f"✅ Classes are reasonably balanced (ratio: {balance_ratio:.2f})")
        else:
            st.write(f"**Mean**: {df[target_col].mean():.4f}")
            st.write(f"**Median**: {df[target_col].median():.4f}")
            st.write(f"**Std Dev**: {df[target_col].std():.4f}")
            st.write(f"**Min**: {df[target_col].min():.4f}")
            st.write(f"**Max**: {df[target_col].max():.4f}")
            st.write(f"**Skewness**: {df[target_col].skew():.4f}")
            st.write(f"**Kurtosis**: {df[target_col].kurtosis():.4f}")
    
    # Feature Distributions with Multiple Plot Types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numeric_cols) > 0:
        st.markdown('<div class="step-header">📈 Numeric Feature Analysis</div>', unsafe_allow_html=True)
        
        # Select columns to analyze
        selected_numeric = st.multiselect(
            "Select numeric columns to analyze:",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        )
        
        if selected_numeric:
            # Plot type selection
            plot_type = st.selectbox(
                "Choose visualization type:",
                ["Histograms", "Box Plots", "Line Plots (Index)", "Distribution with KDE"]
            )
            
            if plot_type == "Histograms":
                n_cols = min(2, len(selected_numeric))
                for i in range(0, len(selected_numeric), n_cols):
                    cols = st.columns(n_cols)
                    for j, col in enumerate(selected_numeric[i:i+n_cols]):
                        with cols[j]:
                            fig = px.histogram(
                                df, x=col, 
                                title=f'Distribution of {col}',
                                marginal="rug"
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Box Plots":
                n_cols = min(2, len(selected_numeric))
                for i in range(0, len(selected_numeric), n_cols):
                    cols = st.columns(n_cols)
                    for j, col in enumerate(selected_numeric[i:i+n_cols]):
                        with cols[j]:
                            fig = px.box(
                                df, y=col,
                                title=f'Box Plot of {col}'
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Line Plots (Index)":
                n_cols = min(2, len(selected_numeric))
                for i in range(0, len(selected_numeric), n_cols):
                    cols = st.columns(n_cols)
                    for j, col in enumerate(selected_numeric[i:i+n_cols]):
                        with cols[j]:
                            fig = px.line(
                                df.reset_index(), 
                                x='index', y=col,
                                title=f'Line Plot of {col}'
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Distribution with KDE":
                n_cols = min(2, len(selected_numeric))
                for i in range(0, len(selected_numeric), n_cols):
                    cols = st.columns(n_cols)
                    for j, col in enumerate(selected_numeric[i:i+n_cols]):
                        with cols[j]:
                            fig = px.histogram(
                                df, x=col,
                                title=f'Distribution of {col} with KDE',
                                marginal="box",
                                histnorm='density'
                            )
                            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical Features Analysis
    if len(categorical_cols) > 0:
        st.markdown('<div class="step-header">📊 Categorical Feature Analysis</div>', unsafe_allow_html=True)
        
        selected_categorical = st.multiselect(
            "Select categorical columns to analyze:",
            categorical_cols,
            default=categorical_cols[:3] if len(categorical_cols) >= 3 else categorical_cols
        )
        
        if selected_categorical:
            for col in selected_categorical:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    value_counts = df[col].value_counts().head(10)  # Top 10 categories
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f'Distribution of {col}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f'Count by {col}',
                        labels={'x': col, 'y': 'Count'}
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    if len(numeric_cols) > 1:
        st.markdown('<div class="step-header">🔥 Correlation Analysis</div>', unsafe_allow_html=True)
        
        corr_matrix = df[numeric_cols].corr()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.imshow(
                corr_matrix, 
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation with target
            if target_col in numeric_cols:
                target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)[1:6]
                st.subheader("🎯 Top Features Correlated with Target")
                for i, (feature, corr_val) in enumerate(target_corr.items(), 1):
                    st.write(f"**{i}. {feature}**: {corr_val:.4f}")
            
            # Highly correlated pairs
            st.subheader("🔗 Highly Correlated Pairs")
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.7:  # High correlation threshold
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_val
                        ))
            
            if high_corr_pairs:
                for feat1, feat2, corr_val in high_corr_pairs[:5]:
                    st.write(f"**{feat1}** ↔ **{feat2}**: {corr_val:.3f}")
            else:
                st.write("No highly correlated pairs found (>0.7)")
    
    # Feature vs Target Analysis
    if target_col and len(numeric_cols) > 1:
        st.markdown('<div class="step-header">🎯 Feature vs Target Relationships</div>', unsafe_allow_html=True)
        
        # Select feature for analysis
        feature_col = st.selectbox(
            "Select a feature to analyze against target:",
            [col for col in numeric_cols if col != target_col]
        )
        
        if feature_col:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.problem_type == 'classification':
                    # Box plot for classification
                    fig = px.box(
                        df, x=target_col, y=feature_col,
                        title=f'{feature_col} by {target_col}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Scatter plot for regression
                    fig = px.scatter(
                        df, x=feature_col, y=target_col,
                        title=f'{target_col} vs {feature_col}',
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statistical relationship
                if st.session_state.problem_type == 'regression':
                    correlation = df[feature_col].corr(df[target_col])
                    st.metric("Correlation with Target", f"{correlation:.4f}")
                    
                    # Additional statistics
                    st.write("**Relationship Strength:**")
                    if abs(correlation) > 0.7:
                        st.success("🔥 Strong relationship")
                    elif abs(correlation) > 0.3:
                        st.info("📈 Moderate relationship")
                    else:
                        st.warning("📉 Weak relationship")
                else:
                    # For classification, show mean values by class
                    st.write("**Mean values by class:**")
                    class_means = df.groupby(target_col)[feature_col].mean().sort_values(ascending=False)
                    for class_val, mean_val in class_means.items():
                        st.write(f"**Class {class_val}**: {mean_val:.4f}")
    
    # Data Quality Summary
    st.markdown('<div class="step-header">✅ Data Quality Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔍 Completeness")
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("Missing Data", f"{missing_percentage:.2f}%")
        if missing_percentage == 0:
            st.success("✅ No missing data")
        elif missing_percentage < 5:
            st.info("📊 Low missing data")
        else:
            st.warning("⚠️ High missing data")
    
    with col2:
        st.subheader("🎯 Target Quality")
        if st.session_state.problem_type == 'classification':
            # Class balance
            value_counts = df[target_col].value_counts()
            balance_ratio = value_counts.min() / value_counts.max()
            st.metric("Class Balance Ratio", f"{balance_ratio:.2f}")
            if balance_ratio > 0.5:
                st.success("✅ Well balanced")
            elif balance_ratio > 0.2:
                st.info("📊 Moderately balanced")
            else:
                st.error("❌ Highly imbalanced")
        else:
            # Distribution normality
            skewness = abs(df[target_col].skew())
            st.metric("Target Skewness", f"{skewness:.2f}")
            if skewness < 0.5:
                st.success("✅ Normal distribution")
            elif skewness < 1:
                st.info("📊 Slightly skewed")
            else:
                st.warning("⚠️ Highly skewed")
    
    with col3:
        st.subheader("🔗 Feature Relationships")
        if len(numeric_cols) > 1:
            high_corr_count = 0
            corr_matrix = df[numeric_cols].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_count += 1
            
            st.metric("High Correlations (>0.8)", high_corr_count)
            if high_corr_count == 0:
                st.success("✅ No multicollinearity")
            elif high_corr_count < 3:
                st.info("📊 Some correlations")
            else:
                st.warning("⚠️ High multicollinearity")
        else:
            st.info("Not enough numeric features")
    
    st.success("🎉 EDA completed! Your data is ready for model training.")
def step_model_training():
"""Step 4: Model Training with RandomizedSearchCV and SMOTE for imbalanced data"""
st.markdown('<div class="section-header">🚀 Model Training & Hyperparameter Optimization</div>', unsafe_allow_html=True)

if st.session_state.processed_df is None:
    st.error("❌ No processed data available. Please complete the preprocessing step.")
    return

if st.session_state.target_col is None:
    st.error("❌ No target column selected. Please go back to data upload step.")
    return

st.markdown("""
### 🎯 Training Strategy

We'll train multiple machine learning models with **RandomizedSearchCV** for hyperparameter optimization:

- **RandomizedSearchCV**: Efficiently searches through hyperparameter space
- **Cross-Validation**: 5-fold CV to ensure robust performance estimates
- **Multiple Models**: Various algorithms to find the best fit for your data
- **Hyperparameter Tuning**: Automatic optimization for better results
- **SMOTE**: Handle imbalanced datasets with synthetic minority oversampling
""")

# Training configuration
col1, col2 = st.columns(2)
with col1:
    cv_folds = st.selectbox("Cross-Validation Folds", [3, 5, 10], index=1)
    n_iter = st.selectbox("RandomSearch Iterations", [10, 20, 50], index=1)

with col2:
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", value=42, min_value=0)

# SMOTE option for classification
use_smote = False
if st.session_state.problem_type == 'classification':
    use_smote = st.checkbox("🔄 Use SMOTE for imbalanced data", value=True, 
                           help="SMOTE (Synthetic Minority Oversampling Technique) creates synthetic samples for minority classes")

if st.button("🚀 Start Model Training", type="primary", use_container_width=True):
    df = st.session_state.processed_df
    target_col = st.session_state.target_col
    problem_type = st.session_state.problem_type
    
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Check for minimum samples per class for stratification
    from collections import Counter
    
    def can_stratify(y, test_size):
        class_counts = Counter(y)
        min_samples_needed = max(2, int(2/test_size))  # At least 2 samples per split
        return all(count >= min_samples_needed for count in class_counts.values())
    
    # Determine stratification parameter
    if problem_type == 'classification':
        stratify_param = y if can_stratify(y, test_size) else None
        if stratify_param is None:
            st.warning("⚠️ Cannot use stratification due to very small class sizes. Using random split.")
    else:
        stratify_param = None
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=stratify_param
    )
    
    st.info(f"🔍 Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    # Check class balance for classification
    if problem_type == 'classification':
        class_counts = Counter(y_train)
        st.subheader("📊 Class Distribution in Training Set")
        
        col1, col2 = st.columns(2)
        with col1:
            for class_val, count in class_counts.items():
                percentage = (count / len(y_train)) * 100
                st.write(f"**Class {class_val}**: {count} ({percentage:.1f}%)")
        
        with col2:
            # Calculate imbalance ratio
            balance_ratio = min(class_counts.values()) / max(class_counts.values())
            st.metric("Balance Ratio", f"{balance_ratio:.3f}")
            
            if balance_ratio < 0.5:
                st.warning("⚠️ Imbalanced dataset detected!")
                if use_smote:
                    st.info("🔄 SMOTE will be applied to balance the classes")
            else:
                st.success("✅ Dataset is reasonably balanced")
    
    # Apply SMOTE if requested and applicable
    X_train_balanced = X_train.copy()
    y_train_balanced = y_train.copy()
    
    if use_smote and problem_type == 'classification':
        try:
            from imblearn.over_sampling import SMOTE
            
            st.info("🔄 Applying SMOTE to balance the dataset...")
            
            # Check if SMOTE can be applied
            min_class_count = min(Counter(y_train).values())
            if min_class_count >= 2:  # SMOTE needs at least 2 samples per class
                smote = SMOTE(random_state=random_state, k_neighbors=min(5, min_class_count-1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                
                # Show new class distribution
                new_class_counts = Counter(y_train_balanced)
                st.success(f"✅ SMOTE applied! New training set size: {len(y_train_balanced)} samples")
                
                st.subheader("📊 Class Distribution After SMOTE")
                for class_val, count in new_class_counts.items():
                    percentage = (count / len(y_train_balanced)) * 100
                    st.write(f"**Class {class_val}**: {count} ({percentage:.1f}%)")
            else:
                st.warning("⚠️ Cannot apply SMOTE: insufficient samples in minority class")
                use_smote = False
        
        except ImportError:
            st.error("❌ SMOTE requires imbalanced-learn package. Install with: pip install imbalanced-learn")
            use_smote = False
        except Exception as e:
            st.error(f"❌ Error applying SMOTE: {str(e)}")
            use_smote = False
    
    # Get models and parameters
    models_params = get_ml_models_with_params(problem_type)
    results = {}
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Progress tracking
    total_models = len(models_params)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training container
    training_container = st.container()
    
    for idx, (name, config) in enumerate(models_params.items()):
        with training_container:
            st.markdown(f'<div class="step-header">🔄 Training {name}...</div>', unsafe_allow_html=True)
            
            status_text.text(f"Training {name} with RandomizedSearchCV... ({idx+1}/{total_models})")
            
            try:
                # Use scaled data for algorithms that need it
                if name in ['SVC', 'SVR', 'LogisticRegression']:
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                else:
                    X_train_use = X_train_balanced
                    X_test_use = X_test
                
                # RandomizedSearchCV
                scoring = 'accuracy' if problem_type == 'classification' else 'r2'
                
                # For cross-validation with SMOTE, we need to be careful about class sizes
                cv_folds_use = cv_folds
                if problem_type == 'classification' and use_smote:
                    min_class_size = min(Counter(y_train_balanced).values())
                    cv_folds_use = min(cv_folds, min_class_size)
                
                random_search = RandomizedSearchCV(
                    estimator=config['model'],
                    param_distributions=config['params'],
                    n_iter=n_iter,
                    cv=cv_folds_use,
                    scoring=scoring,
                    random_state=random_state,
                    n_jobs=-1
                )
                
                # Fit the model
                random_search.fit(X_train_use, y_train_balanced)
                
                # Best model predictions
                best_model = random_search.best_estimator_
                y_pred = best_model.predict(X_test_use)
                
                # Calculate metrics
                if problem_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    cv_scores = cross_val_score(best_model, X_train_use, y_train_balanced, cv=cv_folds_use, scoring='accuracy')
                    
                    # Calculate additional metrics for imbalanced data
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    results[name] = {
                        'model': best_model,
                        'best_params': random_search.best_params_,
                        'best_score': random_search.best_score_,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'y_pred': y_pred,
                        'used_smote': use_smote
                    }
                    
                    # Show results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Best CV Score", f"{random_search.best_score_:.4f}")
                    with col2:
                        st.metric("Test Accuracy", f"{accuracy:.4f}")
                    with col3:
                        st.metric("F1 Score", f"{f1:.4f}")
                    with col4:
                        st.metric("CV Mean ± Std", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                    
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    cv_scores = cross_val_score(best_model, X_train_use, y_train_balanced, cv=cv_folds_use, scoring='r2')
                    results[name] = {
                        'model': best_model,
                        'best_params': random_search.best_params_,
                        'best_score': random_search.best_score_,
                        'mse': mse,
                        'r2': r2,
                        'mae': mae,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'y_pred': y_pred,
                        'used_smote': use_smote
                    }
                    
                    # Show results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Best CV R²", f"{random_search.best_score_:.4f}")
                    with col2:
                        st.metric("Test R²", f"{r2:.4f}")
                    with col3:
                        st.metric("MSE", f"{mse:.4f}")
                    with col4:
                        st.metric("MAE", f"{mae:.4f}")
                
                # Show best parameters
                st.subheader("🔧 Best Hyperparameters")
                param_cols = st.columns(min(3, len(random_search.best_params_)))
                for i, (param, value) in enumerate(random_search.best_params_.items()):
                    with param_cols[i % len(param_cols)]:
                        st.write(f"**{param}**: {value}")
                
                st.success(f"✅ {name} training completed!")
                
            except Exception as e:
                st.error(f"❌ Error training {name}: {str(e)}")
                continue
            
            progress_bar.progress((idx + 1) / total_models)
            time.sleep(0.5)  # Small delay for better UX
    
    status_text.text("✅ All models trained successfully!")
    st.session_state.results = results
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.scaler = scaler
    st.session_state.used_smote = use_smote
    
    st.balloons()
    st.success("🎉 Model training completed! Ready for evaluation.")
def step_model_evaluation():
    """Step 5: Model Evaluation and Comparison with support for imbalanced datasets"""
    st.markdown('<div class="section-header">📈 Model Evaluation & Comparison</div>', unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.error("❌ No training results available. Please complete the model training step.")
        return
    
    results = st.session_state.results
    problem_type = st.session_state.problem_type
    y_test = st.session_state.y_test
    used_smote = getattr(st.session_state, 'used_smote', False)
    
    # Show SMOTE usage info
    if used_smote:
        st.info("🔄 Models were trained using SMOTE for handling class imbalance")
    
    # Sort results by performance - use F1 score for imbalanced classification
    if problem_type == 'classification':
        # Check if we have f1_score in results (new version) or use accuracy (old version)
        if 'f1_score' in list(results.values())[0]:
            sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
            primary_metric = 'F1 Score'
        else:
            sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            primary_metric = 'Accuracy'
    else:
        sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
        primary_metric = 'R² Score'
    
    # Performance Overview
    st.markdown('<div class="step-header">🏆 Model Performance Overview</div>', unsafe_allow_html=True)
    
    # Create performance comparison chart
    model_names = [name for name, _ in sorted_results]
    
    if problem_type == 'classification':
        if 'f1_score' in list(results.values())[0]:
            scores = [metrics['f1_score'] for _, metrics in sorted_results]
            accuracy_scores = [metrics['accuracy'] for _, metrics in sorted_results]
            cv_scores = [metrics['cv_mean'] for _, metrics in sorted_results]
            
            # Multiple metrics chart for classification
            fig = go.Figure()
            fig.add_trace(go.Bar(name='F1 Score', x=model_names, y=scores, marker_color='lightcoral'))
            fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=accuracy_scores, marker_color='lightblue'))
            fig.add_trace(go.Bar(name='CV Score', x=model_names, y=cv_scores, marker_color='darkblue'))
            
            fig.update_layout(
                title='Model Performance Comparison (Classification Metrics)',
                xaxis_title='Models',
                yaxis_title='Score',
                barmode='group',
                xaxis_tickangle=45
            )
        else:
            scores = [metrics['accuracy'] for _, metrics in sorted_results]
            cv_scores = [metrics['cv_mean'] for _, metrics in sorted_results]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Test Accuracy', x=model_names, y=scores, marker_color='lightblue'))
            fig.add_trace(go.Bar(name='CV Score', x=model_names, y=cv_scores, marker_color='darkblue'))
            
            fig.update_layout(
                title='Model Performance Comparison (Accuracy)',
                xaxis_title='Models',
                yaxis_title='Accuracy',
                barmode='group',
                xaxis_tickangle=45
            )
    else:
        scores = [metrics['r2'] for _, metrics in sorted_results]
        cv_scores = [metrics['cv_mean'] for _, metrics in sorted_results]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Test R²', x=model_names, y=scores, marker_color='lightgreen'))
        fig.add_trace(go.Bar(name='CV R²', x=model_names, y=cv_scores, marker_color='darkgreen'))
        
        fig.update_layout(
            title='Model Performance Comparison (R² Score)',
            xaxis_title='Models',
            yaxis_title='R² Score',
            barmode='group',
            xaxis_tickangle=45
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Results Table
    st.markdown('<div class="step-header">📊 Detailed Performance Metrics</div>', unsafe_allow_html=True)
    
    if problem_type == 'classification':
        results_data = []
        for name, metrics in sorted_results:
            if 'f1_score' in metrics:  # New version with additional metrics
                results_data.append({
                    'Model': name,
                    'F1 Score': f"{metrics['f1_score']:.4f}",
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'CV Mean': f"{metrics['cv_mean']:.4f}",
                    'CV Std': f"{metrics['cv_std']:.4f}",
                    'Best CV Score': f"{metrics['best_score']:.4f}"
                })
            else:  # Old version
                results_data.append({
                    'Model': name,
                    'Test Accuracy': f"{metrics['accuracy']:.4f}",
                    'CV Mean': f"{metrics['cv_mean']:.4f}",
                    'CV Std': f"{metrics['cv_std']:.4f}",
                    'Best CV Score': f"{metrics['best_score']:.4f}"
                })
    else:
        results_data = []
        for name, metrics in sorted_results:
            results_data.append({
                'Model': name,
                'Test R²': f"{metrics['r2']:.4f}",
                'MSE': f"{metrics['mse']:.4f}",
                'MAE': f"{metrics['mae']:.4f}",
                'CV Mean': f"{metrics['cv_mean']:.4f}",
                'CV Std': f"{metrics['cv_std']:.4f}",
                'Best CV Score': f"{metrics['best_score']:.4f}"
            })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Top 4 Models
    st.markdown('<div class="step-header">🥇 Top 4 Performing Models</div>', unsafe_allow_html=True)
    
    top_4 = sorted_results[:4]
    cols = st.columns(4)
    
    for idx, (name, metrics) in enumerate(top_4):
        with cols[idx]:
            if problem_type == 'classification':
                if 'f1_score' in metrics:
                    score = metrics['f1_score']
                    metric_name = 'F1 Score'
                else:
                    score = metrics['accuracy']
                    metric_name = 'Accuracy'
                
                st.markdown(f'''
                <div class="metric-card">
                    <h4>#{idx+1}</h4>
                    <h3>{name}</h3>
                    <h2>{score:.3f}</h2>
                    <p>{metric_name}</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                score = metrics['r2']
                st.markdown(f'''
                <div class="metric-card">
                    <h4>#{idx+1}</h4>
                    <h3>{name}</h3>
                    <h2>{score:.3f}</h2>
                    <p>R² Score</p>
                </div>
                ''', unsafe_allow_html=True)
    
    # Model-specific Analysis
    st.markdown('<div class="step-header">🔍 Detailed Model Analysis</div>', unsafe_allow_html=True)
    
    selected_model = st.selectbox(
        "Select a model for detailed analysis:",
        [name for name, _ in sorted_results]
    )
    
    if selected_model:
        model_metrics = results[selected_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {selected_model} Performance")
            
            if problem_type == 'classification':
                if 'f1_score' in model_metrics:
                    st.write(f"**F1 Score**: {model_metrics['f1_score']:.4f}")
                    st.write(f"**Accuracy**: {model_metrics['accuracy']:.4f}")
                    st.write(f"**Precision**: {model_metrics['precision']:.4f}")
                    st.write(f"**Recall**: {model_metrics['recall']:.4f}")
                else:
                    st.write(f"**Test Accuracy**: {model_metrics['accuracy']:.4f}")
                
                st.write(f"**CV Mean**: {model_metrics['cv_mean']:.4f}")
                st.write(f"**CV Std**: {model_metrics['cv_std']:.4f}")
                st.write(f"**Best CV Score**: {model_metrics['best_score']:.4f}")
                
                # Enhanced Confusion Matrix with additional metrics
                from sklearn.metrics import confusion_matrix, classification_report
                cm = confusion_matrix(y_test, model_metrics['y_pred'])
                
                # Create confusion matrix heatmap
                fig = px.imshow(cm, text_auto=True, aspect="auto", 
                              title=f"Confusion Matrix - {selected_model}",
                              labels={'x': 'Predicted', 'y': 'Actual'},
                              color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification Report
                st.subheader("📋 Detailed Classification Report")
                try:
                    report = classification_report(y_test, model_metrics['y_pred'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)
                except:
                    st.write("Classification report not available")
                
            else:
                st.write(f"**Test R²**: {model_metrics['r2']:.4f}")
                st.write(f"**MSE**: {model_metrics['mse']:.4f}")
                st.write(f"**MAE**: {model_metrics['mae']:.4f}")
                st.write(f"**CV Mean**: {model_metrics['cv_mean']:.4f}")
                st.write(f"**CV Std**: {model_metrics['cv_std']:.4f}")
                st.write(f"**Best CV Score**: {model_metrics['best_score']:.4f}")
                
                # Actual vs Predicted scatter plot
                fig = px.scatter(
                    x=y_test, y=model_metrics['y_pred'],
                    title=f"Actual vs Predicted - {selected_model}",
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                )
                fig.add_shape(
                    type="line", line=dict(dash="dash"),
                    x0=y_test.min(), y0=y_test.min(),
                    x1=y_test.max(), y1=y_test.max()
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔧 Best Hyperparameters")
            for param, value in model_metrics['best_params'].items():
                st.write(f"**{param}**: {value}")
            
            # Performance comparison with other models
            st.subheader("📈 Performance Ranking")
            for idx, (name, _) in enumerate(sorted_results, 1):
                if name == selected_model:
                    st.write(f"**#{idx} {name}** ⭐")
                else:
                    st.write(f"#{idx} {name}")
            
            # Show if SMOTE was used
            if used_smote and problem_type == 'classification':
                st.subheader("🔄 Data Balancing")
                st.info("✅ SMOTE was applied during training")
    
    # Class imbalance analysis for classification
    if problem_type == 'classification':
        st.markdown('<div class="step-header">⚖️ Class Imbalance Analysis</div>', unsafe_allow_html=True)
        
        from collections import Counter
        test_class_counts = Counter(y_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Test Set Class Distribution")
            for class_val, count in test_class_counts.items():
                percentage = (count / len(y_test)) * 100
                st.write(f"**Class {class_val}**: {count} ({percentage:.1f}%)")
            
            balance_ratio = min(test_class_counts.values()) / max(test_class_counts.values())
            st.metric("Class Balance Ratio", f"{balance_ratio:.3f}")
        
        with col2:
            st.subheader("🎯 Recommended Metrics")
            if balance_ratio < 0.5:
                st.warning("⚠️ Imbalanced test set detected!")
                st.write("**Recommended metrics for evaluation:**")
                st.write("- 🎯 **F1 Score**: Balances precision and recall")
                st.write("- 📊 **Precision**: Important for false positives")
                st.write("- 🔍 **Recall**: Important for false negatives")
                st.write("- ⚖️ **ROC-AUC**: Overall discrimination ability")
            else:
                st.success("✅ Balanced test set")
                st.write("**Standard metrics are reliable:**")
                st.write("- ✅ **Accuracy**: Good overall measure")
                st.write("- 📊 **F1 Score**: Additional validation")
    
    # Set best model for final results
    best_model_name = sorted_results[0][0]
    st.session_state.best_model = {
        'name': best_model_name,
        'metrics': sorted_results[0][1],
        'primary_metric': primary_metric
    }
def step_final_results():
    """Step 6: Final Results and Model Insights with support for imbalanced datasets"""
    st.markdown('<div class="section-header">🏆 Final Results & Model Insights</div>', unsafe_allow_html=True)
    
    if st.session_state.best_model is None:
        st.error("❌ No best model selected. Please complete the evaluation step.")
        return
    
    best_model_info = st.session_state.best_model
    best_model_name = best_model_info['name']
    best_metrics = best_model_info['metrics']
    problem_type = st.session_state.problem_type
    primary_metric = best_model_info.get('primary_metric', 'Accuracy' if problem_type == 'classification' else 'R² Score')
    used_smote = getattr(st.session_state, 'used_smote', False)
    
    # Winner announcement
    st.markdown(f'<div class="step-header">🎉 Winner: {best_model_name}</div>', unsafe_allow_html=True)
    
    # Show SMOTE usage if applicable
    if used_smote:
        st.info("🔄 This model was trained using SMOTE for handling class imbalance")
    
    # Performance summary - adapt based on available metrics
    if problem_type == 'classification':
        if 'f1_score' in best_metrics:
            # Enhanced metrics for imbalanced classification
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>F1 Score</h3>
                    <h1>{best_metrics['f1_score']:.4f}</h1>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h1>{best_metrics['accuracy']:.4f}</h1>
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Precision</h3>
                    <h1>{best_metrics['precision']:.4f}</h1>
                </div>
                ''', unsafe_allow_html=True)
            with col4:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Recall</h3>
                    <h1>{best_metrics['recall']:.4f}</h1>
                </div>
                ''', unsafe_allow_html=True)
        else:
            # Basic metrics for balanced classification
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Test Accuracy</h3>
                    <h1>{best_metrics['accuracy']:.4f}</h1>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>CV Score</h3>
                    <h1>{best_metrics['cv_mean']:.4f}</h1>
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Best Params Score</h3>
                    <h1>{best_metrics['best_score']:.4f}</h1>
                </div>
                ''', unsafe_allow_html=True)
    else:
        # Regression metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>R² Score</h3>
                <h1>{best_metrics['r2']:.4f}</h1>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <h3>MSE</h3>
                <h1>{best_metrics['mse']:.4f}</h1>
            </div>
            ''', unsafe_allow_html=True)
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <h3>MAE</h3>
                <h1>{best_metrics['mae']:.4f}</h1>
            </div>
            ''', unsafe_allow_html=True)
    
    # Model insights and facts
    st.markdown('<div class="step-header">🧠 Model Insights & Facts</div>', unsafe_allow_html=True)
    
    if best_model_name in MODEL_FACTS:
        facts = MODEL_FACTS[best_model_name]
        
        st.subheader("🎯 Why this model performed best:")
        for i, fact in enumerate(facts, 1):
            st.write(f"**{i}.** {fact}")
    
    # Add SMOTE-specific insights if used
    if used_smote and problem_type == 'classification':
        st.subheader("🔄 SMOTE Impact:")
        st.write("""
        **1.** SMOTE (Synthetic Minority Oversampling Technique) created synthetic samples for minority classes
        **2.** This helps the model learn better decision boundaries for underrepresented classes
        **3.** F1 Score and balanced metrics become more reliable indicators of performance
        **4.** The model should now perform better on minority class predictions
        """)
    
    # Hyperparameters used
    st.subheader("⚙️ Optimal Hyperparameters")
    best_params_df = pd.DataFrame([
        {'Parameter': param, 'Value': value} 
        for param, value in best_metrics['best_params'].items()
    ])
    st.table(best_params_df)
    
    # Performance interpretation
    if problem_type == 'classification' and 'f1_score' in best_metrics:
        st.markdown('<div class="step-header">📊 Performance Interpretation</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Metric Meanings")
            st.write(f"**F1 Score ({best_metrics['f1_score']:.4f})**:")
            if best_metrics['f1_score'] > 0.8:
                st.success("Excellent balance of precision and recall")
            elif best_metrics['f1_score'] > 0.6:
                st.info("Good balance of precision and recall")
            else:
                st.warning("Room for improvement in precision/recall balance")
            
            st.write(f"**Precision ({best_metrics['precision']:.4f})**:")
            if best_metrics['precision'] > 0.8:
                st.success("Low false positive rate")
            elif best_metrics['precision'] > 0.6:
                st.info("Moderate false positive rate")
            else:
                st.warning("Higher false positive rate")
            
            st.write(f"**Recall ({best_metrics['recall']:.4f})**:")
            if best_metrics['recall'] > 0.8:
                st.success("Low false negative rate")
            elif best_metrics['recall'] > 0.6:
                st.info("Moderate false negative rate")
            else:
                st.warning("Higher false negative rate")
        
        with col2:
            st.subheader("📈 Business Impact")
            st.write("**For Imbalanced Data:**")
            st.write("• F1 Score is the most reliable metric")
            st.write("• High precision = fewer false alarms")
            st.write("• High recall = fewer missed cases")
            st.write("• SMOTE helps with minority class detection")
            
            if used_smote:
                st.success("✅ SMOTE was used to address class imbalance")
            else:
                st.info("💡 Consider using SMOTE for better minority class performance")
    
    # Final recommendations
    st.markdown('<div class="step-header">💡 Recommendations & Next Steps</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Model Deployment Tips")
        if used_smote:
            st.write("""
            1. **SMOTE Preprocessing**: Apply same SMOTE strategy to new data if needed
            2. **Feature Scaling**: Remember to apply the same scaling to new data
            3. **Model Persistence**: Save both the model and preprocessing pipeline
            4. **Monitoring**: Track precision, recall, and F1 score in production
            5. **Class Distribution**: Monitor if new data maintains similar class balance
            """)
        else:
            st.write("""
            1. **Feature Scaling**: Remember to apply the same scaling to new data
            2. **Model Persistence**: Save the trained model using joblib or pickle
            3. **Monitoring**: Track model performance over time in production
            4. **Retraining**: Consider retraining with new data periodically
            """)
    
    with col2:
        st.subheader("📈 Performance Optimization")
        if problem_type == 'classification' and used_smote:
            st.write("""
            1. **Advanced SMOTE**: Try ADASYN or BorderlineSMOTE variants
            2. **Cost-Sensitive Learning**: Use class weights in algorithms
            3. **Ensemble Methods**: Combine multiple balanced models
            4. **Threshold Tuning**: Optimize classification threshold for F1 score
            5. **Feature Engineering**: Create features that help distinguish minority class
            """)
        else:
            st.write("""
            1. **Feature Engineering**: Create new features from existing ones
            2. **More Data**: Collect additional training samples if possible
            3. **Advanced Tuning**: Try Bayesian optimization for hyperparameters
            4. **Ensemble Methods**: Combine multiple models for better results
            """)
    
    # Download section
    st.markdown('<div class="step-header">💾 Download Results</div>', unsafe_allow_html=True)
    
    # Create comprehensive results summary
    summary = f"""
# AutoML Pipeline Results Summary

## Dataset Information
- Original Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns
- Processed Shape: {st.session_state.processed_df.shape[0]} rows × {st.session_state.processed_df.shape[1]} columns
- Target Column: {st.session_state.target_col}
- Problem Type: {problem_type.title()}
- SMOTE Applied: {"Yes" if used_smote else "No"}

## Best Model: {best_model_name}
"""
    
    if problem_type == 'classification':
        if 'f1_score' in best_metrics:
            summary += f"""
- F1 Score: {best_metrics['f1_score']:.4f}
- Test Accuracy: {best_metrics['accuracy']:.4f}
- Precision: {best_metrics['precision']:.4f}
- Recall: {best_metrics['recall']:.4f}
- Cross-Validation Mean: {best_metrics['cv_mean']:.4f}
- Cross-Validation Std: {best_metrics['cv_std']:.4f}
- Best CV Score: {best_metrics['best_score']:.4f}
"""
        else:
            summary += f"""
- Test Accuracy: {best_metrics['accuracy']:.4f}
- Cross-Validation Mean: {best_metrics['cv_mean']:.4f}
- Cross-Validation Std: {best_metrics['cv_std']:.4f}
- Best CV Score: {best_metrics['best_score']:.4f}
"""
    else:
        summary += f"""
- R² Score: {best_metrics['r2']:.4f}
- Mean Squared Error: {best_metrics['mse']:.4f}
- Mean Absolute Error: {best_metrics['mae']:.4f}
- Cross-Validation Mean: {best_metrics['cv_mean']:.4f}
- Cross-Validation Std: {best_metrics['cv_std']:.4f}
- Best CV Score: {best_metrics['best_score']:.4f}
"""
    
    summary += f"""
## Best Hyperparameters
"""
    for param, value in best_metrics['best_params'].items():
        summary += f"- {param}: {value}\n"
    
    if used_smote:
        summary += f"""
## Data Balancing
- SMOTE was applied to handle class imbalance
- This improves minority class detection
- F1 Score is the primary evaluation metric
"""
    
    summary += f"""
## Model Facts
"""
    if best_model_name in MODEL_FACTS:
        for i, fact in enumerate(MODEL_FACTS[best_model_name], 1):
            summary += f"{i}. {fact}\n"
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📋 Download Results Summary",
            data=summary,
            file_name=f"automl_results_{best_model_name}_{'smote_' if used_smote else ''}{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Create a CSV with all model results
        all_results_data = []
        for name, metrics in st.session_state.results.items():
            if problem_type == 'classification':
                if 'f1_score' in metrics:
                    all_results_data.append({
                        'Model': name,
                        'F1_Score': metrics['f1_score'],
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'CV_Mean': metrics['cv_mean'],
                        'CV_Std': metrics['cv_std'],
                        'Best_CV_Score': metrics['best_score'],
                        'Used_SMOTE': metrics.get('used_smote', False)
                    })
                else:
                    all_results_data.append({
                        'Model': name,
                        'Test_Accuracy': metrics['accuracy'],
                        'CV_Mean': metrics['cv_mean'],
                        'CV_Std': metrics['cv_std'],
                        'Best_CV_Score': metrics['best_score'],
                        'Used_SMOTE': metrics.get('used_smote', False)
                    })
            else:
                all_results_data.append({
                    'Model': name,
                    'Test_R2': metrics['r2'],
                    'MSE': metrics['mse'],
                    'MAE': metrics['mae'],
                    'CV_Mean': metrics['cv_mean'],
                    'CV_Std': metrics['cv_std'],
                    'Best_CV_Score': metrics['best_score'],
                    'Used_SMOTE': metrics.get('used_smote', False)
                })
        
        results_csv = pd.DataFrame(all_results_data).to_csv(index=False)
        st.download_button(
            label="📊 Download All Results (CSV)",
            data=results_csv,
            file_name=f"all_model_results_{'smote_' if used_smote else ''}{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Success message
    st.success("🎉 AutoML Pipeline completed successfully! Your model is ready for deployment.")
    
    # Reset option
    if st.button("🔄 Start New Analysis", type="secondary", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
