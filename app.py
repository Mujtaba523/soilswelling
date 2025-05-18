import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from io import BytesIO
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import uuid
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Shale Swelling Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
        color: #FFFFFF;
    }
    .sub-section-header {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
        color: #FFFFFF;
    }
    .info-text {
        font-size: 0.9rem;
        color: #FFFFFF;
    }
    .divider {
        height: 2px;
        background-color: #FFFFFF;
        margin: 1.5rem 0;
        border-radius: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        font-size: 1rem;
    }
    /* Make dataframes more compact */
    .dataframe-container {
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 0.5rem;
        background-color: white;
    }
    /* Responsive chart container */
    .chart-container {
        width: 100%;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def preprocess_data(df):
    try:
        df = df.copy()
        df.dropna(inplace=True)
        df['Elap Time'] = pd.to_timedelta(df['Elap Time'].astype(str), errors='coerce')
        df['Seconds'] = df['Elap Time'].dt.total_seconds()
        df.dropna(inplace=True)
        df.set_index('Seconds', inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

# Load all sheets from Excel
@st.cache_data
def load_data(file_path):
    try:
        sheet_names = ['Talhar', 'Ranikot', 'Khadro', 'Ghazij']
        return {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheet_names}, None
    except Exception as e:
        return None, str(e)
    

# Function to make predictions
def predict_swelling(region, time_input, models):
    try:
        # First try to use custom model if available
        custom_model_key = f"{region}_rf_custom"
        if custom_model_key in models:
            rf_model = models[custom_model_key]
            st.info("Using custom trained model for prediction.")
        else:
            # Fall back to original model
            rf_model = models.get(f"{region}_rf")
            
        if rf_model is None:
            return None, "Random Forest model not found for this region"
        
        # Predict
        rf_pred = rf_model.predict(np.array([[time_input]]))[0]
        return rf_pred, None
    except Exception as e:
        return None, str(e)


# Function to train a new Random Forest model
def train_rf_model(df, test_size=0.2, n_estimators=100, max_depth=None, random_state=42):
    try:
        # Prepare data
        X = df.index.values.reshape(-1, 1)  # Time in seconds as feature
        y = df['Swell (%)'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return model, {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'test_size': test_size,
            'n_estimators': n_estimators,
            'max_depth': max_depth if max_depth else "None (unlimited)",
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    except Exception as e:
        return None, str(e)

# Function to save the trained model
def save_model(model, region, model_type='rf'):
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Generate a unique model name with timestamp
        model_id = str(uuid.uuid4())[:8]
        model_path = f"models/{region}_{model_type}_{model_id}.pkl"
        
        # Save the model
        joblib.dump(model, model_path)
        
        return model_path, None
    except Exception as e:
        return None, str(e)

# Main application
def main():
    # Initialize session state for tab selection
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Initialize models dictionary
    if 'models' not in st.session_state:
        st.session_state.models = {}
    
    # Header
    st.markdown('<div class="main-header">Shale Swelling Prediction Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source",
            ["Built-in Data"],
            index=0
        )
        
        if data_source == "Built-in Data":
            # Load internal Excel file
            file_path = 'different shales.xlsx'
            
            # Check if file exists
            if not os.path.exists(file_path):
                st.error(f"Built-in data file not found at: {file_path}")
                st.info("Please update the file path or upload your own data.")
                data, error = None, "File not found"
            else:
                data, error = load_data(file_path)
            
            if error:
                st.error(f"Error loading built-in data: {error}")
                return
            
            if data:
                region = st.selectbox("Select Shale Region", list(data.keys()))
                df, preprocess_error = preprocess_data(data[region])
                
                if preprocess_error:
                    st.error(f"Error preprocessing data: {preprocess_error}")
                    return
        
        # Check if models are loaded
        if region not in st.session_state.models:
            with st.spinner(f"Loading models for {region}..."):
                try:
                    # Try to load models
                    model_types = ['rf', 'lr', 'arima', 'symbolic']
                    for model_type in model_types:
                        model_path = f"models/{region}_{model_type}.pkl"
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                            st.session_state.models[f"{region}_{model_type}"] = model
                    
                    if f"{region}_rf" not in st.session_state.models:
                        st.warning(f"Random Forest model for {region} not found. Some features may not work.")
                except Exception as e:
                    st.error(f"Error loading models: {str(e)}")
    
    st.markdown('<div class="section-header">Swelling Prediction</div>', unsafe_allow_html=True)
    
    if 'df' not in locals():
        st.warning("Please select a data source and region to continue.")
        return
    
    # Create 3 columns for time input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hours = st.number_input("Hours", min_value=0, max_value=23, step=1)
    
    with col2:
        minutes = st.number_input("Minutes", min_value=0, max_value=59, step=1)
    
    with col3:
        seconds = st.number_input("Seconds", min_value=0, max_value=59, step=1)
    
    # Convert to total seconds
    time_input = int(hours * 3600 + minutes * 60 + seconds)
    
    # Center the predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("Predict Swelling", use_container_width=True)
    
    if predict_button:
        with st.spinner("Calculating prediction..."):
            # Make prediction
            prediction, error = predict_swelling(region, time_input, st.session_state.models)
            
            if error:
                st.error(f"Error making prediction: {error}")
            else:
                # Display prediction in a nice format
                st.markdown(
                    f"""
                    <div class="highlight">
                        <div class="sub-section-header">Prediction Results</div>
                        <p>For time input: {hours:02d}:{minutes:02d}:{seconds:02d}</p>
                        <h3>Random Forest Prediction: {prediction:.2f}%</h3>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    
    # Divider between prediction and forecast
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Data Analysis</div>', unsafe_allow_html=True)
    
    if 'df' not in locals():
        st.warning("Please select a data source and region to continue.")
        return
    
    # Display data info
    st.markdown(f"<div class='info-text'>Showing data for region: <b>{region}</b></div>", unsafe_allow_html=True)
    
    # Display data statistics
    col1, col2 = st.columns(2)
    
    # Fixed number of rows to display
    num_rows_to_display = 5
    
    with col1:
        st.markdown('<div class="sub-section-header">Data Preview</div>', unsafe_allow_html=True)
        
        # Reset index for display
        display_df = df.reset_index()
        
        # Convert seconds back to time format for display
        display_df['Time'] = pd.to_timedelta(display_df['Seconds'], unit='s')
        display_df['Time'] = display_df['Time'].apply(lambda x: str(x).split('.')[0])  # Remove microseconds
        
        # Select only relevant columns
        display_df = display_df[['Time', 'Swell (%)']]
        
        # Rename columns for clarity
        display_df.columns = ['Elapsed Time', 'Swell (%)']
        
        # Display fixed number of rows
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(display_df.head(num_rows_to_display), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sub-section-header">Statistics</div>', unsafe_allow_html=True)
        
        # Calculate statistics
        stats = {
            'Total Data Points': len(df),
            'Min Swelling (%)': df['Swell (%)'].min(),
            'Max Swelling (%)': df['Swell (%)'].max(),
            'Average Swelling (%)': df['Swell (%)'].mean(),
            'Standard Deviation': df['Swell (%)'].std()
        }
        
        # Display statistics
        stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        
        # Pad the stats dataframe to match the number of rows in the preview
        if len(stats_df) < num_rows_to_display:
            padding = pd.DataFrame([['', ''] for _ in range(num_rows_to_display - len(stats_df))], 
                                    columns=['Metric', 'Value'])
            stats_df = pd.concat([stats_df, padding], ignore_index=True)
        
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(stats_df.head(num_rows_to_display), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data visualization


    # Plot the data using Plotly
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Swell (%)'],
        mode='lines+markers',
        marker=dict(size=6),
        name='Swell (%)'
    ))
    fig.update_layout(
        title=f'Swelling Progression for {region} Shale',
        xaxis_title='Time (seconds)',
        yaxis_title='Swell (%)',
        template='plotly_white',
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Create two columns for additional analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-section-header">Rate of Change</div>', unsafe_allow_html=True)

        if len(df) > 1:
            df_diff = df.copy()
            df_diff['Rate of Change'] = df_diff['Swell (%)'].diff() / df_diff.index.to_series().diff()
            df_diff = df_diff.dropna()

            fig_diff = go.Figure()
            fig_diff.add_trace(go.Scatter(
                x=df_diff.index,
                y=df_diff['Rate of Change'],
                mode='lines+markers',
                marker=dict(size=4, color='green'),
                name='Rate of Change'
            ))
            fig_diff.update_layout(
                title='Rate of Swelling Change',
                xaxis_title='Time (seconds)',
                yaxis_title='Rate (%/second)',
                template='plotly_white',
                height=300,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_diff, use_container_width=True)
        else:
            st.info("Not enough data points to calculate rate of change.")

    with col2:
        st.markdown('<div class="sub-section-header">Distribution</div>', unsafe_allow_html=True)

        fig_hist = px.histogram(
            df,
            x='Swell (%)',
            nbins=10,
            title='Distribution of Swelling Values',
            opacity=0.75,
            color_discrete_sequence=['purple']
        )
        fig_hist.update_layout(
            xaxis_title='Swell (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_hist, use_container_width=True)


if __name__ == "__main__":
    main()
