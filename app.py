"""
Streamlit Application for Earnings Call Transcript Analysis & Forecasting
Upload PDF transcripts for insights or CSVs for ML-powered forecasts.
"""
import sys
print(sys.executable)

import streamlit as st
import io
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

# Import both the analyzer function AND the new Forecaster class
from analyzer import analyze_transcript, Forecaster

# Page configuration
st.set_page_config(
    page_title="Earnings Call Analyzer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
    <style>
    .main-header { ... } 
    /* ... (all your other custom CSS) ... */
    </style>
""", unsafe_allow_html=True)


# --- GAUGE CHART FUNCTION (Same as before) ---
def create_sentiment_gauge(score, label, title):
    """Creates a gauge chart for sentiment visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        # ... (rest of the gauge code) ...
    ))
    return fig


# --- DISPLAY FUNCTIONS (Same as before) ---
def display_investment_view(view_data):
    """Displays the investment view with visual styling."""
    # ... (all your display_investment_view code) ...

def display_financial_metrics(financials):
    """Displays financial metrics in a structured format."""
    # ... (all your display_financial_metrics code) ...

def display_commentary_analysis(commentary):
    """Displays commentary analysis with expandable sections."""
    # ... (all your display_commentary_analysis code) ...

def display_sentiment_analysis(sentiment):
    """Displays sentiment analysis with gauges."""
    # ... (all your display_sentiment_analysis code) ...

def display_risk_analysis(risks):
    """Displays risk analysis."""
    # ... (all your display_risk_analysis code) ...

def export_to_json(analysis_results, company_name="Company"):
    """Exports analysis results to JSON."""
    # ... (all your export_to_json code) ...


# --- NEW: SAMPLE DATA FUNCTION (For Forecaster) ---
@st.cache_data
def get_sample_data():
    """Returns a sample CSV string for the forecaster."""
    sample = """ds,y
2020-03-31,100
2020-06-30,110
2020-09-30,105
2020-12-31,120
2021-03-31,115
2021-06-30,125
2021-09-30,130
2021-12-31,140
2022-03-31,135
2022-06-30,150
2022-09-30,155
2022-12-31,160
"""
    return sample


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">🔮 Earnings Analyzer & Forecaster</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered insights from transcripts and historical data</p>', 
                unsafe_allow_html=True)
    
    # Sidebar (Same as before)
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/stock-market.png", width=80)
        st.title("About")
        st.info(
            "This tool provides two functions:\n\n"
            "**1. ML Forecaster:**\n"
            "Upload a CSV of historical data (e.g., quarterly revenue) to "
            "generate an ML-powered forecast.\n\n"
            "**2. Transcript Analyzer:**\n"
            "Upload a PDF earnings transcript to extract insights, sentiment, "
            "and risk factors."
        )
        
        st.markdown("---")
        st.subheader("Settings")
        company_name = st.text_input("Company Name (Optional)", placeholder="e.g., ABC Corp")
        show_raw_data = st.checkbox("Show Raw Analysis Data", value=False)
        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit, Prophet, and Transformers")
    
    
    # --- NEW: SECTION 1 - ML FORECASTER ---
    st.header("🔮 1. ML-Powered Financial Forecaster")
    st.info("Upload a CSV with `ds` (date) and `y` (value) columns to predict future trends.")

    # Add a download button for the sample CSV
    st.download_button(
        label="Download Sample CSV Template",
        data=get_sample_data(),
        file_name="sample_financial_data.csv",
        mime="text/csv",
    )
    
    hist_file = st.file_uploader("Upload Historical Data (CSV)", type="csv", key="csv_uploader")
    
    if hist_file is not None:
        with st.spinner("Training ML model and generating forecast..."):
            try:
                # --- Call the Forecaster class ---
                forecast_fig = Forecaster.create_forecast(hist_file)
                st.subheader("Forecast Results")
                st.plotly_chart(forecast_fig, use_container_width=True)
                
            except ValueError as ve:
                st.error(f"Error processing CSV: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.exception(e)
    
    st.divider() # --- SEPARATOR ---

    # --- SECTION 2 - TRANSCRIPT ANALYZER ---
    st.header("📊 2. Earnings Call Transcript Analyzer")
    
    uploaded_file = st.file_uploader(
        "Upload Earnings Call Transcript (PDF)",
        type=['pdf'],
        help="Upload a PDF file containing the earnings call transcript"
    )
    
    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB",
            "File Type": uploaded_file.type
        }
        
        with st.expander("📄 File Details"):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
        
        if st.button("🚀 Analyze Transcript", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing transcript... This may take a minute..."):
                try:
                    uploaded_file.seek(0)
                    results = analyze_transcript(uploaded_file)
                    
                    if 'error' in results:
                        st.error(f"❌ Analysis Error: {results['error']}")
                        return
                    
                    st.session_state['analysis_results'] = results
                    st.session_state['company_name'] = company_name or "Company"
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.exception(e)
    
    # Display results if available
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        company = st.session_state.get('company_name', 'Company')
        
        st.markdown("---")
        st.header(f"📈 Analysis Results{f' - {company}' if company != 'Company' else ''}")
        
        if 'investment_view' in results:
            display_investment_view(results['investment_view'])
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "💰 Financial", 
            "💬 Commentary", 
            "🎭 Sentiment", 
            "⚠️ Risks"
        ])
        
        with tab1:
            display_financial_metrics(results.get('Financial', {}))
        with tab2:
            display_commentary_analysis(results.get('Commentary', {}))
        with tab3:
            display_sentiment_analysis(results.get('Sentiment', {}))
        with tab4:
            display_risk_analysis(results.get('Risks', []))
        
        if show_raw_data:
            st.markdown("---")
            st.subheader("🔍 Raw Analysis Data")
            st.json(results)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            json_data = export_to_json(results, company)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_data,
                file_name=f"{company.replace(' ', '_')}_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with col2:
            if st.button("🔄 Analyze Another Transcript"):
                del st.session_state['analysis_results']
                if 'company_name' in st.session_state:
                    del st.session_state['company_name']
                st.rerun()
    
    # Instructions for first-time users
    if uploaded_file is None and 'analysis_results' not in st.session_state and hist_file is None:
        st.markdown("---")
        st.info("👆 **Get Started:** Upload a CSV for forecasting or a PDF for analysis.")
        
        with st.expander("📖 How to Use This Tool"):
            st.markdown("""
            ### Step-by-Step Guide:
            
            **For ML Forecasting:**
            1.  Create a CSV file with `ds` (YYYY-MM-DD) and `y` (numeric value) columns.
            2.  Upload your CSV to the "ML Financial Forecaster" section.
            3.  Review the interactive forecast graph.
            
            **For Transcript Analysis:**
            1.  Upload a PDF transcript to the "Transcript Analyzer" section.
            2.  Click the "Analyze Transcript" button.
            3.  Review the results in the tabs below.
            """)

if __name__ == "__main__":
    main()