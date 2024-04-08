import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import altair as alt

def calculate_statistics_percentages(daily_returns, daily_log_returns): 
    # Calculate statistics (returns and volatility, shown as percentage to 4 decimals)
    statistics_percentages = {
        'Mean Daily Returns': daily_returns.mean() * 100,
        'Median Daily Return': daily_returns.median() * 100,
        'Mean Absolute Deviation (MAD)': np.mean(np.abs(daily_returns - daily_returns.mean())) * 100,
        'Average Daily Absolute Return': abs(daily_returns).mean() * 100,
        'Average Daily Gain': daily_returns[daily_returns > 0].mean() * 100,
        'Average Daily Loss': daily_returns[daily_returns < 0].mean() * 100,
        'Std Dev Daily Returns': daily_returns.std() * 100,
        'Largest Daily Gain': daily_returns.max() * 100,
        'Largest Daily Loss': daily_returns.min() * 100,
        'Maximum Drawdown': max_drawdown(daily_returns) * 100,  # Maximum Drawdown
        'Annual Volatility': daily_returns.std() * np.sqrt(252) * 100,  # Annualize by multiplying with square root of trading days
        'Daily Value at Risk (95%)': daily_returns.quantile(0.05) * 100,
        'Daily Value at Risk (99%)': daily_returns.quantile(0.01) * 100,
        'Conditional Value at Risk (95%)': daily_returns[daily_returns <= daily_returns.quantile(0.05)].mean() * 100,
        'Conditional Value at Risk (99%)': daily_returns[daily_returns <= daily_returns.quantile(0.01)].mean() * 100,
        'Interquartile Range (IQR)': (daily_returns.quantile(0.75) - daily_returns.quantile(0.25)) * 100,
        'Geometric Mean': (np.exp(np.mean(np.log(1 + daily_returns))) - 1) * 100,
    }
    
    return statistics_percentages


def calculate_statistics_ratios(daily_returns, daily_log_returns): 
    # Calculate statistics (ratios, shown as real number to 4 decimals)
    statistics_ratios = {
        'Skewness of Daily Returns': skew(daily_returns),
        'Kurtosis of Daily Returns': kurtosis(daily_returns),
        'Sharpe Ratio': np.sqrt(252) * daily_log_returns.mean() / daily_log_returns.std(),  # Assuming 252 trading days
        'Sortino Ratio': np.sqrt(252) * daily_log_returns.mean() / daily_log_returns[daily_log_returns < 0].std(),  # Sortino for downside risk
        'Average Gain-to-Pain Ratio': daily_returns[daily_returns > 0].mean() / abs(daily_returns[daily_returns < 0].mean()),
    }
    
    return statistics_ratios

def max_drawdown(returns):
    """
    Calculate the maximum drawdown of a series of returns.
    """
    wealth_index = 1000 * (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns.min()

def main():
    st.title("Martin Robertson Risk App")

    # File selection
    file_path = st.file_uploader("Upload CSV file", type="csv")

    if file_path is not None:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Show the raw data
        st.subheader("Raw Data")
        st.write(df)

        # Check if 'timestamp' column is available
        if 'timestamp' in df.columns:
            # Convert 'timestamp' column to datetime with correct format
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y')
            
            # Line chart
            st.subheader("Open, High, Low, Close Prices Over Time")
            st.line_chart(df[['timestamp', 'open', 'high', 'low', 'close']].set_index('timestamp'))
        else:
            st.error("The 'timestamp' column is missing in the CSV file. Please ensure it is included.")

        # Check if 'close' column is available
        if 'close' in df.columns:
            # Calculate daily returns
            df['Daily Returns'] = df['close'].pct_change()
            df['Daily Log Returns'] = np.log(df['close'] / df['close'].shift(1))

            # Calculate monthly and yearly returns
            df['Monthly Log Returns'] = np.log(df['close'] / df['close'].shift(30))
            df['Yearly Log Returns'] = np.log(df['close'] / df['close'].shift(360))
            
            # Drop NaN values
            df.dropna(subset=['Daily Returns', 'Daily Log Returns'], inplace=True)

            # Calculate and display statistics
            statistics_percentages = calculate_statistics_percentages(df['Daily Returns'], df['Daily Log Returns'])
            statistics_ratios = calculate_statistics_ratios(df['Daily Returns'], df['Daily Log Returns'])
            
            # Create two separate tables for percentages and ratios
            stats_percentages_df = pd.DataFrame.from_dict(statistics_percentages, orient='index', columns=['Returns/Volatility (%)']).round(2)
            stats_ratios_df = pd.DataFrame.from_dict(statistics_ratios, orient='index', columns=['Ratio']).round(2)

            # Display the tables
            st.subheader("Statistics of Daily Returns (Returns/Volatility, %)")
            st.table(stats_percentages_df.style.format({'Returns/Volatility (%)': '{:.2f}%'}))

            st.subheader("Statistics of Daily Returns (Risk Ratios)")
            st.table(stats_ratios_df.style.format({'Ratio': '{:.2f}'}))

            # Calculate statistics for discrete years
            df['Year'] = pd.to_datetime(df['timestamp']).dt.year
            years = df['Year'].unique()

            stats_by_year_percentages = []
            for year in years:
                year_data = df[df['Year'] == year]
                statistics_percentages = calculate_statistics_percentages(year_data['Daily Returns'], year_data['Daily Log Returns'])
                stats_by_year_percentages.append({**statistics_percentages, 'Year': year})

            stats_by_year_ratios = []
            for year in years:
                year_data = df[df['Year'] == year]
                statistics_ratios = calculate_statistics_ratios(year_data['Daily Returns'], year_data['Daily Log Returns'])
                stats_by_year_ratios.append({**statistics_ratios, 'Year': year})

            # Create DataFrames for years
            stats_by_year_percentages_df = pd.DataFrame(stats_by_year_percentages).set_index('Year').transpose()
            stats_by_year_ratios_df = pd.DataFrame(stats_by_year_ratios).set_index('Year').round(2).transpose()

            # Display the tables for each
            st.subheader("Statistics by Year (Returns / Volatility, %)")
            st.table(stats_by_year_percentages_df.style.format(
                '{:.2f}'
            ).set_table_styles(
                [{'selector': '.index_name', 'props': [('width', '1100px')]},  # Wider first column
                 {'selector': 'td', 'props': [('max-width', '44px')], 'index': 1}]  # Narrower data columns except the first
            ))

            st.subheader("Statistics by Year (Ratios)")
            st.table(stats_by_year_ratios_df.style.format(
                '{:.2f}'
            ).set_table_styles(
                [{'selector': '.index_name', 'props': [('width', '1100px')]},  # Wider first column
                 {'selector': 'td', 'props': [('max-width', '44px')], 'index': 1}]  # Narrower data columns except the first
            ))

            # Footnote
            footnote = """
            - *Skewness*: Positive skewness (longer right tail) suggests more frequent small losses and occasional large gains, while negative skewness (longer left tail) suggests more frequent small gains and occasional large losses.
            - *Kurtosis*: High kurtosis (>3) indicates heavy tails with more outliers or extreme events. Low kurtosis (<3) indicates light tails with fewer outliers or extreme moves.
            """

            # Display the footnote
            st.markdown(footnote, unsafe_allow_html=True)
            
           # Custom date range for chart
            st.sidebar.subheader("Select Date Range for Chart")
            today = pd.Timestamp.today().date()
            start_date = pd.to_datetime(st.sidebar.date_input("Start Date", min_value=pd.to_datetime(df['timestamp']).min(), 
                                              max_value=pd.to_datetime(today), 
                                              value =pd.to_datetime(df['timestamp']).min()))
            end_date = pd.to_datetime(st.sidebar.date_input("End Date", min_value=pd.to_datetime(df['timestamp']).min(), 
                                            max_value=pd.to_datetime(today), 
                                            value=pd.to_datetime(df['timestamp']).max()))

            # Filter data based on selected date range
            filtered_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

            # Normalize close prices relative to the start date
            start_close = filtered_data.iloc[0]['close']
            filtered_data['Normalized Close'] = (filtered_data['close'] / start_close) * 100

            # Line chart with custom date range and normalised close prices
            st.subheader("Custom Date Range Chart (Normalised, Close prices)")
            st.line_chart(filtered_data[['timestamp', 'Normalized Close']].set_index('timestamp'))
            
            # Calculate drawdowns as negative percentage
            filtered_data['Previous High'] = filtered_data['close'].cummax()
            filtered_data['Drawdown'] = ((filtered_data['close'] / filtered_data['Previous High']) - 1) * 100

            # Line chart for drawdowns
            st.subheader("Cumulative Drawdown, from previous high")

            # Line chart for drawdowns
            drawdown_chart = alt.Chart(filtered_data).mark_line(color='#B03A2E').encode(
                x='timestamp:T',
                y='Drawdown:Q',
                tooltip=['timestamp', 'Drawdown']
            ).properties(
                width=700,
                height=300
            )

            # Area chart for shading in grayish-purple color
            drawdown_area = alt.Chart(filtered_data).mark_area(opacity=0.5, color='#9B59B6').encode(
                x='timestamp:T',
                y='Drawdown:Q',
            )

            st.altair_chart(drawdown_chart + drawdown_area, use_container_width=True)
            
        else:
            st.error("The 'close' column is missing in the CSV file. Please ensure it is included.")

    else:
        st.warning("Please upload a CSV file.")

if __name__ == "__main__":
    main()
