import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure the file is in the correct directory.")
        return None

def main():
    # Moved lifetimes import here to avoid environment conflicts
    #from lifetimes import BetaGeoFitter, GammaGammaFitter
    #from lifetimes.utils import summary_data_from_transaction_data

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "RFM Analysis", "CLTV Prediction", "Documentation"])

    st.sidebar.title("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
        df = load_data(uploaded_file)
else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()
    #else:
        #df = load_data("shopping_trends.csv")

if df is not None:
        # Data Cleaning and Preparation
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
        df['customer_id'] = df['customer_id'].astype(str)
        df['price'] = pd.to_numeric(df['price'])
        df['quantity'] = pd.to_numeric(df['quantity'])
        df['total_spend'] = df['quantity'] * df['price']

        # RFM Calculation
        max_date = df['invoice_date'].max()
        rfm_df = df.groupby('customer_id').agg({
            'invoice_date': lambda date: (max_date - date.max()).days,
            'invoice_no': 'count',
            'total_spend': 'sum'
        })
        rfm_df.rename(columns={
            'invoice_date': 'R',
            'invoice_no': 'F',
            'total_spend': 'M'
        }, inplace=True)

        rfm_df['r_score'] = pd.qcut(rfm_df['R'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
        rfm_df['f_score'] = pd.qcut(rfm_df['F'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm_df['m_score'] = pd.qcut(rfm_df['M'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm_df['rfm_score'] = rfm_df['r_score'].astype(str) + rfm_df['f_score'].astype(str) + rfm_df['m_score'].astype(str)

        segment_map = {
            r'[1-2][1-2]': 'Hibernating',
            r'[1-2][3-4]': 'At Risk',
            r'[1-2]5': 'Cannot Lose Them',
            r'3[1-2]': 'About to Sleep',
            r'33': 'Need Attention',
            r'[3-4][4-5]': 'Loyal Customers',
            r'41': 'Promising',
            r'51': 'New Customers',
            r'[4-5][2-3]': 'Potential Loyalists',
            r'5[4-5]': 'Champions'
        }
        rfm_df['segment'] = rfm_df['r_score'].astype(str) + rfm_df['f_score'].astype(str)
        rfm_df['segment'] = rfm_df['segment'].replace(segment_map, regex=True)

        # CLTV Calculation
        #summary = summary_data_from_transaction_data(df, 'customer_id', 'invoice_date', 'total_spend')
        #bgf = BetaGeoFitter(penalizer_coef=0.0)
        #bgf.fit(summary['frequency'], summary['recency'], summary['T'])

        # The GammaGammaFitter model can only be trained on customers with frequency > 0 and monetary_value > 0.
        ggf_df = summary[(summary['frequency'] > 0) & (summary['monetary_value'] > 0)]
        ggf = GammaGammaFitter(penalizer_coef=0.0)
        ggf.fit(ggf_df['frequency'], ggf_df['monetary_value'])

        summary['predicted_cltv'] = ggf.customer_lifetime_value(
            bgf,
            summary['frequency'],
            summary['recency'],
            summary['T'],
            summary['monetary_value'],
            time=12,  # 12 months
            freq='D', # Daily frequency
            discount_rate=0.01
        )
        rfm_cltv_df = rfm_df.merge(summary[['predicted_cltv']], left_index=True, right_index=True, how='left')
        rfm_cltv_df['predicted_cltv'].fillna(0, inplace=True)

        # --- Simple CLTV Calculation (RFM + Linear Regression) ---
st.subheader("Simple CLTV Prediction")

from sklearn.linear_model import LinearRegression

# Prepare data
rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
rfm_features = rfm_features.fillna(0)
target = rfm_df['Monetary']

# Fit simple model
model = LinearRegression()
model.fit(rfm_features, target)

# Predict monthly spend and compute 12-month CLTV
rfm_df['Predicted_Monthly_Spend'] = model.predict(rfm_features).clip(min=0)
rfm_df['CLTV_12mo'] = rfm_df['Predicted_Monthly_Spend'] * 12

# Display results
st.write(rfm_df[['Recency', 'Frequency', 'Monetary', 'CLTV_12mo']].head())

if page == "Home":
            st.title("Home")
            st.write("Welcome to the RFM Analysis and CLTV Prediction App.")
            st.subheader("1. Data Loading & Cleaning")
            st.write("**Dataset Preview:**")
            st.dataframe(df.head())
            st.write("**Cleaned Dataset Info:**")
            st.text(f"Number of rows: {df.shape[0]} | Number of columns: {df.shape[1]}")

elif page == "RFM Analysis":
            st.title("RFM Analysis")
            st.write("***RFM Summary Metrics:***")
            st.write(f"Average Recency: {rfm_cltv_df['R'].mean():.2f} days")
            st.write(f"Average Frequency: {rfm_cltv_df['F'].mean():.2f} purchases")
            st.write(f"Average Monetary: ${rfm_cltv_df['M'].mean():.2f}")

            st.write("***RFM Analysis Results:***")
            st.dataframe(rfm_cltv_df.head())

            st.write("***Customer Segmentation:***")
            segment_counts = rfm_cltv_df['segment'].value_counts()
            fig_segments = px.bar(
                x=segment_counts.index,
                y=segment_counts.values,
                labels={'x': 'Segment', 'y': 'Number of Customers'},
                title="Customer Segments"
            )
            st.plotly_chart(fig_segments)

elif page == "CLTV Prediction":
            st.title("CLTV Prediction")
            st.write("***Predicted CLTV (12 months):***")
            st.dataframe(rfm_cltv_df.head())

            st.write("***Distribution of Predicted CLTV:***")
            fig_cltv_dist = px.histogram(
                rfm_cltv_df,
                x="predicted_cltv",
                nbins=50,
                title="CLTV Distribution"
            )
            st.plotly_chart(fig_cltv_dist)

            st.write("***Top 10 Customers by Predicted CLTV:***")
            st.dataframe(
                rfm_cltv_df.sort_values(by="predicted_cltv", ascending=False).head(10)
            )

        # Download button (RFM + CLTV results)
            csv = rfm_cltv_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results (RFM + CLTV CSV)",
                data=csv,
                file_name="rfm_cltv_results.csv",
                mime="text/csv",
            )

elif page == "Documentation":
            st.title("Documentation")
            st.markdown("""
        ## RFM Analysis and CLTV Prediction

        This application performs Recency, Frequency, Monetary (RFM) analysis and predicts
        Customer Lifetime Value (CLTV) from a dataset of customer transactions. It helps
        businesses understand their customer segments and estimate future value.

        ### What the app does
        - Cleans your uploaded CSV
        - Computes RFM (Recency, Frequency, Monetary)
        - Scores and segments customers
        - Estimates a simple 12-month CLTV using Linear Regression on R, F, and M

        ### Expected columns
        - `customer_id`
        - `invoice_date`
        - `total_spend` **or** both `quantity` and `price` (to compute `total_spend`)

        ### How this can be used
        - **Segment Customers:** Identify different customer segments and understand their behavior.
        - **Targeted Marketing:** Create personalized campaigns for different segments.
        - **Customer Retention:** Focus on retaining high-value customers (e.g., “Champions”, “Loyal Customers”).
        - **Optimize Spend:** Allocate marketing resources more effectively based on predicted CLTV.
        """)

if __name__ == "__main__":
    main()
