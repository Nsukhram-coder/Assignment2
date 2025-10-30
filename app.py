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
        df = load_data("shopping_trends.csv")

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
        summary = summary_data_from_transaction_data(df, 'customer_id', 'invoice_date', 'total_spend')
        bgf = BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(summary['frequency'], summary['recency'], summary['T'])

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

        if page == "Home":
            st.title("Home")
            st.write("Welcome to the RFM Analysis and CLTV Prediction App.")
            st.subheader("1. Data Loading & Cleaning")
            st.write("**Dataset Preview:**")
            st.dataframe(df.head())
            st.write("**Cleaned Dataset Info:**")
            st.text(f"Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}")

        elif page == "RFM Analysis":
            st.title("RFM Analysis")
            st.write("**RFM Summary Metrics:**")
            st.write(f"Average Recency: {rfm_cltv_df['R'].mean():.2f} days")
            st.write(f"Average Frequency: {rfm_cltv_df['F'].mean():.2f} purchases")
            st.write(f"Average Monetary: ${rfm_cltv_df['M'].mean():.2f}")

            st.write("**RFM Analysis Results:**")
            st.dataframe(rfm_cltv_df.head())

            st.write("**Customer Segmentation:**")
            segment_counts = rfm_cltv_df['segment'].value_counts()
            fig_segments = px.bar(segment_counts, x=segment_counts.index, y=segment_counts.values, labels={'x': 'Segment', 'y': 'Number of Customers'})
            st.plotly_chart(fig_segments)

        elif page == "CLTV Prediction":
            st.title("CLTV Prediction")
            st.write("**Predicted CLTV (12 months):**")
            st.dataframe(rfm_cltv_df.head())

            st.write("**Distribution of Predicted CLTV:**")
            fig_cltv_dist = px.histogram(rfm_cltv_df, x='predicted_cltv', nbins=50, title='CLTV Distribution')
            st.plotly_chart(fig_cltv_dist)

            st.write("**Top 10 Customers by Predicted CLTV:**")
            st.dataframe(rfm_cltv_df.sort_values(by='predicted_cltv', ascending=False).head(10))

        elif page == "Documentation":
            st.title("Documentation")
            st.markdown("""
            # RFM Analysis and CLTV Prediction

            This application performs Recency, Frequency, Monetary (RFM) analysis and predicts Customer Lifetime Value (CLTV) from a given dataset of customer transactions. It is designed to help businesses understand their customer segments and make data-driven decisions.

            ## What are RFM and CLTV?

            ### RFM Analysis
            RFM stands for:
            - **Recency**: How recently a customer has made a purchase.
            - **Frequency**: How often a customer makes a purchase.
            - **Monetary**: How much money a customer spends on purchases.

            RFM analysis is a marketing technique used to quantitatively rank and group customers based on their transaction history. It helps businesses identify their best customers and segment them into different groups for targeted marketing campaigns.

            ### CLTV Prediction
            Customer Lifetime Value (CLTV) is a prediction of the net profit attributed to the entire future relationship with a customer. It helps businesses understand the long-term value of their customers and make strategic decisions about customer acquisition and retention.

            ## How are they computed in this app?

            ### RFM Calculation
            1.  **Recency**: Calculated as the number of days between the last purchase date of a customer and the latest transaction date in the dataset.
            2.  **Frequency**: Calculated as the total number of purchases made by a customer.
            3.  **Monetary**: Calculated as the total amount of money spent by a customer.

            Customers are then scored on a scale of 1 to 5 for each RFM metric, and a combined RFM score is created. Based on these scores, customers are segmented into categories like "Champions," "Loyal Customers," and "At Risk."

            ### CLTV Prediction
            The app uses the **BG/NBD (Beta-Geometric/Negative Binomial Distribution)** and **Gamma-Gamma** models from the `lifetimes` library to predict CLTV.
            - The **BG/NBD model** predicts the number of future transactions a customer is likely to make.
            - The **Gamma-Gamma model** predicts the average monetary value of those future transactions.

            By combining these two models, the app estimates the total lifetime value of each customer over a 12-month period.

            ## How can this app be used in a business scenario?

            This application can be used by businesses to:
            - **Segment Customers**: Identify different customer segments and understand their behavior.
            - **Targeted Marketing**: Create personalized marketing campaigns for different customer segments. For example, a business could offer a special discount to "At Risk" customers to encourage them to make a purchase.
            - **Customer Retention**: Focus on retaining high-value customers, such as "Champions" and "Loyal Customers."
            - **Optimize Marketing Spend**: Allocate marketing resources more effectively by focusing on acquiring and retaining customers with the highest predicted CLTV.
            """)

        # Download Button
        csv = rfm_cltv_df.to_csv().encode('utf-8')
        st.sidebar.download_button(
            label="Download Results",
            data=csv,
            file_name='rfm_cltv_results.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
