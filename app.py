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
    st.title("RFM + Simple CLTV Analyzer")

    # --- Sidebar navigation
    page = st.sidebar.radio("Go to:", ["Home", "RFM Analysis", "CLTV Prediction", "Documentation"])

    # --- Data upload (keep this EXACTLY as-is and inside main())
    st.sidebar.title("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV to continue.")
        st.stop()

    # Load the CSV now that we know it's present
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # (continue with cleaning, compute RFM, etc…)

    # ---- Basic cleaning & required columns
    # Expect columns: customer_id, invoice_date, and either total_spend or (quantity, price)
    if "total_spend" not in df.columns:
        if {"quantity", "price"}.issubset(df.columns):
            df["total_spend"] = pd.to_numeric(df["quantity"], errors="coerce") * pd.to_numeric(df["price"], errors="coerce")
        else:
            st.error("Your CSV must include 'total_spend' or both 'quantity' and 'price'.")
            st.stop()

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["customer_id"] = df["customer_id"].astype(str)
    df["total_spend"] = pd.to_numeric(df["total_spend"], errors="coerce")
    df = df.dropna(subset=["customer_id", "invoice_date", "total_spend"])

    # ---- Compute RFM
    snapshot_date = df["invoice_date"].max() + pd.Timedelta(days=1)
    rfm_df = (
        df.groupby("customer_id")
          .agg(
              Recency=("invoice_date", lambda x: (snapshot_date - x.max()).days),
              Frequency=("invoice_date", "count"),
              Monetary=("total_spend", "sum"),
          )
          .reset_index()
    )

    # ---- RFM scoring & segmentation
    # Rank then qcut to be robust to ties
    r = rfm_df["Recency"].rank(method="first", ascending=False)   # lower recency (more recent) is better
    f = rfm_df["Frequency"].rank(method="first", ascending=True)
    m = rfm_df["Monetary"].rank(method="first", ascending=True)

    rfm_df["R_Score"] = pd.qcut(r, q=5, labels=[5,4,3,2,1], duplicates="drop").astype(int)
    rfm_df["F_Score"] = pd.qcut(f, q=5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
    rfm_df["M_Score"] = pd.qcut(m, q=5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
    rfm_df["RFM_Sum"] = rfm_df[["R_Score","F_Score","M_Score"]].sum(axis=1)

    def segment(total):
        if total >= 13: return "Champions"
        if total >= 10: return "Loyal"
        if total >= 7:  return "Potential"
        if total >= 5:  return "At Risk"
        return "Hibernating"
    rfm_df["segment"] = rfm_df["RFM_Sum"].apply(segment)

    # ---- Simple CLTV (no lifetimes)
    from sklearn.linear_model import LinearRegression
    import numpy as np

    X = rfm_df[["Recency","Frequency","Monetary"]].fillna(0)
    y = rfm_df["Monetary"].fillna(0)

    lr = LinearRegression().fit(X, y)
    monthly = lr.predict(X)
    rfm_df["Predicted_Monthly_Spend"] = np.clip(monthly, 0, None)
    rfm_df["predicted_cltv"] = rfm_df["Predicted_Monthly_Spend"] * 12

    # Table used by your pages
    rfm_cltv_df = rfm_df.copy()
    
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
