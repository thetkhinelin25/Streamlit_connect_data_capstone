import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client, Client
from statsforecast import StatsForecast
from statsforecast.models import CrostonOptimized


# Initialize connection to Supabase
@st.cache_resource
def init_connection():
    url: str = st.secrets['supabase_url']
    key: str = st.secrets['supabase_key']
    client: Client = create_client(url, key)
    return client


supabase = init_connection()


# Query the database
@st.cache_data(ttl=600)
def run_query():
    result = supabase.table('car_parts_monthly_sales').select("*").execute()
    return result.data


# Load data into a DataFrame
@st.cache_data(ttl=600)
def create_dataframe():
    rows = run_query()
    df = pd.json_normalize(rows)
    df['volume'] = df['volume'].astype(int)
    df['date'] = pd.to_datetime(df['date'])
    return df


# Plot historical volumes
@st.cache_data
def plot_volume(df, ids):
    fig, ax = plt.subplots()
    for id in ids:
        part_df = df[df['parts_id'] == id]
        ax.plot(part_df['date'], part_df['volume'], label=f'Part {id}')
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)


# Prepare the dataset for forecasting (with monthly gaps filled)
@st.cache_data
def format_dataset(df, ids):
    model_df = df[df['parts_id'].isin(ids)].copy()
    model_df = model_df[['parts_id', 'date', 'volume']]
    model_df.rename(columns={
        'parts_id': 'unique_id',
        'date': 'ds',
        'volume': 'y'
    }, inplace=True)

    model_df['ds'] = pd.to_datetime(model_df['ds'])
    all_dfs = []

    for uid, group in model_df.groupby('unique_id'):
        idx = pd.date_range(group['ds'].min(), group['ds'].max(), freq='MS')
        full = group.set_index('ds').reindex(idx).fillna(0.0).rename_axis('ds').reset_index()
        full['unique_id'] = uid
        full['y'] = full['y'].astype(float)
        all_dfs.append(full)

    full_df = pd.concat(all_dfs)
    return full_df


# Initialize the Croston model
@st.cache_resource
def create_sf_object():
    models = [CrostonOptimized()]
    sf = StatsForecast(models=models, freq='MS', n_jobs=-1)
    return sf


# Make and export predictions
@st.cache_data(show_spinner="Making predictions...")
def make_predictions(df, ids, horizon):
    model_df = format_dataset(df, ids)
    sf = create_sf_object()
    forecast_df = sf.forecast(df=model_df, h=horizon)
    return forecast_df.to_csv(index=False)


# Streamlit UI
if __name__ == "__main__":
    st.title("📦 Forecast Product Demand (CrostonOptimized)")

    df = create_dataframe()

    st.subheader("Select a product")
    product_ids = st.multiselect(
        "Select product ID", options=df['parts_id'].unique())

    if product_ids:
        plot_volume(df, product_ids)

        with st.expander("📈 Forecast"):
            horizon = st.slider("Forecast Horizon (months)", 1, 12, step=1)
            forecast_btn = st.button("Forecast", type="primary")

            if forecast_btn:
                csv_file = make_predictions(df, product_ids, horizon)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv_file,
                    file_name="forecast_croston.csv",
                    mime="text/csv"
                )
    else:
        st.info("Please select at least one product ID to view data and forecast.")
