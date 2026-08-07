"""
Project FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
==========================================================================
A production-ready Streamlit dashboard for demand forecasting and
inventory risk intelligence, built for the NorthBay Living engagement.

Run with:
    streamlit run streamlit_app.py

Required files in the same directory:
    - dashboard_data.csv     (Invoice, StockCode, Description, Quantity,
                               InvoiceDate, Price, Customer ID, Country)
    - prediction_results.csv (forecast / demand prediction output)

Author: Data Science Intern — Project FORESIGHT
==========================================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ==========================================================================
# 1. PAGE CONFIGURATION
# ==========================================================================
st.set_page_config(
    page_title="Project FORESIGHT | Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# 2. CUSTOM CSS — MODERN BLUE PROFESSIONAL THEME
# ==========================================================================
st.markdown(
    """
    <style>
        /* Overall app background */
        .stApp {
            background-color: #f5f7fa;
        }

        /* Header banner */
        .main-header {
            background: linear-gradient(90deg, #0f2c5c 0%, #1e5aa8 55%, #2f86d6 100%);
            padding: 2rem 2.2rem;
            border-radius: 16px;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px rgba(15, 44, 92, 0.25);
        }
        .main-header h1 {
            color: #ffffff;
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .main-header h3 {
            color: #dce8fb;
            font-weight: 400;
            margin-top: 0;
        }
        .main-header p {
            color: #b9d3f5;
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }

        /* KPI Cards */
        .kpi-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 1.2rem 1.1rem;
            box-shadow: 0 4px 14px rgba(15, 44, 92, 0.08);
            border-left: 5px solid #1e5aa8;
            text-align: left;
            height: 100%;
        }
        .kpi-card h4 {
            color: #5c6b7a;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 0.35rem;
        }
        .kpi-card h2 {
            color: #0f2c5c;
            font-size: 1.65rem;
            font-weight: 800;
            margin: 0;
        }

        /* Section headers */
        .section-header {
            background: #0f2c5c;
            color: white;
            padding: 0.6rem 1.1rem;
            border-radius: 10px;
            margin-top: 1.6rem;
            margin-bottom: 1rem;
            font-size: 1.25rem;
            font-weight: 700;
        }

        /* Risk badges */
        .risk-badge-stockout {
            background-color: #ffe1e1;
            color: #a30000;
            padding: 3px 10px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.8rem;
        }
        .risk-badge-overstock {
            background-color: #fff2d6;
            color: #a86400;
            padding: 3px 10px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.8rem;
        }
        .risk-badge-healthy {
            background-color: #dcf7e3;
            color: #157347;
            padding: 3px 10px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.8rem;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 1.4rem;
            margin-top: 2.5rem;
            border-top: 1px solid #d8dfe8;
            color: #5c6b7a;
            font-size: 0.85rem;
        }

        [data-testid="stMetricValue"] {
            color: #0f2c5c;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# 3. DATA LOADING (cached, safe, handles missing files/columns)
# ==========================================================================
DASHBOARD_FILE = "dashboard_data.csv"
PREDICTION_FILE = "prediction_results.csv"


@st.cache_data(show_spinner="Loading dashboard data...")
def load_dashboard_data(path: str) -> pd.DataFrame:
    """Load and clean the core sales dataset."""
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8", low_memory=False, on_bad_lines="skip")

    # Normalize column names (strip whitespace, keep original naming)
    df.columns = [c.strip() for c in df.columns]

    # Ensure expected columns exist; create if missing so app never crashes
    expected_cols = [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Type conversions
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=False)
    df["Customer ID"] = df["Customer ID"].astype(str)
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()

    # Drop rows with no usable quantity/price for revenue calc, but keep rest
    df = df.dropna(subset=["Quantity", "Price"], how="all")

    # Revenue = Quantity x Price
    df["Revenue"] = (df["Quantity"].fillna(0) * df["Price"].fillna(0)).round(2)

    return df


@st.cache_data(show_spinner="Loading prediction results...")
def load_prediction_data(path: str) -> pd.DataFrame:
    """Load the demand forecast / prediction results file."""
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8", low_memory=False, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    # Try to auto-detect a date-like column and parse it
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


dashboard_df = load_dashboard_data(DASHBOARD_FILE)
prediction_df = load_prediction_data(PREDICTION_FILE)

# ==========================================================================
# 4. HEADER
# ==========================================================================
st.markdown(
    """
    <div class="main-header">
        <h1>📦 Project FORESIGHT</h1>
        <h3>AI-Powered Demand &amp; Inventory Intelligence Platform</h3>
        <p>🚀 Turning transactional sales data into forecasts, stockout alerts,
        and inventory action plans — built for NorthBay Living.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if dashboard_df.empty:
    st.error(
        f"⚠️ Could not find **{DASHBOARD_FILE}** in the app directory. "
        "Please place the file alongside `streamlit_app.py` and reload."
    )
    st.stop()

# ==========================================================================
# 5. SIDEBAR — FILTERS
# ==========================================================================
st.sidebar.markdown("## 🔍 Filters")
st.sidebar.markdown("Use these filters to slice the dashboard.")

# --- Country filter ---
country_options = sorted(
    [c for c in dashboard_df["Country"].dropna().unique().tolist() if c and c != "nan"]
)
selected_countries = st.sidebar.multiselect(
    "🌍 Country", options=country_options, default=[]
)

# --- Product filter ---
product_options = sorted(
    [p for p in dashboard_df["Description"].dropna().unique().tolist() if p and p != "nan"]
)
selected_products = st.sidebar.multiselect(
    "🛒 Product", options=product_options, default=[]
)

# --- Date range filter ---
valid_dates = dashboard_df["InvoiceDate"].dropna()
if not valid_dates.empty:
    min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
else:
    date_range = None
    st.sidebar.info("No valid dates found in InvoiceDate column.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Inventory Risk Rules")
st.sidebar.markdown(
    """
    - 🔴 **Quantity < 5** → Stockout Risk → *Reorder Now*
    - 🟡 **Quantity > 100** → Overstock Risk → *Markdown / Reduce Purchase*
    - 🟢 **Otherwise** → Healthy → *Monitor*
    """
)

# --- Apply filters ---
filtered_df = dashboard_df.copy()

if selected_countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]

if selected_products:
    filtered_df = filtered_df[filtered_df["Description"].isin(selected_products)]

if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["InvoiceDate"].dt.date >= start_date)
        & (filtered_df["InvoiceDate"].dt.date <= end_date)
    ]

if filtered_df.empty:
    st.warning("No records match the selected filters. Showing full dataset instead.")
    filtered_df = dashboard_df.copy()

# ==========================================================================
# 6. KPI CARDS
# ==========================================================================
st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

total_revenue = filtered_df["Revenue"].sum()
total_orders = filtered_df["Invoice"].nunique()
total_customers = filtered_df["Customer ID"].nunique()
total_products = filtered_df["StockCode"].nunique()
avg_order_value = (total_revenue / total_orders) if total_orders else 0

k1, k2, k3, k4, k5 = st.columns(5)

kpi_data = [
    (k1, "💰 Total Revenue", f"₹{total_revenue:,.0f}"),
    (k2, "🧾 Total Orders", f"{total_orders:,}"),
    (k3, "👥 Total Customers", f"{total_customers:,}"),
    (k4, "📦 Total Products", f"{total_products:,}"),
    (k5, "🎯 Avg Order Value", f"₹{avg_order_value:,.2f}"),
]

for col, label, value in kpi_data:
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <h4>{label}</h4>
                <h2>{value}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================================================================
# 7. DATASET OVERVIEW
# ==========================================================================
st.markdown('<div class="section-header">🗂️ Dataset Overview</div>', unsafe_allow_html=True)

tab_preview, tab_info, tab_missing = st.tabs(["📋 Preview", "ℹ️ Information", "🧩 Missing Values"])

with tab_preview:
    st.dataframe(filtered_df.head(100), use_container_width=True)

with tab_info:
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown("**Shape**")
        st.write(f"{filtered_df.shape[0]:,} rows × {filtered_df.shape[1]:,} columns")
        st.markdown("**Column Data Types**")
        st.dataframe(filtered_df.dtypes.astype(str).rename("dtype"), use_container_width=True)
    with info_col2:
        st.markdown("**Numeric Summary**")
        numeric_cols = filtered_df.select_dtypes(include="number")
        if not numeric_cols.empty:
            st.dataframe(numeric_cols.describe().round(2), use_container_width=True)
        else:
            st.info("No numeric columns available for summary.")

with tab_missing:
    missing_summary = filtered_df.isna().sum().reset_index()
    missing_summary.columns = ["Column", "Missing Count"]
    missing_summary["Missing %"] = (
        (missing_summary["Missing Count"] / len(filtered_df) * 100).round(2)
        if len(filtered_df) > 0 else 0
    )
    missing_summary = missing_summary.sort_values("Missing Count", ascending=False)
    st.dataframe(missing_summary, use_container_width=True)
    if missing_summary["Missing Count"].sum() == 0:
        st.success("✅ No missing values detected in the current view.")

# ==========================================================================
# 8. SALES ANALYTICS
# ==========================================================================
st.markdown('<div class="section-header">📈 Sales Analytics</div>', unsafe_allow_html=True)

color_sequence = px.colors.sequential.Blues_r

# --- Monthly Revenue Trend ---
monthly_df = filtered_df.dropna(subset=["InvoiceDate"]).copy()
if not monthly_df.empty:
    monthly_df["Month"] = monthly_df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    monthly_revenue = monthly_df.groupby("Month", as_index=False)["Revenue"].sum()
    fig_monthly = px.line(
        monthly_revenue, x="Month", y="Revenue",
        markers=True, title="🗓️ Monthly Revenue Trend",
        color_discrete_sequence=["#1e5aa8"],
    )
    fig_monthly.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        title_font_size=18, xaxis_title="Month", yaxis_title="Revenue (₹)",
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
else:
    st.info("No valid InvoiceDate values available to plot the monthly revenue trend.")

col_a, col_b = st.columns(2)

with col_a:
    top_products = (
        filtered_df.groupby("Description", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )
    fig_products = px.bar(
        top_products.sort_values("Revenue"), x="Revenue", y="Description",
        orientation="h", title="🏆 Top 10 Products by Revenue",
        color="Revenue", color_continuous_scale="Blues",
    )
    fig_products.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=16)
    st.plotly_chart(fig_products, use_container_width=True)

with col_b:
    top_countries = (
        filtered_df.groupby("Country", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )
    fig_countries = px.bar(
        top_countries.sort_values("Revenue"), x="Revenue", y="Country",
        orientation="h", title="🌍 Top 10 Countries by Revenue",
        color="Revenue", color_continuous_scale="Teal",
    )
    fig_countries.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=16)
    st.plotly_chart(fig_countries, use_container_width=True)

# --- World Map: Revenue by Country ---
st.markdown("#### 🗺️ Revenue by Country - World Map")
country_map_df = (
    filtered_df.groupby("Country", as_index=False)["Revenue"].sum()
)
fig_map = px.choropleth(
    country_map_df,
    locations="Country",
    locationmode="country names",
    color="Revenue",
    hover_name="Country",
    color_continuous_scale="Blues",
    title="🌍 Global Revenue Distribution",
)
fig_map.update_layout(
    geo=dict(showframe=False, showcoastlines=True, bgcolor="rgba(0,0,0,0)"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    title_font_size=18,
)
st.plotly_chart(fig_map, use_container_width=True)

col_c, col_d = st.columns(2)

with col_c:
    top_customers = (
        filtered_df.groupby("Customer ID", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )
    fig_customers = px.bar(
        top_customers.sort_values("Revenue"), x="Revenue", y="Customer ID",
        orientation="h", title="👑 Top Customers by Revenue",
        color="Revenue", color_continuous_scale="Purples",
    )
    fig_customers.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", title_font_size=16,
        yaxis=dict(type="category"),
    )
    st.plotly_chart(fig_customers, use_container_width=True)

with col_d:
    fig_rev_dist = px.histogram(
        filtered_df[filtered_df["Revenue"] > 0], x="Revenue", nbins=40,
        title="💵 Revenue Distribution", color_discrete_sequence=["#2f86d6"],
    )
    fig_rev_dist.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=16)
    st.plotly_chart(fig_rev_dist, use_container_width=True)

fig_qty_dist = px.histogram(
    filtered_df.dropna(subset=["Quantity"]), x="Quantity", nbins=40,
    title="📦 Quantity Distribution", color_discrete_sequence=["#0f2c5c"],
)
fig_qty_dist.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=16)
st.plotly_chart(fig_qty_dist, use_container_width=True)

# ==========================================================================
# 9. DEMAND FORECAST
# ==========================================================================
st.markdown('<div class="section-header">🔮 Demand Forecast</div>', unsafe_allow_html=True)

if prediction_df.empty:
    st.warning(
        f"⚠️ Could not find **{PREDICTION_FILE}**. Place it alongside `streamlit_app.py` "
        "to enable the forecast section."
    )
else:
    st.markdown("#### 📋 Prediction Table")
    st.dataframe(prediction_df, use_container_width=True)

    pred_cols_lower = {c.lower(): c for c in prediction_df.columns}

    # Try to auto-detect a date column and a prediction/value column
    date_col = next((pred_cols_lower[k] for k in pred_cols_lower if "date" in k), None)
    predicted_col = next(
        (pred_cols_lower[k] for k in pred_cols_lower if "predict" in k or "forecast" in k),
        None,
    )
    actual_col = next((pred_cols_lower[k] for k in pred_cols_lower if "actual" in k), None)

    if date_col and predicted_col:
        st.markdown("#### 📈 Forecast Trend")
        plot_df = prediction_df.dropna(subset=[date_col]).sort_values(date_col)
        fig_forecast = px.line(
            plot_df, x=date_col, y=predicted_col, markers=True,
            title="Forecasted Demand Over Time", color_discrete_sequence=["#1e5aa8"],
        )
        fig_forecast.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=16)
        st.plotly_chart(fig_forecast, use_container_width=True)

        if actual_col:
            st.markdown("#### 🎯 Actual vs Predicted")
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=plot_df[date_col], y=plot_df[actual_col],
                mode="lines+markers", name="Actual", line=dict(color="#0f2c5c"),
            ))
            fig_avp.add_trace(go.Scatter(
                x=plot_df[date_col], y=plot_df[predicted_col],
                mode="lines+markers", name="Predicted", line=dict(color="#e07b00", dash="dash"),
            ))
            fig_avp.update_layout(
                title="Actual vs Predicted Demand", plot_bgcolor="white",
                paper_bgcolor="white", title_font_size=16,
            )
            st.plotly_chart(fig_avp, use_container_width=True)
        else:
            st.info("No 'Actual' column detected — showing forecast trend only.")
    elif predicted_col:
        st.markdown("#### 📈 Forecast Values")
        fig_forecast_simple = px.line(
            prediction_df.reset_index(), x="index", y=predicted_col,
            markers=True, title="Forecasted Demand", color_discrete_sequence=["#1e5aa8"],
        )
        fig_forecast_simple.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_forecast_simple, use_container_width=True)
    else:
        st.info(
            "Could not auto-detect date/prediction columns in the prediction file. "
            "The raw table above is still fully available."
        )

# ==========================================================================
# 10. INVENTORY INTELLIGENCE (Business Rules)
# ==========================================================================
st.markdown('<div class="section-header">🧠 Inventory Intelligence</div>', unsafe_allow_html=True)


def classify_risk(qty):
    """Apply business rules to classify inventory risk."""
    if pd.isna(qty):
        return "Unknown", "Review Data"
    if qty < 5:
        return "Stockout Risk", "Reorder Now"
    elif qty > 100:
        return "Overstock Risk", "Markdown / Reduce Purchase"
    else:
        return "Healthy", "Monitor"


# Aggregate quantity per product to evaluate stock-level risk
inventory_df = (
    filtered_df.groupby(["StockCode", "Description"], as_index=False)
    .agg(Total_Quantity=("Quantity", "sum"), Total_Revenue=("Revenue", "sum"))
)

inventory_df[["Risk Status", "Recommended Action"]] = inventory_df["Total_Quantity"].apply(
    lambda q: pd.Series(classify_risk(q))
)


def badge_html(status):
    mapping = {
        "Stockout Risk": "risk-badge-stockout",
        "Overstock Risk": "risk-badge-overstock",
        "Healthy": "risk-badge-healthy",
    }
    css_class = mapping.get(status, "risk-badge-healthy")
    return f'<span class="{css_class}">{status}</span>'


st.markdown("#### 📋 Inventory Risk Table")
display_inventory = inventory_df.copy()
display_inventory["Risk Status"] = display_inventory["Risk Status"].apply(badge_html)
st.write(
    display_inventory.rename(columns={"Total_Quantity": "Total Quantity", "Total_Revenue": "Total Revenue (₹)"})
    .to_html(escape=False, index=False),
    unsafe_allow_html=True,
)

col_e, col_f = st.columns([1, 1])

with col_e:
    st.markdown("#### 📊 Risk Distribution")
    risk_counts = inventory_df["Risk Status"].value_counts().reset_index()
    risk_counts.columns = ["Risk Status", "Count"]
    color_map = {
        "Stockout Risk": "#c93b3b",
        "Overstock Risk": "#e0a400",
        "Healthy": "#2f9e5c",
        "Unknown": "#9aa5b1",
    }
    fig_risk = px.pie(
        risk_counts, names="Risk Status", values="Count", hole=0.45,
        title="Inventory Risk Distribution",
        color="Risk Status", color_discrete_map=color_map,
    )
    fig_risk.update_layout(title_font_size=16)
    st.plotly_chart(fig_risk, use_container_width=True)

with col_f:
    st.markdown("#### ✅ Recommended Actions Summary")
    action_counts = inventory_df["Recommended Action"].value_counts().reset_index()
    action_counts.columns = ["Recommended Action", "Product Count"]
    fig_actions = px.bar(
        action_counts, x="Recommended Action", y="Product Count",
        title="Recommended Actions Across Products",
        color="Recommended Action",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_actions.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=16)
    st.plotly_chart(fig_actions, use_container_width=True)

# ==========================================================================
# 11. BUSINESS IMPACT KPIs
# ==========================================================================
st.markdown('<div class="section-header">💼 Business Impact</div>', unsafe_allow_html=True)

potential_revenue = inventory_df["Total_Revenue"].sum()
stockout_count = int((inventory_df["Risk Status"] == "Stockout Risk").sum())
overstock_count = int((inventory_df["Risk Status"] == "Overstock Risk").sum())
healthy_count = int((inventory_df["Risk Status"] == "Healthy").sum())
estimated_inventory_value = (
    filtered_df.assign(_val=filtered_df["Quantity"].fillna(0) * filtered_df["Price"].fillna(0))["_val"].sum()
)

b1, b2, b3, b4 = st.columns(4)
business_kpis = [
    (b1, "💰 Potential Revenue", f"₹{potential_revenue:,.0f}"),
    (b2, "📦 Estimated Inventory Value", f"₹{estimated_inventory_value:,.0f}"),
    (b3, "🔴 High-Risk Products", f"{stockout_count + overstock_count:,}"),
    (b4, "🟢 Healthy Products", f"{healthy_count:,}"),
]
for col, label, value in business_kpis:
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <h4>{label}</h4>
                <h2>{value}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

b5, b6 = st.columns(2)
with b5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <h4>🔻 Stockout Risk Count</h4>
            <h2>{stockout_count:,}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
with b6:
    st.markdown(
        f"""
        <div class="kpi-card">
            <h4>📈 Overstock Risk Count</h4>
            <h2>{overstock_count:,}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================================
# 12. FOOTER
# ==========================================================================
st.markdown(
    f"""
    <div class="footer">
        📦 <strong>Project FORESIGHT</strong> | AI-Powered Demand &amp; Inventory Intelligence Platform<br>
        Built with Streamlit, Pandas &amp; Plotly · Last refreshed: {datetime.now().strftime('%d %b %Y, %H:%M')}
    </div>
    """,
    unsafe_allow_html=True,
)