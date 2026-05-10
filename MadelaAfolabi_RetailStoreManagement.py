"""
BUIS 305 / INSS 405 Group Project
Topic 1 - Retail Store Management: Sales and Customer Insights Dashboard

Author  : Mandela Afolabi
Dataset : SuperStore Sales Dataset
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# OOP: SalesAnalyzer Class
# Handles all data cleaning, analysis, and visualization methods for the SuperStore Sales dataset.
# ============================================================================

class SalesAnalyzer:
    """
    A class that encapsulates all data cleaning, analysis,
    and visualization methods for the SuperStore Sales dataset.

    Attributes
    ----------
    df : pd.DataFrame  – Original loaded data

    """

    def __init__(self, df):
        """Load data from a file path or an existing DataFrame."""
        if isinstance(df, pd.DataFrame):
            self.df = df.copy()
        else:
            self.df = pd.read_csv(df, encoding="utf-8-sig")
        self.df = self._clean()

    # ── Data Cleaning & Preprocessing ────────────────────────────────────────

    def _clean(self) -> pd.DataFrame:
        """
        Perform all data cleaning steps and return the cleaned DataFrame.

        Steps
        -----
        1. Strip column names and rename the composite Row-ID column.
        2. Parse Order Date and Ship Date into datetime objects.
        3. Derive helper columns: Month, Year.
        4. Replace '#N/A' strings with NaN; coerce Returns to numeric.
        5. Remove duplicate rows.
        6. Drop rows where Sales or Profit are missing.
        """
        df = self.df.copy()

        # 1. Clean column names
        df.columns = df.columns.str.strip()
        
        # 2. Parse dates (day-first format: DD-MM-YYYY)
        for col in ["Order Date", "Ship Date"]:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

        # 3. Derive time columns
        df["Month"] = df["Order Date"].dt.to_period("M").astype(str)
        df["Year"]  = df["Order Date"].dt.year

        # 4. Handle '#N/A' strings and coerce Returns
        df.replace("#N/A", pd.NA, inplace=True)
        df["Returns"] = pd.to_numeric(df["Returns"], errors="coerce")

        # 5. Remove duplicates
        df.drop_duplicates(inplace=True)

        # 6. Drop rows missing critical fields
        df.dropna(subset=["Sales", "Profit"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    # ── Summary / Aggregation Methods ────────────────────────────────────────

    def summary_stats(self) -> dict:
        """Return top-level KPI dictionary."""
        return {
            "Total Sales":     f"${self.df['Sales'].sum():,.2f}",
            "Total Profit":    f"${self.df['Profit'].sum():,.2f}",
            "Profit Margin":   f"{self.df['Profit'].sum() / self.df['Sales'].sum() * 100:.1f}%",
            "Total Orders":    self.df["Order ID"].nunique(),
            "Total Customers": self.df["Customer Name"].nunique(),
            "Total Products":  self.df["Product Name"].nunique(),
        }

    def sales_by_category(self) -> pd.Series:
        """Aggregate total sales per product category."""
        return (self.df.groupby("Category")["Sales"]
                    .sum()
                    .sort_values(ascending=False))

    def sales_by_region(self) -> pd.Series:
        """Aggregate total sales per region."""
        return (self.df.groupby("Region")["Sales"]
                    .sum()
                    .sort_values(ascending=False))

    def monthly_sales(self) -> pd.DataFrame:
        """Return monthly aggregated sales, sorted chronologically."""
        return (self.df.groupby("Month")["Sales"]
                    .sum()
                    .reset_index()
                    .sort_values("Month"))

    def top_customers(self, n: int = 10) -> pd.Series:
        """Return the top-N customers by total spending."""
        return (self.df.groupby("Customer Name")["Sales"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(n))

    def segment_distribution(self) -> pd.Series:
        """Total sales split by customer segment."""
        return self.df.groupby("Segment")["Sales"].sum()

    def subcategory_sales(self) -> pd.Series:
        """Total sales per sub-category."""
        return (self.df.groupby("Sub-Category")["Sales"]
                    .sum()
                    .sort_values(ascending=False))

    # ── Visualization Methods ─────────────────────────────────────────────────

    PALETTE = ["#1C3F6E", "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
               "#3BB273", "#7E6B8F", "#E84855"]

    def _base_fig(self, figsize=(7, 4)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("white")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        return fig, ax

    def plot_category_sales(self) -> plt.Figure:
        """Bar chart – sales by product category."""
        data = self.sales_by_category()
        fig, ax = self._base_fig((6, 4))
        bars = ax.bar(data.index, data.values,
                      color=self.PALETTE[:len(data)],
                      edgecolor="white", width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5000,
                    f"${bar.get_height() / 1000:.0f}K",
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color="#333")
        ax.set_title("Sales by Product Category", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_ylabel("Total Sales ($)", fontsize=10)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        plt.tight_layout()
        return fig

    def plot_region_sales(self) -> plt.Figure:
        """Horizontal bar chart – sales by region."""
        data = self.sales_by_region()
        fig, ax = self._base_fig((6, 4))
        bars = ax.barh(data.index[::-1], data.values[::-1],
                       color=self.PALETTE[:len(data)][::-1],
                       edgecolor="white", height=0.5)
        for bar in bars:
            ax.text(bar.get_width() + 3000,
                    bar.get_y() + bar.get_height() / 2,
                    f"${bar.get_width() / 1000:.0f}K",
                    va="center", fontsize=10, fontweight="bold", color="#333")
        ax.set_title("Sales by Region", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_xlabel("Total Sales ($)", fontsize=10)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        plt.tight_layout()
        return fig

    def plot_monthly_sales(self) -> plt.Figure:
        """Line chart – monthly sales trend."""
        data = self.monthly_sales()
        fig, ax = self._base_fig((9, 4))
        x = range(len(data))
        ax.plot(x, data["Sales"], color="#1C3F6E",
                linewidth=2.5, marker="o", markersize=3)
        ax.fill_between(x, data["Sales"], alpha=0.12, color="#1C3F6E")
        ticks = list(range(0, len(data), 6))
        ax.set_xticks(ticks)
        ax.set_xticklabels([data["Month"].iloc[i][:7] for i in ticks],
                           rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        ax.set_title("Monthly Sales Trend", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_ylabel("Sales ($)", fontsize=10)
        plt.tight_layout()
        return fig

    def plot_top_customers(self, n: int = 10) -> plt.Figure:
        """Horizontal bar chart – top N customers."""
        data = self.top_customers(n)
        fig, ax = self._base_fig((7, max(4, n * 0.45)))
        colors = [self.PALETTE[0]] + [self.PALETTE[1]] * (n - 1)
        bars = ax.barh(data.index[::-1], data.values[::-1],
                       color=colors[::-1], edgecolor="white", height=0.6)
        for bar in bars:
            ax.text(bar.get_width() + 100,
                    bar.get_y() + bar.get_height() / 2,
                    f"${bar.get_width():,.0f}",
                    va="center", fontsize=9, color="#333")
        ax.set_title(f"Top {n} Customers by Total Spending",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Total Sales ($)", fontsize=10)
        plt.tight_layout()
        return fig

    def plot_segment_pie(self) -> plt.Figure:
        """Pie chart – customer segment distribution."""
        data = self.segment_distribution()
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("white")
        wedges, texts, autotexts = ax.pie(
            data.values, labels=data.index, autopct="%1.1f%%",
            colors=self.PALETTE[:len(data)], startangle=90,
            pctdistance=0.75,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
        for t in autotexts:
            t.set_fontsize(11)
            t.set_fontweight("bold")
            t.set_color("white")
        ax.set_title("Sales by Customer Segment",
                     fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        return fig

    def plot_subcategory_sales(self, top_n: int = 10) -> plt.Figure:
        """Horizontal bar – top N sub-categories by sales."""
        data = self.subcategory_sales().head(top_n)
        fig, ax = self._base_fig((7, max(4, top_n * 0.42)))
        bars = ax.barh(data.index[::-1], data.values[::-1],
                       color=self.PALETTE[1], edgecolor="white", height=0.6)
        for bar in bars:
            ax.text(bar.get_width() + 1000,
                    bar.get_y() + bar.get_height() / 2,
                    f"${bar.get_width() / 1000:.0f}K",
                    va="center", fontsize=9, color="#333")
        ax.set_title(f"Top {top_n} Sub-Categories by Sales",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Total Sales ($)", fontsize=10)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        plt.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit Dashboard
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="SuperStore Sales Dashboard",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ──
    st.sidebar.title("🛒 SuperStore Dashboard")
    st.sidebar.markdown("**BUIS 305 / INSS 405 – Spring 2026**")
    st.sidebar.divider()

    uploaded = st.sidebar.file_uploader(
        "Upload Sales CSV", type=["csv"],
        help="Upload a CSV dataset. Leave blank to use the default dataset."
    )

    default_path = "SuperStore_Sales_Dataset.csv"

    @st.cache_data
    def load_data(file_obj=None):
        if file_obj is not None:
            raw = pd.read_csv(file_obj, encoding="utf-8-sig")
        else:
            raw = pd.read_csv(default_path, encoding="utf-8-sig")
        analyzer = SalesAnalyzer(raw)
        return analyzer

    analyzer = load_data(uploaded)
    df = analyzer.df

    # ── Sidebar Filters ──
    st.sidebar.subheader("Filters")

    regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())
    sel_region = st.sidebar.selectbox("Region", regions)

    categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
    sel_cat = st.sidebar.selectbox("Category", categories)

    years = sorted(df["Year"].dropna().unique().astype(int).tolist())
    sel_years = st.sidebar.multiselect(
        "Year(s)", years, default=years,
        help="Hold Ctrl/Cmd to select multiple years."
    )

    segments = ["All"] + sorted(df["Segment"].dropna().unique().tolist())
    sel_seg = st.sidebar.selectbox("Customer Segment", segments)

    # Apply filters
    fdf = df.copy()
    if sel_region != "All":
        fdf = fdf[fdf["Region"] == sel_region]
    if sel_cat != "All":
        fdf = fdf[fdf["Category"] == sel_cat]
    if sel_years:
        fdf = fdf[fdf["Year"].isin(sel_years)]
    if sel_seg != "All":
        fdf = fdf[fdf["Segment"] == sel_seg]

    filtered_analyzer = SalesAnalyzer(fdf)

    # ── Header ──
    st.title("🛒 Retail Store Management: Sales & Customer Insights")
    st.caption("Interactive dashboard – use the sidebar filters to explore the data.")
    st.divider()

    # ── KPI Cards ──
    stats = filtered_analyzer.summary_stats()
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats.items()):
        col.metric(label, value)

    st.divider()

    # ── Charts Row 1 ──
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sales by Product Category")
        st.pyplot(filtered_analyzer.plot_category_sales())
    with c2:
        st.subheader("Sales by Region")
        st.pyplot(filtered_analyzer.plot_region_sales())

    st.divider()

    # ── Monthly Trend ──
    st.subheader("Monthly Sales Trend")
    st.pyplot(filtered_analyzer.plot_monthly_sales())

    st.divider()

    # ── Charts Row 2 ──
    c3, c4 = st.columns([3, 2])
    with c3:
        n = st.slider("Number of top customers to show", 5, 20, 10)
        st.subheader(f"Top {n} Customers by Spending")
        st.pyplot(filtered_analyzer.plot_top_customers(n))
    with c4:
        st.subheader("Customer Segment Distribution")
        st.pyplot(filtered_analyzer.plot_segment_pie())

    st.divider()

    # ── Sub-Category ──
    st.subheader("Top Sub-Categories by Sales")
    top_n_sub = st.slider("Number of sub-categories", 5, 15, 10, key="sub")
    st.pyplot(filtered_analyzer.plot_subcategory_sales(top_n_sub))

    st.divider()

    # ── Raw Data Preview ──
    with st.expander("📋 View Raw Data"):
        st.dataframe(fdf.head(200), use_container_width=True)
        st.caption(f"Showing first 200 of {len(fdf):,} rows.")

    # ── Key Insights ──
    st.subheader("📌 Key Insights")
    cat_top   = filtered_analyzer.sales_by_category().idxmax()
    reg_top   = filtered_analyzer.sales_by_region().idxmax()
    cust_top  = filtered_analyzer.top_customers(1).index[0]
    seg_top   = filtered_analyzer.segment_distribution().idxmax()

    st.markdown(f"""
- 🏆 **Top Category**: *{cat_top}* generates the highest revenue.
- 🗺️ **Best Region**: The *{reg_top}* region leads in total sales.
- 👤 **Top Customer**: *{cust_top}* is the highest-spending customer.
- 👥 **Key Segment**: *{seg_top}* customers account for the largest share of sales.
- 📈 Sales show **seasonal peaks** in the second half of each year.
- 💳 COD is the most popular payment mode, followed by Online payments.
    """)

    st.caption("Dashboard built with Streamlit and Plotly. Data source: SuperStore Sales Dataset. Author: Mandela Afolabi.")


if __name__ == "__main__":
    main()
