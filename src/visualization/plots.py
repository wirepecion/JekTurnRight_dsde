"""
src/visualization/plots.py
--------------------------
Dedicated module for all plotting logic.
Refactored from 'insight.ipynb' to be modular and reusable.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
import folium
from folium.plugins import HeatMap, HeatMapWithTime
import ipywidgets as widgets
from ipywidgets import interact

# --- 1. REPORT DISTRIBUTION (Bar & Line) ---
def plot_monthly_daily_distribution(df: pd.DataFrame):
    """Plots static monthly and daily report counts."""
    # Ensure date column exists
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
    # Monthly
    monthly_counts = df.groupby(pd.to_datetime(df['date']).dt.month_name()).size()
    order = ['January', 'February', 'March', 'April', 'May', 'June', 
             'July', 'August', 'September', 'October', 'November', 'December']
    monthly_counts = monthly_counts.reindex(order)

    plt.figure(figsize=(10, 5))
    monthly_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Reports by Month')
    plt.ylabel('Number of Reports')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Daily
    daily_counts = df.set_index(pd.to_datetime(df['date'])).resample('D').size()
    plt.figure(figsize=(12, 5))
    daily_counts.plot(kind='line', color='teal')
    plt.title('Daily Trend of Reports')
    plt.ylabel('Number of Reports')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- 2. INTERACTIVE DISTRIBUTION ---
def plot_interactive_distribution(df: pd.DataFrame):
    """Interactive widget to filter reports by Year."""
    # Pre-processing
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    all_years = sorted(df['year'].unique())

    def update_charts(selected_years):
        filtered_df = df[df['year'].isin(selected_years)].copy()
        if filtered_df.empty:
            print("No data for selected years.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Monthly
        monthly_counts = filtered_df.groupby([filtered_df['date'].dt.month, filtered_df['year']]).size().unstack()
        monthly_counts.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title(f"Monthly Distribution for: {selected_years}")
        ax1.set_xlabel("Month")
        
        # Daily
        daily_counts = filtered_df.set_index('date').resample('D').size()
        daily_counts.plot(kind='line', ax=ax2, color='teal')
        ax2.set_title("Daily Trend")
        
        plt.tight_layout()
        plt.show()

    year_selector = widgets.SelectMultiple(
        options=all_years,
        value=list(all_years),
        description='Years:',
        disabled=False
    )
    interact(update_charts, selected_years=year_selector)

# --- 3. TAG TRENDS (Exploded) ---
def plot_tag_trends(df: pd.DataFrame):
    """Plots trends for specific issue tags over time."""
    # Requires 'type_list' column (list of strings)
    # Explode list to rows
    exploded = df.explode('type_list').copy()
    exploded['date'] = pd.to_datetime(exploded['date'])
    exploded['month'] = exploded['date'].dt.to_period('M').dt.start_time
    exploded['year'] = exploded['date'].dt.year

    all_years = sorted(exploded['year'].unique())
    all_tags = sorted(exploded['type_list'].unique())
    
    # Color map
    palette = sns.color_palette("tab20", len(all_tags))
    color_map = dict(zip(all_tags, palette))

    def _plot(years):
        if not years: return
        filtered = exploded[exploded['year'].isin(years)]
        if filtered.empty: return

        trend_data = filtered.groupby(['month', 'type_list']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        for tag in trend_data.columns:
            ax.plot(trend_data.index, trend_data[tag], marker='o', label=tag, color=color_map.get(tag, 'black'))
        
        ax.set_title(f"Issue Trends: {years}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    year_widget = widgets.SelectMultiple(options=all_years, value=[all_years[-1]], description='Years:')
    interact(_plot, years=year_widget)

# --- 4. GEOSPATIAL HEATMAP (Static) ---
def plot_static_heatmap(df: pd.DataFrame):
    """Standard Folium Heatmap."""
    # Needs latitude/longitude
    df = df.dropna(subset=['latitude', 'longitude'])
    
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    # If pre-aggregated (Spark), weight by count. If raw, weight is 1.
    if 'number_of_report_flood' in df.columns:
        data = df[['latitude', 'longitude', 'number_of_report_flood']].values.tolist()
    else:
        # Assign weight 1 for raw rows
        data = df[['latitude', 'longitude']].assign(weight=1).values.tolist()

    HeatMap(data, radius=15, max_zoom=13).add_to(m)
    return m

# --- 5. CHOROPLETH WITH BASEMAP (Contextily) ---
def plot_choropleth_basemap(df: pd.DataFrame, shape_gdf):
    """Plots aggregated data on a real map background."""
    # 1. Aggregate if raw
    if 'total_report' not in df.columns:
        summary = df.groupby(['district', 'subdistrict']).size().reset_index(name='total_report')
    else:
        # If using Spark output, it's already aggregated by date, sum it up
        summary = df.groupby(['district', 'subdistrict'])['total_report'].sum().reset_index()

    # 2. Merge
    map_df = shape_gdf.merge(summary, on=['district', 'subdistrict'], how='left')
    map_df['total_report'] = map_df['total_report'].fillna(0)

    # 3. Reproject to Web Mercator for Contextily
    if map_df.crs.to_string() != "EPSG:3857":
        map_df = map_df.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    map_df.plot(
        column='total_report',
        cmap='OrRd',
        linewidth=0.5,
        ax=ax,
        edgecolor='black',
        alpha=0.6,
        legend=True,
        legend_kwds={'label': "Total Reports", 'orientation': "horizontal", 'shrink': 0.7}
    )
    
    # Add Real Map Background
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    plt.title("Report Density (with Base Map)")
    plt.tight_layout()
    plt.show()

# --- 6. ANIMATED HEATMAP (Time Series) ---
def plot_animated_heatmap(df: pd.DataFrame):
    """Folium HeatMapWithTime."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.start_time
    
    # Aggregate by Month & Location (approx to 3 decimal places to group nearby points)
    df['lat_ round'] = df['latitude'].round(3)
    df['lon_round'] = df['longitude'].round(3)
    
    if 'number_of_report_flood' in df.columns:
        val_col = 'number_of_report_flood'
    else:
        val_col = 'count'
        df['count'] = 1

    monthly_data = df.groupby(['month', 'lat_ round', 'lon_round'])[val_col].sum().reset_index()
    
    time_index = sorted(monthly_data['month'].unique())
    data_by_month = []
    time_labels = []

    for m in time_index:
        batch = monthly_data[monthly_data['month'] == m]
        data_by_month.append(batch[['lat_ round', 'lon_round', val_col]].values.tolist())
        time_labels.append(m.strftime('%Y-%m'))

    m = folium.Map(location=[13.7563, 100.5018], zoom_start=10)
    HeatMapWithTime(data_by_month, index=time_labels, radius=20, auto_play=True).add_to(m)
    return m

# --- 7. RESOLUTION TIME ANALYSIS ---
def plot_resolution_time(df: pd.DataFrame):
    """Bar chart of time taken to resolve issues."""
    # Requires raw data with 'timestamp' and 'last_activity'
    if 'state' in df.columns:
        done_df = df[df['state'] == 'เสร็จสิ้น'].copy()
    else:
        done_df = df.copy()

    # Calc duration
    done_df['duration'] = (pd.to_datetime(done_df['last_activity']) - pd.to_datetime(done_df['timestamp'])).dt.days
    
    bins = [0, 7, 14, 30, 90, 365, float('inf')]
    labels = ['<1 Week', '1-2 Weeks', '2-4 Weeks', '1-3 Months', '3-12 Months', '>1 Year']
    
    done_df['group'] = pd.cut(done_df['duration'], bins=bins, labels=labels)
    
    counts = done_df['group'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    
    # Annotate
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
    plt.title("Resolution Time Distribution")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    plt.show()