import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import patches as mpatches
import time

DEFAULT_FILE = "Frontline Test.csv"
PAGE_INTRO = """
# 📊 Frontline Coding Test
For this task, the uploaded data is analyzed and visualized as it was specified in the instructions of the test.

**Try uploading your own data in the sidebar!** 📤

Uploaded dataset overview:
-------
"""

# ---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(
    page_title="Frontline Coding Test",
    page_icon="https://api.iconify.design/openmoji/hammer.svg?width=50",
    layout="wide",
)


# Formatting ---------------------------------#


hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {	
            visibility: hidden;
        }
        footer:after {
            content:'Enrique Botía Barberá | 2023';
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
        }
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ---------------------------------#
# Data preprocessing and Model building


def preprocess_data(datafile_path):
    df_raw = pd.read_csv(datafile_path)
    df = df_raw.copy()
    # convert 'Task Name' to lower case
    df["Task Name"] = df["Task Name"].str.lower()
    # drop rows with missing values in 'Duration' column
    df = df.dropna(subset=["Duration"])
    # convert 'Duration' to int by removing the 'day(s)' string
    df["Duration"] = df["Duration"].str.rstrip("days").astype("int")
    # convert 'Start' and 'Finish' from 'm/dd/yyyy h:mm'format to datetime
    df["Start"] = pd.to_datetime(df["Start"], format="%m/%d/%Y %H:%M")
    # edit finish date to correspond with duration
    df["Finish"] = df.apply(
        lambda row: row["Start"] + pd.offsets.BDay(row["Duration"] - 1), axis=1
    )
    df["Finish"] += pd.offsets.Hour(9)
    # sort by start and finish date
    df = df.sort_values(by=["Start", "Finish"], ascending=False).reset_index(drop=True)
    # convert 'Predecessors' to list of integers
    df["Predecessors"] = df["Predecessors"].str.split(",").fillna("")
    df["Predecessors"] = df["Predecessors"].apply(
        lambda x: [int(i) if i != "" else [] for i in x]
    )
    # convert '% Work Complete' to float by removing the '%' string
    df["% Work Complete"] = (
        df["% Work Complete"].str.rstrip("%").astype("float") / 100.0
    )
    # convert 'Work' to float by removing the 'hrs' string
    df["Work"] = df["Work"].str.rstrip("hrs").astype("float")
    return df_raw, df


# Visualization --------------------------------------


def write_kpis(df):
    # compute the project duration
    c1, c2, c3, c4 = st.columns(4)
    project_duration = df["Finish"].max() - df["Start"].min()
    c1.write("Project Duration (days)")
    c1.subheader(project_duration.days)
    # compute the nuber of different resources
    n_resources = df["Resource Names"].nunique()
    c2.write("Resources")
    c2.subheader(n_resources)
    # compute the number of different tasks
    n_tasks = df["ID"].nunique()
    c3.write("Tasks")
    c3.subheader(n_tasks)
    # compute the Remaining Work (% of (remaining work hours)/(total project work hours))
    remaining_work = (1 - df["% Work Complete"]) * df["Work"]
    remaining_work_kpi = remaining_work.sum() / df["Work"].sum() * 100
    c4.write("Remaining Work")
    c4.subheader(f"{remaining_work_kpi:.2f}%")


def plot_gantt_chart(df, st_container=st):
    """Plot a Gantt chart from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the following columns: 'ID', 'Start', 'Finish' and 'Predecessors'.
    """
    start_t = time.time()
    # compute critical path
    df = df[["ID", "Start", "Finish", "Predecessors"]].copy()
    df["Critical Path"] = False
    # find the task with the latest finish date
    df.loc[df["Finish"].idxmax(), "Critical Path"] = True
    for i in range(len(df)):
        if df.loc[i, "Critical Path"]:
            for j in df.loc[i, "Predecessors"]:
                df.loc[df["ID"] == j, "Critical Path"] = True
    end = time.time()
    st.write(f"Critical path computation time: {end - start_t:.2f} seconds")
    start_t = time.time()
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))
    # Set axis limits
    ax.set_xlim(df["Start"].min(), df["Finish"].max())
    ax.set_ylim(0, len(df) + 1)
    # Format the y-axis
    ax.set_yticks(np.arange(len(df) + 1, 0, -10))
    ax.set_yticklabels(np.arange(0, len(df) - len(df) % 10 + 1, 10))
    # Format the x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B"))
    # Plot the grid
    ax.grid(axis="x", color="gray", linestyle="dotted", linewidth=1, alpha=0.7)
    ax.grid(axis="y", color="gray", linestyle="dotted", linewidth=1, alpha=0.7)
    # Plot the bars
    for task, start, finish in zip(df.index + 1, df["Start"], df["Finish"]):
        task_color = "r" if df.loc[task - 1, "Critical Path"] else "c"
        ax.plot([start, finish], [task, task], color=task_color, linewidth=2)
    # Create a legend
    red_patch = mpatches.Patch(color="red", label="Critical Path")
    plt.legend(handles=[red_patch])
    # Show the plot
    st_container.pyplot(fig)
    end = time.time()
    st.write(f"Plotting time: {end - start_t:.2f} seconds")
    return fig


def plot_resource_distribution(df, st_container=st):
    """
    Plots in the same axis the histograms of the number tasks per day among the different resources
    (overlapping the distributions with opacity) with total time range in the x-axis.
    """
    # create date ranges
    df_dates = df.apply(
        lambda row: pd.date_range(row["Start"], row["Finish"], freq="D")
        .map(lambda date: {"Resource": row["Resource Names"], "Date": date})
        .tolist(),
        axis=1,
    )
    # Concatenate all the lists of dictionaries and create a dataframe
    df_dates = pd.DataFrame([d for sublist in df_dates for d in sublist])
    # Compute the number of tasks per day for each resource
    df_dates = df_dates.groupby(["Resource", "Date"]).size().reset_index(name="Tasks")
    # plot the histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(df["Start"].min(), df["Finish"].max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B"))
    ax.set_yticks(np.arange(0, max(df_dates["Tasks"]) * 2, 2))
    ax.set_yticklabels(np.arange(0, max(df_dates["Tasks"]) * 2, 2))
    ax.grid(axis="x", color="gray", linestyle="dotted", linewidth=1, alpha=0.5)
    ax.grid(axis="y", color="gray", linestyle="dotted", linewidth=1, alpha=0.5)
    grouped_resources = df_dates.groupby("Resource")
    for resource, group in grouped_resources:
        ax.hist(
            group["Date"],
            weights=group["Tasks"],
            bins=len(group) + 30,
            alpha=0.5,
            label=resource,
        )
    ax.legend()
    st_container.pyplot(fig)
    return fig


# ---------------------------------#
# Sidebar

## Sidebar - Collects user input features into dataframe
with st.sidebar.header("UPLOAD DATA HERE"):
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])


# ---------------------------------#
# Main panel

st.write(PAGE_INTRO.strip())
uploaded_file = uploaded_file if uploaded_file is not None else DEFAULT_FILE
start = time.time()
df_raw, df = preprocess_data(uploaded_file)
end = time.time()
st.write("Raw data:")
st.dataframe(df_raw, height=100)
st.write("Preprocessed data:")
st.dataframe(df, height=100)
st.write(f"Data preprocessed in {end - start:.2f} seconds.")
st.header("Task Components:")
st.subheader("1. Key Project KPIs")
start = time.time()
write_kpis(df)
end = time.time()
st.write(f"KPIs computed in {end - start:.2f} seconds.")
st.subheader("2. Gantt Chart")
start = time.time()
plot_gantt_chart(df)
end = time.time()
st.write(f"Gantt chart computed in {end - start:.2f} seconds.")
st.subheader("3. Resource Distribution")
start = time.time()
plot_resource_distribution(df)
end = time.time()
st.write(f"Resource distribution computed in {end - start:.2f} seconds.")


# task_ids = df.index + 1
# starts = df["Start"]
# finishes = df["Finish"]

# # Create a 2D array where each row contains the start and finish times of a task
# task_times = np.column_stack((starts, finishes))

# # Plot all tasks at once
# ax.plot(task_times.T, np.vstack((task_ids, task_ids)).T, linewidth=2)
