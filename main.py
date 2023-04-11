import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import patches as mpatches

# ---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(
    page_title="Frontline Coding Test",
    page_icon="https://api.iconify.design/openmoji/hammer.svg?width=50",
    layout="wide",
)

# PAge Intro
st.write(
    """
# 📊 Frontline Coding Test
For this task, the uploaded data is analyzed and visualized as it was specified in the instructions of the test.

**Try uploading your own data in the sidebar!** 📤

Uploaded dataset overview:
-------
""".strip()
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


@st.cache(allow_output_mutation=True, show_spinner=False)
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
    df["Finish"] = pd.to_datetime(df["Finish"], format="%m/%d/%Y %H:%M")
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
    df["Remaining Work"] = (1 - df["% Work Complete"]) * df["Work"]
    remaining_work = df["Remaining Work"].sum() / df["Work"].sum() * 100
    c4.write("Remaining Work")
    c4.subheader(f"{remaining_work:.2f}%")


def plot_gantt_chart(df, st_container=st):
    """Plot a Gantt chart from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the following columns: 'ID', 'Start', 'Finish' and 'Predecessors'.
    """
    # compute critical path
    df = df[["ID", "Start", "Finish", "Predecessors"]].copy()
    df["Critical Path"] = False
    df.loc[0, "Critical Path"] = True
    for i in range(len(df)):
        if df.loc[i, "Critical Path"]:
            for j in df.loc[i, "Predecessors"]:
                df.loc[df["ID"] == j, "Critical Path"] = True
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
    return fig


def plot_resource_distribution(df, st_container=st):
    """
    Plots in the same axis the histograms of the number tasks per day among the different resources
    (overlapping the distributions with opacity) with total time range in the x-axis.
    """
    # create a new dataframe with the start and finish dates of each task
    df_dates = pd.DataFrame(columns=["Resource", "Date"])
    for i in range(len(df)):
        date_range = pd.date_range(df.loc[i, "Start"], df.loc[i, "Finish"], freq="D")
        # get the resource name
        resource = df.loc[i, "Resource Names"]
        # create a dataframe with the dates and the resource name
        for date in date_range:
            df_dates = pd.concat(
                [df_dates, pd.DataFrame({"Resource": [resource], "Date": [date]})]
            )
            df_dates = pd.concat(
                [df_dates, pd.DataFrame({"Resource": [resource], "Date": [date]})]
            )
    # compute the number of tasks per day for each resource
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
    colors = ["pink", "c", "b", "r", "gold", "gray", "g"]
    for i, resource in enumerate(df_dates["Resource"].unique()):
        df_resource = df_dates[df_dates["Resource"] == resource]
        col = colors[i] if len(df_dates["Resource"].unique()) == len(colors) else None
        ax.hist(
            df_resource["Date"],
            weights=df_resource["Tasks"],
            bins=len(df_resource) + 30,
            alpha=0.5,
            label=resource,
            color=col,
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

## Displays the dataset
default_file = "Frontline Test.csv"
uploaded_file = uploaded_file if uploaded_file is not None else default_file
df_raw, df = preprocess_data(uploaded_file)
st.write("Raw data:")
st.dataframe(df_raw, height=100)
st.write("Preprocessed data:")
st.dataframe(df, height=100)
st.header("Task Components:")
st.subheader("1. Key Project KPIs")
write_kpis(df)
st.subheader("2. Gantt Chart")
plot_gantt_chart(df)
st.subheader("3. Resource Distribution")
plot_resource_distribution(df)
