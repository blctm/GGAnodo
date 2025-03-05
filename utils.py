import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import base64
import io
# Load dataset
@st.cache_data
def load_data():
    file_path = "data.xlsx"  # Ensure this path is correct
    xls = pd.ExcelFile(file_path)
    return xls.parse("Final")

data = load_data()

# @st.cache_data
# def load_data():
#     # Load Base64-encoded XLSX data from secrets
#     file_base64 = st.secrets["xls"]["file_base64"]
    
#     # Decode Base64 into a BytesIO stream (mimics a file)
#     file_bytes = base64.b64decode(file_base64)
#     file_stream = io.BytesIO(file_bytes)

#     # Load Excel file from memory
#     xls = pd.ExcelFile(file_stream)
    
#     # Read the specific sheet
#     return xls.parse("Final")

# # Load data
# data = load_data()


# Data Cleaning
#cleaned_data = data.drop(columns=[col for col in data.columns if "Comments" in col])
columns_to_drop = ["Si/C​\n(step #)", "CMC \n(step #)", "C65\n(step #)", "SBR\n(step #)", "Mixing speed  STEP 5 (rpm)",
"Mixing time  STEP 5 (minutes)", "Viscosity (filename)", "Scalable?"]
cleaned_data = data.drop(columns=[col for col in data.columns if "Comments" in col or col in columns_to_drop])

# Rename columns
rename_dict = {
    "Content of Si/C (%)": "Si/C (%)",               
    "Content of CMC (%)" : "CMC (%)" ,          
    "Content of C65 (%)" : "C65 (%)",             
    "Content of SBR (%)" : "SBR (%)",
    "Solid content (weight %)": "SC(weight)",
    "Mixing speed  STEP 1 (rpm)": "STEP 1 (rpm)",
    "Mixing speed  STEP 2 (rpm)": "STEP 2 (rpm)",
    "Mixing speed  STEP 3 (rpm)": "STEP 3 (rpm)",
    "Mixing speed  STEP 4 (rpm)": "STEP 4 (rpm)",
    "Mixing time  STEP 1 (minutes)": "STEP 1 (min)",
    "Mixing time  STEP 2 (minutes)": "STEP 2 (min)",
    "Mixing time  STEP 3 (minutes)": "STEP 3 (min)",
    "Mixing time  STEP 4 (minutes)": "STEP 4 (min)",
    "pH" : "pH",                             
    "pH Temperature (ºC)" : "pH (ºC)",             
    "Doctor blade gap (µm)":  "Doctor blade",       
    "Current Collector" : "Current Collector",    
    "Coating speed (mm/s)" : "Coating (mm/s)" ,          
    "Drying temperature (ºC)" : "Drying (ºC)" ,          
    "Drying time (min)" :  "Drying (min)",          
    "Vacuum temperature (ºC)" : "Vacuum (ºC)",    
    "Vacuum pressure (mbar)" : "Vacuum (mbar)",
    "Vacuum time (min)" :  "Vacuum (min)",    
    "Dry thickness (µm)" :  "Dry thickness",   
    "Active material loading (mg/cm2)" : "AML",
    "Good (Y) or trash (N) sample" : "Outcome",
}
cleaned_data.rename(columns=rename_dict, inplace=True)

#Current collector normalisation# 🔹 **Normalize Current Collector Categories**
def clean_current_collector(value):
    if "Pristine" in value:
        return "Pristine"
    return value  # Keep Cu1, Cu2, Cu3 unchanged

cleaned_data["Current Collector"] = cleaned_data["Current Collector"].apply(clean_current_collector)


# Re-identify categorical and numerical columns
categorical_cols = cleaned_data.select_dtypes(exclude=[float, int]).columns.tolist()
numerical_cols = cleaned_data.select_dtypes(include=[float, int]).columns.tolist()
#removing electrode id and date
categorical_cols = [col for col in categorical_cols if col not in ["Electrode id.", "Date"]]

# **Categorical Imputation**
# for col in categorical_cols:
#     if cleaned_data[col].nunique() <= 5:  # Low variability -> Mode imputation
#         cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
#     else:  # High variability -> Assign "Unknown"
#         cleaned_data[col].fillna("Unknown", inplace=True)
for col in categorical_cols:
    if cleaned_data[col].nunique() <= 5:  # Low variability -> Mode imputation
        cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])
    else:  # High variability -> Assign "Unknown"
        cleaned_data[col] = cleaned_data[col].fillna("Unknown")


# **Numerical Imputation** - Reapply KNN imputation with the full dataset
imputer = KNNImputer(n_neighbors=5)
cleaned_data[numerical_cols] = imputer.fit_transform(cleaned_data[numerical_cols])

# # Verify if the dropped columns are still present in cleaned_data
# remaining_columns = [col for col in cleaned_data.columns if "step" in col or "Comments" in col]
# remaining_columns


# Function to plot feature distributions
def plot_feature_distributions(df, feature):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[feature], kde=True, bins=20, color="#3498db", ax=ax)
    ax.set_title(f"Distribution of {feature}", fontsize=14)
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Boxplot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df[feature], color="#2ecc71", ax=ax)
    ax.set_title(f"Boxplot of {feature}", fontsize=14)
    st.pyplot(fig)

# Function to plot scatter plots vs Dry Thickness or Active Material Loading
def plot_scatter_vs_target(df, feature, target):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(x=df[feature], y=df[target], scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax)
    ax.set_title(f"{feature} vs {target}", fontsize=14)
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    st.pyplot(fig)

# Function to plot bar charts for categorical variables
def plot_categorical_bars(df, feature):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=df[feature], palette="muted", ax=ax)
    ax.set_title(f"Frequency of {feature}", fontsize=14)
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)
# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    df = df[numerical_cols]  # Ensure only numerical columns are used
    correlation_matrix = df.corr()

    # Filter to show only correlations with key targets
    target_features = ["Dry thickness", "AML"]
    correlation_subset = correlation_matrix[target_features].sort_values(by=target_features, ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size
    sns.heatmap(correlation_subset, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, ax=ax, cbar=True)

    # Improve label readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    ax.set_title("Feature Correlation Heatmap (Focused View)", fontsize=14)

    st.pyplot(fig)

# Function to plot full correlation heatmap
def plot_full_correlation_heatmap(df):
    df = df[numerical_cols]  # Ensure only numerical columns are used
    correlation_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))  # Bigger size
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                linewidths=0.5, ax=ax, cbar=True, annot_kws={"size": 8})  # Display values in cells

    # Improve label readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    ax.set_title("Full Feature Correlation Heatmap", fontsize=16)

    st.pyplot(fig)


