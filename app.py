import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import base64
import io
from scipy import stats
from utils import (
    cleaned_data, numerical_cols, categorical_cols,
    plot_feature_distributions, plot_categorical_bars,
    plot_scatter_vs_target, plot_correlation_heatmap, plot_full_correlation_heatmap
)
# 🔹 Custom CSS for Background & Sidebar
st.markdown(
    """
   <style>
        /* Change background color */
        [data-testid="stAppViewContainer"] {
            background-color: #D2E5E3;
        }

        /* Change sidebar background */
        [data-testid="stSidebar"] {
            background-color: #2E5B5887;
        }

        /* Change text color */
        h1, h2, h3, h4 {
            color: #333333; 
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔹 Sidebar Navigation
st.sidebar.image("logo.png", width=150)
st.sidebar.title("📖 Anode Analysis Wiki")
menu = st.sidebar.radio(
    "Select a section:",
    ["🏠 Home", "🔍Overview", "🔢Statistical Analyses","📖Understanding the Analysis","❓Hypothesis Testing", "📖Understanding Hypothesis", "📊Correlation Matrix"]
)

# # Function to load dataset
@st.cache_data
def load_data():
    return pd.read_excel("data.xlsx", sheet_name="Final")


data = load_data()

# @st.cache_data
# def load_data():
#     # Load Base64-encoded XLSX data from secrets
#     file_base64 = st.secrets["xls"]["file_base64"]  # Use correct key
    
#     # Decode Base64 into a BytesIO stream (mimics a file)
#     file_bytes = base64.b64decode(file_base64)
#     file_stream = io.BytesIO(file_bytes)

#     # Load Excel file from memory
#     return pd.read_excel(file_stream, sheet_name="Final")

# # Load data
# data = load_data()

# 🔹 Home Page
if menu == "🏠 Home":
    st.title("📖 Project Analysis Wiki")
    st.write(
        """
        Welcome to the project wiki! 📚  
        Use the sidebar to navigate through different insights.
        """
    )

# 🔹 Overview Page
elif menu == "🔍Overview":
    st.title("🔍 Overview")
    
    st.subheader("📌 Context")
    st.write("""
    The production of **high-quality anode electrodes** is crucial for ensuring the efficiency and longevity of electrochemical devices such as **batteries and fuel cells**. This study aims to analyze the **key parameters** influencing anode electrode quality to improve process control and optimization.
    """)

    st.subheader("🎯 Purpose of the Study")
    st.write("""
    The primary goal of this study is to **identify and understand the factors** that contribute to high-quality anode electrodes. Specifically, we aim to:
    - Determine the most **influential process parameters**.
    - Establish **correlations** between variables.
    - Develop **predictive models** to improve quality control.
    """)

    st.subheader("🔬 Process Stages")
    st.write("The anode electrode production process consists of four main stages:")

    # Creating a simple table for process stages
    process_stages = pd.DataFrame({
        "Stage": ["Slurry Preparation", "Mixing", "Coating", "Drying"],
        "Description": [
            "Raw materials are combined to create a homogeneous slurry.",
            "Slurry is mixed to achieve the required consistency and dispersion.",
            "The mixed slurry is applied to a substrate to form a uniform layer.",
            "The coated material is dried to remove solvents and stabilize the structure."            
        ],
        "Features": [
            "SC(weight)", 
            "Mixing speeds & times (STEP 1-4)", 
            "Doctor blade (µm), Coating speed (mm/s)", 
            "Drying temperature (ºC), Drying time (min), Vacuum temperature (ºC), Vacuum pressure (mbar), Vacuum time (min)"
        ]
    })
    # Displaying the table
    st.table(process_stages)

    st.subheader("⚠️ Challenges Faced")
    st.write("""
    Despite following a structured process, several challenges impact the ability to predict and control quality outcomes:
    
    - **📉 Limited Data:** The dataset contains only **32 data points**, limiting predictive modeling capabilities.
    - **❓ No Intermediate Quality Checks:** Quality is only assessed at the **end of the process**, making early defect detection difficult.
    - **🔄 Complex Interdependencies:** The relationships between process variables are highly **non-linear**, making clear cause-effect analysis challenging.
    - **🎯 Need for More Granular Data:** A finer breakdown of process measurements would enhance predictive accuracy.
    
    Addressing these challenges is essential for **optimizing quality control** and improving **anode electrode production**.
    """)



# 🔹 Statistical Analysis Page
elif menu == "🔢Statistical Analyses":
    st.title("🔢Statistical Analyses")

    st.subheader("📊 Feature Distributions")
    selected_feature = st.selectbox("Select a Feature to Visualize", numerical_cols + categorical_cols)

    if selected_feature in numerical_cols:
        plot_feature_distributions(cleaned_data, selected_feature)
    elif selected_feature in categorical_cols:
        plot_categorical_bars(cleaned_data, selected_feature)

    with st.expander("ℹ️ What does this graph tell us?"):
        st.markdown("""
        - **Histograms & KDE** show the **distribution** of numerical values.
        - **Box plots** help **detect outliers and variations** in the data.
        - **Bar charts** display **categorical variable frequency distributions**.
        """)

    st.subheader("📉 How Features Affect Dry Thickness or Active Material Loading")
    target_variable = st.selectbox("Select Target Variable", ["Dry thickness", "AML"]) #Active material loading (mg/cm2)
    feature_vs_target = st.selectbox("Select Feature to Compare", numerical_cols)

    if feature_vs_target:
        plot_scatter_vs_target(cleaned_data, feature_vs_target, target_variable)

    with st.expander("ℹ️ What does this scatter plot tell us?"):
        st.markdown("""
        - **Scatter plots** reveal relationships between a feature and a target variable.
        - **Regression lines** (in red) indicate **positive or negative correlations**.
        - The **closer the points are to the line**, the stronger the correlation.

                """)

# 📊 **Hypothesis Testing & Regression Analysis Section**
elif menu == "❓Hypothesis Testing":
    st.title("❓ Hypothesis Testing & Regression Analysis")
    
    # 📌 **Introduction**
    st.markdown(
        """
        Statistical hypothesis testing helps determine whether **observed patterns** in the data are meaningful
        or simply due to **random chance**. We use different techniques based on the type of data:
        - **t-Tests & ANOVA** for comparing means.
        - **Chi-Square Tests** for categorical relationships.
        - **Regression Analysis** for predicting relationships between variables.
        """
    )

    # 📊 **Table: Statistical Analysis Techniques**
    st.subheader("📋 Summary of Statistical Techniques")
    methods_df = pd.DataFrame({
        "Analysis Technique": [
            "Descriptive Statistics", "Histograms/Box Plots", "t-Test / ANOVA", "Chi-Square Test",
            "Correlation Analysis", "Regression Models"
        ],
        "Purpose": [
            "Summarize central tendency and dispersion",
            "Visualize distributions and outliers",
            "Compare group means",
            "Compare categorical distributions",
            "Identify linear relationships",
            "Quantify and predict relationships"
        ],
        "Example Application": [
            "Mean, median, std. of viscosity, mixing speed",
            "Distribution of dry thickness across batches",
            "Mixing type differences in dry thickness",
            "Good vs. Trash samples across mixing types",
            "Solid content vs. active material loading",
            "Predictive model for sample quality"
        ]
    })
    st.table(methods_df)

    # 📌 **T-Tests & ANOVA**
    st.subheader("🔬 Group Comparison: t-Test & ANOVA")

    # Select numerical & categorical variables
    target_num_var = st.selectbox("Select a Numerical Variable", numerical_cols)
    group_cat_var = st.selectbox("Select a Categorical Grouping Variable", categorical_cols)

    if target_num_var and group_cat_var:
        unique_groups = cleaned_data[group_cat_var].nunique()

        # ✅ Function to sanitize column names for Statsmodels
        def sanitize_column_name(name):
            """Replaces invalid characters to ensure compatibility with Statsmodels."""
            return (
                name.replace(" ", "_")   # Replace spaces with underscores
                    .replace("%", "pct") # Replace "%" with "pct"
                    .replace("/", "_")   # Replace "/" with "_"
                    .replace("(", "")    # Remove "("
                    .replace(")", "")    # Remove ")"
                    .replace("-", "_")   # Replace "-" with "_"
                    .replace(".", "_")   # Replace "." with "_"
                    .replace("º", "o")   # Replace "º" with "o" (degrees symbol issue)
            )

        # ✅ Apply sanitization to selected column names
        safe_target_var = sanitize_column_name(target_num_var)
        safe_group_var = sanitize_column_name(group_cat_var)

        # ✅ Rename dataset to match new sanitized column names
        cleaned_data_renamed = cleaned_data.rename(columns={target_num_var: safe_target_var, group_cat_var: safe_group_var})
        cleaned_data_renamed[safe_group_var] = cleaned_data_renamed[safe_group_var].astype(str) #fixing code
        filtered_data = cleaned_data_renamed.dropna(subset=[safe_target_var, safe_group_var])
        category_counts = filtered_data[safe_group_var].value_counts()
        valid_categories = category_counts[category_counts > 1].index
        filtered_data = filtered_data[filtered_data[safe_group_var].isin(valid_categories)]
        
        if unique_groups == 2:
            # Perform t-test
            group1 = cleaned_data_renamed[cleaned_data_renamed[safe_group_var] == cleaned_data_renamed[safe_group_var].unique()[0]][safe_target_var]
            group2 = cleaned_data_renamed[cleaned_data_renamed[safe_group_var] == cleaned_data_renamed[safe_group_var].unique()[1]][safe_target_var]
            t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
            st.write(f"**T-Test Results:** t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        # else:
        #     # ✅ Fixed OLS Formula: Use cleaned variable names
        #     formula = f"{safe_target_var} ~ C({safe_group_var})"
        #     anova_result = smf.ols(formula, data=cleaned_data_renamed).fit()
        #     anova_table = sm.stats.anova_lm(anova_result, typ=2)
        #     st.write("**ANOVA Results:**")
        #     st.write(anova_table)

        if filtered_data[safe_group_var].nunique() > 1:
            formula = f"{safe_target_var} ~ C({safe_group_var})"
            anova_result = smf.ols(formula, data=filtered_data).fit()
            anova_table = sm.stats.anova_lm(anova_result, typ=2)
            st.write("**ANOVA Results:**")
            st.write(anova_table)
        else:
            st.warning(f"Not enough valid groups in {group_cat_var} to perform ANOVA.")


        # 📌 Boxplot Visualization
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(x=cleaned_data_renamed[safe_group_var], y=cleaned_data_renamed[safe_target_var], palette="coolwarm", ax=ax)
        ax.set_title(f"{target_num_var} Distribution by {group_cat_var}")
        st.pyplot(fig)

    # 📌 **Chi-Square Test**
    st.subheader("📊 Chi-Square Test for Categorical Data")

    cat_var1 = st.selectbox("Select First Categorical Variable", categorical_cols)
    cat_var2 = st.selectbox("Select Second Categorical Variable", categorical_cols)

    if cat_var1 and cat_var2:
        contingency_table = pd.crosstab(cleaned_data[cat_var1], cleaned_data[cat_var2])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        st.write(f"**Chi-Square Test Results:** Chi2 = {chi2_stat:.4f}, p-value = {p_value:.4f}")

        # 📌 Bar Chart Visualization
        fig, ax = plt.subplots(figsize=(7, 5))
        contingency_table.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax)
        ax.set_title(f"{cat_var1} vs. {cat_var2}")
        st.pyplot(fig)

    # 📌 **Regression Analysis**
    st.subheader("📈 Regression Analysis")

    target_reg_var = st.selectbox("Select a Target Variable (Numerical)", ["Dry thickness", "AML"])
    predictor_vars = st.multiselect("Select Predictor Variables", numerical_cols)

    if target_reg_var and predictor_vars:
        # ✅ Fix Column Formatting for Regression
        safe_target_reg_var = sanitize_column_name(target_reg_var)
        safe_predictors = [sanitize_column_name(var) for var in predictor_vars]

        cleaned_data_renamed = cleaned_data.rename(columns={target_reg_var: safe_target_reg_var, **{var: safe_predictors[i] for i, var in enumerate(predictor_vars)}})

        formula = f"{safe_target_reg_var} ~ {' + '.join(safe_predictors)}"
        model = smf.ols(formula, data=cleaned_data_renamed).fit()

        st.write(model.summary())

        # 📌 Scatter Plot for Regression
        if len(predictor_vars) == 1:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.regplot(x=cleaned_data[predictor_vars[0]], y=cleaned_data[target_reg_var], scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}, ax=ax)
            ax.set_title(f"Regression: {predictor_vars[0]} vs. {target_reg_var}")
            st.pyplot(fig)
# 📊 **Hypothesis Testing & Regression Analysis Section**
elif menu == "❓Hypothesis Testing":
    st.title("❓ Hypothesis Testing & Regression Analysis")
    
    # 📌 **Introduction**
    st.markdown(
        """
        Statistical hypothesis testing helps determine whether **observed patterns** in the data are meaningful
        or simply due to **random chance**. We use different techniques based on the type of data:
        - **t-Tests & ANOVA** for comparing means.
        - **Chi-Square Tests** for categorical relationships.
        - **Regression Analysis** for predicting relationships between variables.
        """
    )

    # 📊 **Table: Statistical Analysis Techniques**
    st.subheader("📋 Summary of Statistical Techniques")
    methods_df = pd.DataFrame({
        "Analysis Technique": [
            "Descriptive Statistics", "Histograms/Box Plots", "t-Test / ANOVA", "Chi-Square Test",
            "Correlation Analysis", "Regression Models"
        ],
        "Purpose": [
            "Summarize central tendency and dispersion",
            "Visualize distributions and outliers",
            "Compare group means",
            "Compare categorical distributions",
            "Identify linear relationships",
            "Quantify and predict relationships"
        ],
        "Example Application": [
            "Mean, median, std. of viscosity, mixing speed",
            "Distribution of dry thickness across batches",
            "Mixing type differences in dry thickness",
            "Good vs. Trash samples across mixing types",
            "Solid content vs. active material loading",
            "Predictive model for sample quality"
        ]
    })
    st.table(methods_df)

    # 📌 **T-Tests & ANOVA**
    st.subheader("🔬 Group Comparison: t-Test & ANOVA")

    # Select numerical & categorical variables
    target_num_var = st.selectbox("Select a Numerical Variable", numerical_cols)
    group_cat_var = st.selectbox("Select a Categorical Grouping Variable", categorical_cols)

    if target_num_var and group_cat_var:
        unique_groups = cleaned_data[group_cat_var].nunique()

        # ✅ Function to sanitize column names for Statsmodels
        def sanitize_column_name(name):
            """Replaces invalid characters to ensure compatibility with Statsmodels."""
            return (
                name.replace(" ", "_")   # Replace spaces with underscores
                    .replace("%", "pct") # Replace "%" with "pct"
                    .replace("/", "_")   # Replace "/" with "_"
                    .replace("(", "")    # Remove "("
                    .replace(")", "")    # Remove ")"
                    .replace("-", "_")   # Replace "-" with "_"
                    .replace(".", "_")   # Replace "." with "_"
            )

        # ✅ Apply sanitization to selected column names
        safe_target_var = sanitize_column_name(target_num_var)
        safe_group_var = sanitize_column_name(group_cat_var)

        # ✅ Rename dataset to match new sanitized column names
        cleaned_data_renamed = cleaned_data.rename(columns={target_num_var: safe_target_var, group_cat_var: safe_group_var})

        if unique_groups == 2:
            # Perform t-test
            group1 = cleaned_data_renamed[cleaned_data_renamed[safe_group_var] == cleaned_data_renamed[safe_group_var].unique()[0]][safe_target_var]
            group2 = cleaned_data_renamed[cleaned_data_renamed[safe_group_var] == cleaned_data_renamed[safe_group_var].unique()[1]][safe_target_var]
            t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
            st.write(f"**T-Test Results:** t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        else:
            # ✅ Fixed OLS Formula: Use cleaned variable names
            formula = f"{safe_target_var} ~ C({safe_group_var})"
            anova_result = smf.ols(formula, data=cleaned_data_renamed).fit()
            anova_table = sm.stats.anova_lm(anova_result, typ=2)
            st.write("**ANOVA Results:**")
            st.write(anova_table)

        with st.expander("ℹ️ How to Interpret These Results"):
            st.markdown("""
            - **p-value < 0.05** → **Statistically significant difference** between groups.
            - **p-value ≥ 0.05** → No strong evidence of a real difference.
            """)

        # 📌 Boxplot Visualization
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(x=cleaned_data_renamed[safe_group_var], y=cleaned_data_renamed[safe_target_var], palette="coolwarm", ax=ax)
        ax.set_title(f"{target_num_var} Distribution by {group_cat_var}")
        st.pyplot(fig)

    # 📌 **Chi-Square Test**
    st.subheader("📊 Chi-Square Test for Categorical Data")

    cat_var1 = st.selectbox("Select First Categorical Variable", categorical_cols)
    cat_var2 = st.selectbox("Select Second Categorical Variable", categorical_cols)

    if cat_var1 and cat_var2:
        contingency_table = pd.crosstab(cleaned_data[cat_var1], cleaned_data[cat_var2])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        st.write(f"**Chi-Square Test Results:** Chi2 = {chi2_stat:.4f}, p-value = {p_value:.4f}")
    
        with st.expander("ℹ️ How to Interpret This Test"):
            st.markdown("""
            - **p-value < 0.05** → The two variables are **related**.
            - **p-value ≥ 0.05** → No significant relationship.
            """)
        # 📌 Bar Chart Visualization
        fig, ax = plt.subplots(figsize=(7, 5))
        contingency_table.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax)
        ax.set_title(f"{cat_var1} vs. {cat_var2}")
        st.pyplot(fig)
        
    # 📌 **Regression Analysis**
    st.subheader("📈 Regression Analysis")

    target_reg_var = st.selectbox("Select a Target Variable (Numerical)", ["Dry thickness", "AML"])
    predictor_vars = st.multiselect("Select Predictor Variables", numerical_cols)

    if target_reg_var and predictor_vars:
        # ✅ Fix Column Formatting for Regression
        safe_target_reg_var = sanitize_column_name(target_reg_var)
        safe_predictors = [sanitize_column_name(var) for var in predictor_vars]

        cleaned_data_renamed = cleaned_data.rename(columns={target_reg_var: safe_target_reg_var, **{var: safe_predictors[i] for i, var in enumerate(predictor_vars)}})

        formula = f"{safe_target_reg_var} ~ {' + '.join(safe_predictors)}"
        model = smf.ols(formula, data=cleaned_data_renamed).fit()

        st.write(model.summary())

        # 📌 Scatter Plot for Regression
        if len(predictor_vars) == 1:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.regplot(x=cleaned_data[predictor_vars[0]], y=cleaned_data[target_reg_var], scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}, ax=ax)
            ax.set_title(f"Regression: {predictor_vars[0]} vs. {target_reg_var}")
            st.pyplot(fig)
        


# 🔹 Correlation Matrix Page
elif menu == "📊Correlation Matrix":
    st.title("📊 Correlation Matrix")
    
    # 🎯 Display Focused Correlation Heatmap
    st.subheader("📈 Feature Correlation Heatmap (Focused on Key Targets)")
    plot_correlation_heatmap(cleaned_data)
    
    with st.expander("ℹ️ Understanding the Focused Heatmap"):
        st.markdown("""
        - This **heatmap visualizes the correlation** between features in the production process.  
        - **Values range from -1 to +1**:  
          - **+1 (Red)** → Strong **positive** correlation (Both features increase together).  
          - **-1 (Blue)** → Strong **negative** correlation (One increases while the other decreases).  
          - **0 (White)** → No correlation.  
        - **Why is this useful?**  
          - Helps identify **critical process dependencies**.  
          - Determines which **factors strongly influence Dry Thickness & Active Material Loading (AML)**.  
          - Detects **variables that can be adjusted together or need isolation**.  
        """)
    
    # 🎯 Display Full Correlation Heatmap
    st.subheader("📈 Full Feature Correlation Heatmap")
    plot_full_correlation_heatmap(cleaned_data)
    
    with st.expander("ℹ️ Understanding the Full Heatmap"):
        st.markdown("""
        - This **full correlation heatmap** displays **all variable relationships**.  
        - It helps uncover **hidden dependencies** between all parameters in the dataset.  
        - **Use it to explore complex interactions** that may affect electrode quality.  
        - You can spot clusters of related features and analyze process variations.
        """)
    
    # 🔬 **Deep Dive: Key Correlation Insights**
    st.subheader("🔬 Key Insights from Correlation Analysis")

    # 🔹 **Strong Positive Relationships**
    with st.expander("✅ Strong Positive Correlations"):
        st.write("""
        - **Vacuum Pressure & Vacuum Temperature (~1.00)**  
          - Changes in vacuum pressure are **directly linked** to vacuum temperature.
          - Implication: These **must be adjusted together** to maintain process stability.
        
        - **Dry Thickness & Doctor Blade Gap (~0.72)**  
          - A **wider doctor blade gap leads to greater thickness**.
          - Implication: **Doctor Blade settings should be precisely controlled** to maintain thickness targets.
        
        - **Mixing Speed & Time in Individual Steps (e.g., STEP 4 Speed & Time ~0.85)**  
          - Higher speeds **are usually paired** with longer times in the same step.
          - Implication: **Mixing adjustments should account for both speed & time to maintain consistency.**
        """)

    # 🔹 **Strong Negative Relationships**
    with st.expander("⚠️ Strong Negative Correlations"):
        st.write("""
        - **Vacuum Pressure & Vacuum Time (~ -0.98)**  
          - Higher vacuum pressure **reduces the required vacuum time**.
          - Implication: **Increasing vacuum pressure shortens process time** but may impact quality.

        - **Vacuum Temperature & Vacuum Time (~ -0.90)**  
          - Higher temperatures **lead to shorter vacuum times**.
          - Implication: **Optimizing vacuum temp & time can balance efficiency & material properties**.
        """)

    # 🔹 **Clustered Relationships**
    with st.expander("📊 Grouped Parameter Relationships"):
        st.write("""
        - **Vacuum Parameters (Pressure, Temperature, & Time)**
          - Highly correlated **(+0.98 to +1.00)** → These **must be adjusted together** for stability.
        
        - **Mixing Process Variables (Speeds & Times)**
          - Correlate **within the same step** and **partially across steps**.
          - Implication: **Mixing adjustments should be analyzed per step rather than across steps.**
        
        - **Independent Variables**
          - **Drying Temperature & Time**: **Weak correlations** with other parameters → **independently controlled**.
          - **Solid Content (SC weight%)**: Generally weak correlations → **not a dominant factor**.
        """)

    # 🚀 **Final Recommendations**
    st.subheader("🚀 Key Recommendations")
    st.markdown("""
    - **🛠️ Doctor Blade Gap is a critical control parameter** → Directly impacts Dry Thickness.
    - **⚙️ Optimize Coating Speed** carefully → Faster speeds can reduce layer thickness & material loading.
    - **🔍 Mixing consistency is key** → Each step should be optimized independently.
    - **📊 Vacuum parameters should be adjusted together** → Strong interdependencies require balancing pressure, temp, & time.
    """)
# 🔹 Understanding the analysis
elif menu == "📖Understanding the Analysis":
    st.title("📖Understanding the Analysis")

    # Introduction
    st.markdown("### 📌 Why Statistical Analysis Matters in Electrode Manufacturing")
    st.write("""
    Statistical analysis helps identify key process relationships and ensures that changes in production 
    parameters are backed by data, not guesswork. This section explains the main statistical techniques used 
    in this app and how to interpret their results.
    """)

    # Section 1: Feature Distributions
    st.header("1️⃣ Feature Distributions: Understanding Your Data")
    st.write("""
Feature distribution analysis helps in understanding the variability and consistency of electrode production. The provided visualizations—a histogram with a KDE plot and a boxplot—give insights into how the dry thickness measurements are distributed across different batches.

### **What the Histogram & KDE Plot Show:**
- The histogram (bar chart) represents the frequency of different dry thickness values.
- The KDE (smooth curve) helps identify the overall shape of the distribution.
- This plot suggests that most values are concentrated around 50-60, but there are variations across batches.

### **How to Interpret the Histogram:**
- Bell shape = normal distribution (evenly spread values)
- Multiple peaks = different production conditions
- Long tails = potential outliers from inconsistencies or measurement errors

### **What the Boxplot Shows:**
- The **box (green)** spans from Q1 (25th percentile) to Q3 (75th percentile)
- The **vertical line inside the box** is the median Q2 (50th percetnile).
- The **whiskers**: Typically extend to the most extreme data points within 1.5 × IQR from the box edges
- Points beyond the whiskers: Considered outliers

### **Key Takeaways:**
- ✔ If you notice multiple peaks in the histogram, check for changes in production parameters across batches.
- ✔ Outliers in the boxplot may indicate measurement errors, machine calibration issues, or process variability.
- ✔ Consistent distributions suggest stable manufacturing conditions, while inconsistent patterns may highlight areas for process improvement.

📸 **Example Screenshot:**
""")

# Display uploaded screenshot for Feature Distributions
    uploaded_image_path = "images/distributions.png"
    st.image(uploaded_image_path, caption="Feature Distributions: Histogram & Boxplot of Dry Thickness", use_container_width=True)

# Section 2: Correlation Analysis
    st.header("2️⃣ Correlation Analysis: Identifying Relationships")
    st.write("""
Scatter plots help visualize relationships between two continuous variables. In electrode manufacturing, this is crucial for understanding how process parameters influence product characteristics.

### **What These Scatter Plots Show:**
- The left scatter plot represents the relationship between **Doctor Blade Gap** and **Dry Thickness**.
- The right scatter plot represents the relationship between **Vacuum Pressure (mbar)** and **Dry Thickness**.
- The **red regression lines** indicate the general trend between the variables.
- The **shaded areas** around the regression lines represent confidence intervals.

### **How to Interpret the Left Scatter Plot (Doctor Blade vs. Dry Thickness):**
- There is a **strong positive correlation** between the **Doctor Blade Gap** and **Dry Thickness**.
- As the **Doctor Blade Gap increases**, **Dry Thickness also increases**.
- The data points are relatively close to the regression line, suggesting a strong and consistent relationship.
- This means that adjusting the **Doctor Blade Gap** can directly control the **Dry Thickness** of the electrodes.

### **How to Interpret the Right Scatter Plot (Vacuum vs. Dry Thickness):**
- The relationship between **Vacuum Pressure and Dry Thickness** appears **very weak or nonexistent**.
- The regression line is almost flat, indicating that changes in vacuum pressure have little to no impact on dry thickness.
- The points are widely scattered, suggesting that other factors influence Dry Thickness more significantly than vacuum pressure.
- This means that adjusting **Vacuum Pressure alone** is unlikely to control **Dry Thickness**, and other process parameters should be considered.

### **Key Takeaways:**
✔ **Doctor Blade Gap has a strong impact** on Dry Thickness—adjusting it can help control coating thickness.
✔ **Vacuum Pressure does not significantly influence Dry Thickness**—other parameters like drying time or material composition may be more important.
✔ If trying to improve Dry Thickness uniformity, focus on factors with stronger correlations rather than those with weak or no relationship.
✔ Outliers in both plots should be checked for process anomalies or measurement errors.
📸 **Example Screenshot:**
""")

    # Display uploaded screenshot for Scatter Plot
    uploaded_image_path = "images/scatter.png"
    st.image(uploaded_image_path, caption="Scatter Plot: Doctor Blade vs Dry Thickness", use_container_width=True)

# 🔹 Understanding Hypothesis
elif menu == "📖Understanding Hypothesis":
    st.title("📖Understanding Hypothesis")
    st.write("""
Hypothesis testing provides a systematic framework to determine whether observed differences in electrode manufacturing processes represent genuine effects or random variation. This approach transforms subjective assessments into quantifiable evidence.

### **Why Hypothesis Testing Matters**
- Helps verify if process modifications lead to meaningful improvements.
- Ensures data-driven decisions rather than assumptions.
- Identifies whether variations in production settings significantly impact key quality parameters.

This section will cover **t-Test and ANOVA**, two commonly used hypothesis tests in manufacturing analysis.

---""")  

# Section: Group Comparison - t-Test & ANOVA
    st.header("ℹ️ Group Comparison: t-Test & ANOVA")
    st.write("""
        
    ### **🔹 t-Test**
    - Compares the means of **two groups** to determine if they are significantly different.
    - Provides a **t-statistic** and a **p-value**.
        
    ### **📌 What is the t-Statistic?**
    The t-statistic (or t-value) measures the difference between the means of two groups relative to the variability in the data. It is used in a t-test to determine whether the means of two groups are significantly different from each other.
    
    **Interpretation of t-values:**
    - Higher absolute values of t → greater difference between groups
    - Closer to 0 → groups are similar
    
    **Typical Ranges for t-statistic:**
    - Small effect: |t| ≈ 1 - 2 (Weak evidence against null hypothesis)
    - Moderate effect: |t| ≈ 2 - 3 (Moderate evidence)
    - Strong effect: |t| > 3 (Stronger evidence)
    - If p-value < 0.05, the difference is statistically significant
    
    🔹 **Example from our results:**
    The t-statistic = 3.35, which is quite high, meaning there is a strong difference in Dry Thickness between Good (Y) and Bad (N) electrodes.
    
    ---
    
    ### **🔹 ANOVA (Analysis of Variance)**
    - Compares the means of **three or more groups** to determine if at least one group is significantly different.
    - Provides an **F-statistic** and a **p-value**.
        
    ### **📌 What is the F-Statistic?**
    The F-statistic is used in ANOVA (Analysis of Variance) to compare variance among multiple groups. It tells us whether the groups have significantly different means.
    
    **Interpretation of F-values:**
    - Higher F-value → greater difference between group means relative to the variation within groups.
    - Lower F-value → more overlap between groups, meaning they are similar
    
    **Typical Ranges for F-statistic:**
    - F ≈ 1: Groups are similar; no significant difference.
    - F > 1: Some difference exists; the larger the F, the stronger the difference.
    - F >> 1 (e.g., 4 or higher): Strong evidence that at least one group differs significantly.
    - If p-value < 0.05, at least one group differs significantly.
    
   
    
    ## 📊 Dry Thickness in Good and Bad Electrodes
    
    ### **T-Test Analysis**
    📌 **T-statistic** = **3.3524**  
    📌 **p-value** = **0.0017** (which is < 0.05, meaning the difference is significant)  
    
    🔎 **Interpretation:**
    - There is a significant difference in **Dry Thickness** between **Good (Y)** and **Bad (N)** electrodes.
    - The **Bad (N) group** has a **higher median Dry Thickness** compared to the **Good (Y) group**.
    - This suggests that rejected electrodes tend to have **higher thickness values**.
   
       
    ✅ **Conclusion:** **Thickness impacts electrode classification**.  
    Bad electrodes tend to have **higher thickness**, which might be influencing technician decisions.
    
    ---
    
    ## 📊 ANOVA Test: Dry Thickness Partner Comparison (POLITO, ABEE, NANOMAKERS)
    
    - The **F-statistic** measures variance among groups.
    - A **p-value  > 0.05** suggests that at least **one group significantly differs**.
    - If significant, **post-hoc tests** identify which groups differ.
    
    ### **Example Interpretation**
    The boxplot compares **Dry Thickness** between two groups:
    - **N (No Issue)** vs. **Y (Yes, Issue)**
    
    📌 **Key Takeaways:**
    ✔ **p-value = 0.0017** → Indicates a significant difference in Dry Thickness.
    ✔ **Mean thickness is higher for group N**, suggesting **process variations impact outcomes**.
    ✔ **Outliers in group Y** indicate **higher variability**, potentially due to **inconsistent process conditions**.
    ✔ This insight helps technicians **adjust process parameters** to improve **product consistency**.
    
    📸 **Example Screenshot:**
    *(Hypothesis Testing: t-Test & ANOVA on Dry Thickness)*
    
    """)

 
# Display uploaded screenshot for Hypothesis Testing
    uploaded_image_path = "images/anova.png"
    st.image(uploaded_image_path, caption="Hypothesis Testing: t-Test & ANOVA on Dry Thickness", use_container_width=True)