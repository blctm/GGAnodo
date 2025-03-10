import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import base64
from scipy.optimize import minimize
import io
from scipy import stats
from utils import (
    cleaned_data, numerical_cols, categorical_cols,
    plot_feature_distributions, plot_categorical_bars,
    plot_scatter_vs_target, plot_correlation_heatmap, plot_full_correlation_heatmap
)

# 🔐 Password Protection
def check_password():
    """Authenticate user before accessing the app."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.sidebar.subheader("🔐 Login Required")
        password_input = st.sidebar.text_input("Enter Password:", type="password")

        correct_password = st.secrets.get("app_password")

        if correct_password is None:
            st.sidebar.error("⚠️ Password is not set! Please check Streamlit Cloud settings.")
            st.stop()

        if st.sidebar.button("Login"):
            if password_input == correct_password:
                st.session_state.authenticated = True
                st.sidebar.success("✅ Access Granted!")
                st.rerun()  # 🔥 Use `st.rerun()` instead of `st.experimental_rerun()`
            else:
                st.sidebar.error("❌ Incorrect password. Try again.")

        st.stop()

# Call password check before anything else runs
check_password()

# 🔹 Custom CSS for Background & Sidebar
st.markdown(
    """
   <style>
        [data-testid="stAppViewContainer"] {
            background-color: #D2E5E3;
        }
        [data-testid="stSidebar"] {
            background-color: #2E5B5887;
        }
        h1, h2, h3, h4 {
            color: #333333; 
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔹 Sidebar Navigation

def sidebar_navigation():
    st.sidebar.image("logo.png", width=150)
    st.sidebar.title("📖 Anode Analysis Wiki")
    return st.sidebar.radio(
        "Select a section:",
        ["🏠 Home", "🔍Overview", "📊Feature Statistics","📖Visualisations", "🔢Visualising Statistics", 
         "📊Correlation Matrix", "📖Understanding Hypothesis", "❓Hypothesis Testing", "📖Understanding OLS","📉Regression Analysis", "🔍Reverse Engineer OLS"]
    )

menu = sidebar_navigation()

# 🔹 Load Dataset
@st.cache_data
def load_data():
    return pd.read_excel("data.xlsx", sheet_name="Final")

data = load_data()

def sanitize_column_name(name):
    return name.replace(" ", "_").replace("%", "pct").replace("/", "_").replace("(", "").replace(")", "").replace("-", "_").replace(".", "_").replace("º", "o")

# 🔹 Page Handling
if menu == "🏠 Home":
    st.title("📖 Anode Analysis Wiki")

    # Styled welcome message
    st.markdown(
        "<h2 style='color: green; text-align: center;'>Welcome to the anode wiki! 📚</h2>"
        "<h4 style='color: green; text-align: center;'>Use the sidebar to navigate through different insights.</h4>",
        unsafe_allow_html=True
    )

    # Display image in center
    # Load and resize the image
    # image = Image.open("images/factory.png")
    # width, height = image.size
    # new_size = (int(width * 0.85), int(height * 0.85))  # Reduce size by 15%
    # resized_image = image.resize(new_size)
    # st.image(resized_image)
    uploaded_image_path = "images/gfactory.png"
    st.image(uploaded_image_path)

    
#overview
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

elif menu == "📊Feature Statistics":
    st.title("📊Feature Statistics")

    st.markdown("""
    This section provides **descriptive statistics** for numerical features in the dataset.  
    Use it to understand distributions before performing regression or hypothesis testing.
    """)

    # ✅ Display full statistics table
    st.subheader("📋 Summary Statistics for All Features")
    full_stats = cleaned_data[numerical_cols].describe().T  # Transpose for better readability
    st.dataframe(full_stats)

    # ✅ Let users select a feature for detailed statistics
    selected_feature = st.selectbox("🔍 Select a Numerical Feature", numerical_cols)

    if selected_feature:
        st.subheader(f"📈 Detailed Statistics for: {selected_feature}")

        # Extract statistics
        stats_data = cleaned_data[selected_feature].describe()
        stats_df = pd.DataFrame(stats_data).T  # Convert to dataframe

        # ✅ Display statistics
        st.table(stats_df)



# 🔹 Understanding the analysis
elif menu == "📖Visualisations":
    st.title("📖Visualisations")

    # Introduction
    st.markdown("### 📌Why Data Visualization Matters in Electrode Manufacturing")
    st.write("""
    Visualizing data through scatterplots, histograms, and correlation analysis helps uncover patterns, relationships, and trends in electrode manufacturing. This section explores key graphical techniques used in this app and how to interpret them effectively..
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
- ✔ **Doctor Blade Gap has a strong impact** on Dry Thickness—adjusting it can help control coating thickness.
- ✔ **Vacuum Pressure does not significantly influence Dry Thickness**—other parameters like drying time or material composition may be more important.
- ✔ If trying to improve Dry Thickness uniformity, focus on factors with stronger correlations rather than those with weak or no relationship.
- ✔ Outliers in both plots should be checked for process anomalies or measurement errors.
📸 **Example Screenshot:**
""")

    # Display uploaded screenshot for Scatter Plot
    uploaded_image_path = "images/scatter.png"
    st.image(uploaded_image_path, caption="Scatter Plot: Doctor Blade vs Dry Thickness", use_container_width=True)


# 🔹 Understanding Hypothesis
elif menu == "📖Understanding Hypothesis":
    st.title("📖Understanding Hypothesis")
   
    # Create Tabs for Different Tests
    tab1, tab2 = st.tabs(["**1️⃣t-Test & ANOVA (Numerical Comparisons)**", "**2️⃣ Chi-Square Test (Categorical Comparisons)**"])

    # Tab for t-Test & ANOVA
    with tab1:
        st.header("Group Comparison: t-Test & ANOVA")
        st.write("""
        These tests help determine if there are meaningful differences between groups based on production parameters.

        ### **t-Test**
        - Compares the means of **two groups** (e.g., Dry Thickness for two different process settings).
        - Provides a **t-statistic** and a **p-value** to determine statistical significance.
        - **p-value < 0.05**: A significant difference exists.

        ### **ANOVA (Analysis of Variance)**
        - Compares means across **multiple groups**.
        - Uses an **F-statistic** to measure variance among groups.
        - **p-value < 0.05**: At least one group differs significantly.

        ### **Interpretation:**
        - Higher **t-statistic** or **F-statistic** = Greater difference among groups.
        - **p-value < 0.05** means the observed difference is statistically significant, not just random.

        **Example Insight:**
        - If a t-test finds a significant difference in **Dry Thickness** between two production settings, the manufacturing process may need adjustments.
        - If ANOVA suggests differences across multiple suppliers, it may indicate inconsistent material quality.
        """)
        # Display uploaded screenshot for ANOVA Test
        # image = "images/anova.png"  # Corrected path
        st.image("images/anova.png", caption="ANOVA: Dry Thickness vs Process Setting", use_container_width=True)

    # Tab for Chi-Square Test
    with tab2:
        st.header("Chi-Square Test: Categorical Comparisons")
        st.write("""
        The **Chi-Square Test** is used to determine whether two categorical variables are related.
    
    ### **What This Test Shows:**
    - Evaluates whether there is a **statistical relationship** between two categorical variables.
    - If the **p-value is < 0.05**, it suggests a **significant relationship** between the categories.
    - If the **p-value is > 0.05**, the variables are likely **independent** (no strong relationship).
    
    ### **How to Interpret the Results:**
    - **Chi-Square Statistic (Chi²)**: Measures the difference between observed and expected frequencies.
    - **p-value**:
      - **p < 0.05**: The variables are significantly related (e.g., Mixing Type impacts Outcome).
      - **p > 0.05**: No significant relationship; the variables are independent.
    
    ### **Example Interpretation:**
    The bar chart compares **Mixing Type (Dispermat vs. Planetary)** against **Outcome (Pass/Fail)**. The results indicate:
    ✔ A **Chi² value of 2.3130** with a **p-value of 0.1283** suggests no significant relationship.
    ✔ This means **Mixing Type does not strongly influence Outcome** in this dataset.
    ✔ The differences observed in the chart may be due to random variation rather than a true process effect.
    ✔ If the p-value were lower (e.g., <0.05), we would conclude that Mixing Type does impact Outcome.
    """)
    
    # Display uploaded screenshot for Chi-Square Test
    # uploaded_image_path_chi = "images/chi.png"
        st.image("images/chi.png", caption="Chi-Square Test: Mixing Type vs. Outcome", use_container_width=True)
        ### 



elif menu == "🔢Visualising Statistics":
    st.title("🔢Visualising Statistics")
    st.subheader("📊 Feature Distributions")
    selected_feature = st.selectbox("Select a Feature to Visualize", numerical_cols + categorical_cols)
    if selected_feature in numerical_cols:
        plot_feature_distributions(cleaned_data, selected_feature)
    elif selected_feature in categorical_cols:
        plot_categorical_bars(cleaned_data, selected_feature)

elif menu == "📖Understanding OLS":
    st.title("📖 Understanding OLS Regression")

    st.markdown("""
    Ordinary Least Squares (OLS) regression helps us **understand relationships between variables**.
    However, interpreting OLS results correctly is crucial for making **informed decisions**.
    """)

    st.subheader("🔹 Key Metrics in OLS Regression")

    # ✅ R-squared Explanation
    with st.expander("📊 **R² & Adjusted R²**"):
        st.markdown("""
        - **R² (Coefficient of Determination)** measures how much variance in the dependent variable is explained by the model.
        - **Adjusted R²** adjusts for the number of predictors (important when comparing models).
        
        **How to Interpret:**
        - **High R² (close to 1):** The model explains most of the variance.
        - **Low R² (close to 0):** The model does not explain much variance.
        - **Two models with the same R²?** Choose the simpler one (fewer predictors). 

        🔍 **Example Insight:**
        - If **R² = 0.85**, the model explains **85% of the variance** in `Dry_thickness`.
        """)

    # ✅ P-values Explanation
    with st.expander("📌 **P-values & Statistical Significance**"):
        st.markdown("""
        - A **p-value < 0.05** means the predictor is **statistically significant** (has a real effect).
        - A **p-value ≥ 0.05** means the predictor **may not be important** in the model.

        **How to Interpret:**
        - Keep predictors with **low p-values** and remove those with **high p-values** (unless theoretically important).
        
        🔍 **Example Insight:**
        - If `Doctor_blade` has a **p-value = 0.001**, it's a strong predictor of `Dry_thickness`.
        - If `Vacuum_temperature` has **p-value = 0.89**, it likely has no real effect.
        """)

    # ✅ Model Complexity & Parsimony
    with st.expander("⚖️ **Choosing the Best Model: Simplicity vs. Complexity**"):
        st.markdown("""
        - **AIC (Akaike Information Criterion) & BIC (Bayesian Information Criterion)** help compare models.
        - **Lower AIC/BIC** means a **better** model (balance between fit & simplicity).
        - **More predictors ≠ better model** (risk of overfitting!).

        **Key Rule:**
        - If **two models have the same R², pick the one with fewer predictors**.
        - **Avoid multicollinearity** (when predictors are highly correlated).

        🔍 **Example Insight:**
        - Model A (3 predictors, R² = 0.85, AIC = 250)
        - Model B (6 predictors, R² = 0.85, AIC = 280)
        - **Choose Model A** (simpler but equally good).
        """)

    # ✅ Checking Residuals
    with st.expander("📉 **Checking Residuals for Model Quality**"):
        st.markdown("""
        - Residuals should be **randomly distributed** (no patterns).
        - If residuals have patterns, the model **may be missing important factors**.

        🔍 **Example Insight:**
        - If residuals **cluster in certain areas**, the model is **not fully capturing relationships**.
        """)

    # st.success("Understanding these metrics will help you choose **the best regression model** for your analysis! 🚀")


elif menu == "📉Regression Analysis":
    st.title("📉 Regression Analysis")

    # User Inputs
    target_reg_var = st.selectbox("Select a Target Variable (Numerical)", ["Dry thickness", "AML"])
    predictor_vars = st.multiselect("Select Predictor Variables", numerical_cols)

    # Check if valid inputs are selected
    if target_reg_var and predictor_vars:
        # Sanitize column names
        safe_target_reg_var = sanitize_column_name(target_reg_var)
        safe_predictors = [sanitize_column_name(var) for var in predictor_vars]

        # Rename dataset columns for compatibility
        cleaned_data_renamed = cleaned_data.rename(columns={
            target_reg_var: safe_target_reg_var, 
            **{var: safe_predictors[i] for i, var in enumerate(predictor_vars)}
        })

        # Fit OLS Model
        formula = f"{safe_target_reg_var} ~ {' + '.join(safe_predictors)}"
        try:
            model = smf.ols(formula, data=cleaned_data_renamed).fit()
            st.write(model.summary())

            # Generate Regression Equation
            coef_dict = model.params.to_dict()
            intercept = coef_dict.pop("Intercept", 0)  # Extract Intercept
            equation = f"{safe_target_reg_var} = {intercept:.4f} "
            for var, coef in coef_dict.items():
                equation += f"+ {coef:.4f} * {var} "

            # Display Regression Formula
            st.markdown(f"### 📌 Regression Formula:\n**{equation}**")

            # Plot for Simple Regression (Only if 1 predictor is chosen)
            if len(predictor_vars) == 1:
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.regplot(x=cleaned_data[predictor_vars[0]], y=cleaned_data[target_reg_var], 
                            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}, ax=ax)
                ax.set_title(f"Regression: {predictor_vars[0]} vs. {target_reg_var}")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in Regression Model: {e}")



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
        

elif menu == "📊Correlation Matrix":
    st.title("📊 Correlation Matrix")
    st.subheader("📈 Feature Correlation Heatmap")
    plot_correlation_heatmap(cleaned_data)
    st.subheader("📈 Full Feature Correlation Heatmap")
    plot_full_correlation_heatmap(cleaned_data)

    # 📌 Explanation (Collapsible)
# 📌 Explanation (Collapsible)
    with st.expander("ℹ️ Understanding the Heatmap"):
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


elif menu == "🔍Reverse Engineer OLS":
    st.title("🔍Reverse Engineering OLS Predictions")

    target_reg_var = st.selectbox("Select Target Variable to Solve For", ["Dry thickness", "AML"])
    predictor_vars = st.multiselect("Select Predictor Variables", numerical_cols)

    if target_reg_var and predictor_vars:
        # Fit OLS Model
        safe_target_reg_var = sanitize_column_name(target_reg_var)
        safe_predictors = [sanitize_column_name(var) for var in predictor_vars]
        cleaned_data_renamed = cleaned_data.rename(columns={
            target_reg_var: safe_target_reg_var, 
            **{var: safe_predictors[i] for i, var in enumerate(predictor_vars)}
        })

        formula = f"{safe_target_reg_var} ~ {' + '.join(safe_predictors)}"
        model = smf.ols(formula, data=cleaned_data_renamed).fit()

        # Extract coefficients & p-values
        coef_dict = model.params.to_dict()
        pvalues_dict = model.pvalues.to_dict()
        intercept = coef_dict.pop("Intercept", 0)  # Extract intercept
        
        # Identify important & non-important features
        significant_predictors = [p for p in safe_predictors if pvalues_dict[p] < 0.05]
        non_significant_predictors = [p for p in safe_predictors if pvalues_dict[p] >= 0.05]

        # 📌 User input range for target variable
        target_min = st.number_input(f"Enter Minimum {target_reg_var}", value=50.0)
        target_max = st.number_input(f"Enter Maximum {target_reg_var}", value=60.0)

        if st.button("Find Predictor Values"):
            if not significant_predictors:
                st.warning("No significant predictors found! Using all predictors for optimization.")

            # Define the function to minimize (error function)
            def error_function(predictor_values):
                estimated_value = intercept + sum(coef_dict[p] * predictor_values[i] for i, p in enumerate(significant_predictors))
                return (estimated_value - ((target_min + target_max) / 2)) ** 2  # Squared error minimization

            # Initial guesses for significant predictors
            initial_guesses = [cleaned_data_renamed[p].mean() for p in significant_predictors]

            # Solve for best predictor values
            result = minimize(error_function, initial_guesses, method='Powell')

            if result.success:
                optimized_values = result.x
                solution_dict = {significant_predictors[i]: optimized_values[i] for i in range(len(significant_predictors))}
                
                # Include non-significant predictors set to their mean
                for ns_pred in non_significant_predictors:
                    solution_dict[ns_pred] = cleaned_data_renamed[ns_pred].mean()

                # ✅ Display results
                st.success("✅ Optimized Predictor Values to achieve target range:")
                for key, value in solution_dict.items():
                    st.write(f"- **{key}** = {value:.4f}")

            else:
                st.error("Optimization failed. Try adjusting your input range.")
