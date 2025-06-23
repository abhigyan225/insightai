import streamlit as st
import pandas as pd
import os



st.set_page_config(page_title="InsightAI - Your Data Science Copilot", layout="wide")

st.markdown("""
    <style>
        /* Page background */
        .stApp {
            background-color: #8DE5C9;
        }

        /* Responsive container */
        .block-container {
            max-width: 100%;
            margin: auto;
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }

        /* Title */
        .custom-title {
            color: #4B8BBE;
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0.2rem;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Subheading */
        .custom-subtitle {
            color: #6c757d;
            font-size: 18px;
            text-align: center;
            margin-bottom: 2rem;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Ensure all default Streamlit headers are visible */
        h1, h2, h3, h4 {
            color: #4B8BBE;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
        }

        /* Buttons */
        .stButton>button, .stDownloadButton>button {
            background-color: #4B8BBE;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
        }

        /* Text inputs */
        .stTextInput>div>div>input,
        .stTextArea textarea {
            background-color: #f0f0f0;
            color: black;
            text-align: center;
            font-size: 16px;
        }

        /* Responsive tweaks for mobile */
        @media screen and (max-width: 768px) {
            .custom-title {
                font-size: 32px;
            }
            .custom-subtitle {
                font-size: 16px;
                padding: 0 1rem;
            }
            .block-container {
                padding: 1rem;
                margin: 0.5rem;
            }
        }
    </style>
""", unsafe_allow_html=True)


# Render title and subheading
st.markdown("""
    <div class='custom-title'>🤖 InsightAI</div>
    <div class='custom-subtitle'>Your smart, no-code data science copilot – explore, clean, visualize & model your data effortlessly.</div>
""", unsafe_allow_html=True)



# Upload CSV
uploaded_file = st.file_uploader("📄 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    st.success("CSV uploaded successfully! Here's a preview:")
    st.dataframe(df.head())
else:
    st.info("Please upload a CSV file to begin.")

import eda_engine  # import your engine

# Input box for user question
st.subheader("🤖 Ask a question about your data")
question = st.text_input("Ask something like 'missing values' or 'mean of columns'")

if uploaded_file is not None and question:
    answer = eda_engine.get_answer(df, question)
    st.code(answer)


import plot_engine  # import the plot module
import matplotlib.pyplot as plt  # for rendering

st.subheader("📊 Want a chart? Ask here!")
plot_query = st.text_input("Try 'plot sales vs profit' or 'histogram of age'")

if uploaded_file is not None and plot_query:
    plotted, message = plot_engine.plot_from_query(df, plot_query)
    if plotted:
        st.pyplot(plt.gcf())  # show the generated plot
        st.success(message)
    else:
        st.error(message)

import predict_engine
import uuid

st.subheader("📄 Export Report")

if st.button("Generate PDF Report"):
    pdf = predict_engine.PDF()
    pdf.add_page()
    pdf.add_table(df)

    # Save current plot as image
    chart_id = f"plot_{uuid.uuid4().hex}.png"
    plt.savefig(chart_id)
    pdf.add_image(chart_id, "User-Generated Plot")

    pdf_path = f"report_{uuid.uuid4().hex}.pdf"
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download Report", f, file_name="data_report.pdf", mime="application/pdf")

    os.remove(chart_id)
    os.remove(pdf_path)


# ✅ AUTO INSIGHTS SECTION
st.subheader("🧠 Auto Insights")
if uploaded_file is not None:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

    # 1. Shape
    st.write(f"📐 Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # 2. Column types
    if st.checkbox("🔍 Show column types"):
        st.dataframe(
            df.dtypes.astype(str)
            .reset_index()
            .rename(columns={"index": "Column", 0: "Type"})
        )

    # 3. Missing values
    if st.checkbox("🚨 Show columns with missing values"):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.dataframe(
                missing.reset_index()
                .rename(columns={"index": "Column", 0: "Missing Count"})
            )
        else:
            st.success("✅ No missing values found!")

    # 4. Top correlations
    if st.checkbox("📈 Show top 5 correlated column pairs"):
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr().abs().unstack().sort_values(ascending=False)
        corr = corr[corr < 1].drop_duplicates().head(5)
        if not corr.empty:
            st.dataframe(
                corr.reset_index()
                .rename(columns={
                    "level_0": "Column A",
                    "level_1": "Column B",
                    0: "Correlation"
                })
            )
        else:
            st.warning("Not enough numeric columns to compute correlation.")

else:
    st.info("Please upload a CSV file to begin.")

# ----------------------------
# 🧹 DATA CLEANING SECTION
# ----------------------------
st.subheader("🧹 Data Cleaning")

if uploaded_file is not None:
    # 1. Find columns with missing values
    missing_cols = df.columns[df.isnull().any()].tolist()

    if missing_cols:
        st.write("Columns with missing values:", missing_cols)

        # 2. Let user choose columns to clean
        selected_cols = st.multiselect("Choose columns to clean", missing_cols)

        # 3. Choose cleaning method
        method = st.selectbox("Select cleaning method", ["Drop rows", "Fill with mean", "Fill with zero"])

        # 4. Apply button
        if st.button("Apply Cleaning"):
            if method == "Drop rows":
                df.dropna(subset=selected_cols, inplace=True)

            elif method == "Fill with mean":
                for col in selected_cols:
                    if df[col].dtype in ['float64', 'int64']:
                        df[col].fillna(df[col].mean(), inplace=True)

            elif method == "Fill with zero":
                df[selected_cols] = df[selected_cols].fillna(0)

            st.success("✅ Cleaning applied!")
            st.dataframe(df.head())
    else:
        st.info("✅ No missing values to clean.")
else:
    st.info("Please upload a CSV file to begin.")


# ----------------------------
# 🏷️ COLUMN RENAMING SECTION
# ----------------------------
st.subheader("🏷️ Rename Columns")

if uploaded_file is not None:
    st.write("Current Columns:", list(df.columns))

    col_to_rename = st.selectbox("Select a column to rename", df.columns)
    new_name = st.text_input(f"Enter new name for '{col_to_rename}'")

    if st.button("Rename Column"):
        if new_name.strip() != "":
            df.rename(columns={col_to_rename: new_name}, inplace=True)
            st.success(f"✅ Renamed '{col_to_rename}' to '{new_name}'")
            st.dataframe(df.head())
        else:
            st.warning("⚠️ New name cannot be empty.")
else:
    st.info("Please upload a CSV file to begin.")

# ----------------------------
# 🧪 ROW FILTERING SECTION
# ----------------------------
st.subheader("🧪 Filter Rows")

if uploaded_file is not None:
    filter_col = st.selectbox("Choose column to filter", df.columns)

    condition = st.selectbox("Select condition", ["equals", "not equals", "greater than", "less than", "contains"])
    user_value = st.text_input("Enter value to filter by")

    if st.button("Apply Filter"):
        try:
            if condition == "equals":
                filtered_df = df[df[filter_col] == eval(user_value)]
            elif condition == "not equals":
                filtered_df = df[df[filter_col] != eval(user_value)]
            elif condition == "greater than":
                filtered_df = df[df[filter_col] > eval(user_value)]
            elif condition == "less than":
                filtered_df = df[df[filter_col] < eval(user_value)]
            elif condition == "contains":
                filtered_df = df[df[filter_col].astype(str).str.contains(user_value)]
            st.success(f"✅ Filter applied: {filter_col} {condition} {user_value}")
            st.dataframe(filtered_df.head())
        except Exception as e:
            st.error(f"❌ Filter error: {e}")
else:
    st.info("Please upload a CSV file to begin.")

# ----------------------------
# 📊 SUMMARY STATISTICS
st.subheader("📊 Summary Statistics")

if uploaded_file is not None:
    if st.checkbox("📋 Show summary stats"):
        numeric_df = df.select_dtypes(include="number")
        st.dataframe(numeric_df.describe().transpose())
else:
    st.info("Please upload a CSV file to begin.")

# ----------------------------
# 💾 EXPORT CLEANED CSV
# ----------------------------
st.subheader("💾 Export Cleaned Data")

if uploaded_file is not None:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a CSV file to begin.")

# ----------------------------
# 🤖 ML MODEL TRAINER
# ----------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report

st.subheader("🤖 ML Model Trainer")

if uploaded_file is not None:
    target = st.selectbox("🎯 Select your target column", df.columns)

    if target:
        # Auto-detect if classification or regression
        if df[target].dtype == 'object' or df[target].nunique() <= 10:
            problem_type = "classification"
        else:
            problem_type = "regression"

        st.write(f"🔍 Detected task: **{problem_type}**")

        # Split features and target
        X = df.drop(columns=[target])
        y = df[target]

        # Handle non-numeric features
        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if problem_type == "classification":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"🎯 Accuracy: {acc:.2f}")
            #st.text("📋 Classification Report:")
            #st.text(classification_report(y_test, y_pred))
            from sklearn.metrics import classification_report
            import pandas as pd

            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            st.text("📋 Classification Report:")
            st.dataframe(report_df.style.format(precision=2))
            # 🔽 Save model and input columns
            import pickle
            with open("trained_model.pkl", "wb") as f:
               pickle.dump(model, f)

            with open("input_columns.pkl", "wb") as f:
               pickle.dump(X.columns.tolist(), f)

            st.success("✅ Model and structure saved!")


        else:  # regression
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            st.success(f"📈 R² Score: {r2:.2f}")
            import pickle
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)

            with open("input_columns.pkl", "wb") as f:
                pickle.dump(X.columns.tolist(), f)

            st.success("✅ Model and structure saved!")
else:
    st.info("Please upload a CSV file to begin.")

import predict_engine  # ✅ Add this at the top of app.py with your other imports
print("✅ LOADED MODULE FROM:", predict_engine.__file__)

# 📡 NEW SECTION — Predict on new uploaded data
st.subheader("📡 Predict Using Saved Model")

new_data_file = st.file_uploader("Upload new data for prediction", type=["csv"], key="predict_upload")

if new_data_file is not None:
    df_predicted, status = predict_engine.predict_from_uploaded_file(new_data_file)

    if df_predicted is not None:
        st.success(status)
        st.dataframe(df_predicted)

        # Download predictions
        csv = df_predicted.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    else:
        st.error(status)






