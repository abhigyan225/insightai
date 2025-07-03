
# 📊 InsightAI

**Your smart, no-code data science copilot – explore, clean, visualize & model your data effortlessly.**

---

## 🔗 Live App

> [Click here to open the app](https://insightai-j9mmsuah5apurrkmlappcgu.streamlit.app)

---

## 📁 Features Overview

| Feature Category      | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| 📂 Upload CSV         | Upload data file (CSV, max 200MB)                                           |
| 💬 Ask Questions      | Ask about data (e.g., mean, missing values)                                 |
| 📈 Chart Generator    | Generate charts with simple prompts (e.g., `plot sales vs profit`)          |
| 📊 Summary Stats      | Get quick descriptive statistics for all columns                            |
| 🧼 Data Cleaning      | Handle missing values, duplicates, etc.                                     |
| 🏷️ Rename Columns     | Rename column headers easily                                                 |
| 🧪 Filter Rows        | Filter rows based on conditions                                              |
| 💡 Auto Insights      | Automatically detects trends & insights                                     |
| 💾 Export Cleaned Data| Download the cleaned version of your dataset                                |
| 🤖 ML Model Trainer   | Train a machine learning model with minimal configuration                   |
| 📡 Predict with Model | Upload new data and use saved model for predictions                         |
| 📝 Export Report      | Generate a PDF report of your data exploration                              |

---

## 🚀 How to Use

### 1. 📂 Upload CSV File

- Drag and drop a `.csv` file.
- Size limit: 200MB.

### 2. 💬 Ask a Question About Your Data

- Natural language queries like:
  - `"mean of age"`
  - `"missing values"`
  - `"unique categories in gender"`

### 3. 📈 Want a Chart?

- Ask for charts using simple phrases like:
  - `"plot sales vs profit"`
  - `"histogram of age"`
  - `"bar chart of region vs revenue"`

### 4. 📊 Summary Statistics

- Automatically shows:
  - Count, mean, std, min, max, etc.
  - Useful for numerical and categorical variables.

### 5. 🧼 Data Cleaning

- Cleans the dataset by:
  - Removing nulls or filling them.
  - Dropping duplicates.
  - Detecting outliers or inconsistencies.

### 6. 🏷️ Rename Columns

- Simple interface to rename any column headers to readable names.

### 7. 🧪 Filter Rows

- Filter the dataset using logical conditions:
  - e.g., `"age > 25"`, `"region == 'South'"`

### 8. 💡 Auto Insights

- Provides automated summaries and trends:
  - "Top contributing factors"
  - "Most frequent values"
  - "Important numeric distributions"

### 9. 💾 Export Cleaned Data

- Allows download of the cleaned dataset in `.csv` format.

### 10. 🤖 ML Model Trainer

- Select target column and input features.
- Automatically trains models (like Decision Tree, Random Forest, etc.).
- Evaluates accuracy and other metrics.

### 11. 📡 Predict Using Saved Model 

- Upload new data (same format as training).
- Outputs predictions based on the previously trained model.

### 12. 📝 Export Report

- Generate a full PDF summary:
  - Summary stats
  - Plots
  - Insights
  - Model metrics

---

## 🧱 Tech Stack

- **Streamlit** – UI Framework
- **Pandas** – Data Handling
- **Scikit-learn** – ML Training
- **Matplotlib / Seaborn / Plotly** – Visualizations
- **PDFKit / ReportLab** – PDF Report Generation
- **LangChain/OpenAI (if used)** – Natural language queries

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`**:

```txt
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
pdfkit
openai
langchain
```

---

## 📂 File Structure

```plaintext
├── app.py                   # Main Streamlit App
├── data/                    # (Optional) Example CSV files
├── models/                  # Trained ML models (saved via joblib)
├── outputs/                 # Cleaned data, reports
└── README.md                # Project Documentation
```

---

## 🧠 Example Prompts

| Task                 | Prompt Example                |
|----------------------|-------------------------------|
| Summary Stats        | `"summary statistics"`         |
| Missing Values       | `"show missing values"`        |
| Plotting             | `"plot revenue vs region"`     |
| Filtering Rows       | `"filter where age > 30"`      |
| Model Training       | `"train model on income"`      |
| Predictions          | `"predict using saved model"`  |

---
