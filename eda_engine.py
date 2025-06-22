import pandas as pd

def get_answer(df: pd.DataFrame, query: str) -> str:
    query = query.lower()

    

    if "null" in query or "missing" in query:
        nulls = df.isnull().sum()
        missing = nulls[nulls > 0]
        if missing.empty:
            return "✅ No missing values found in any column!"
        return f"Missing values per column:\n{missing.to_string()}"

    

    elif "mean" in query:
        numeric = df.select_dtypes(include='number')
        means = numeric.mean()
        return f"Mean of numeric columns:\n{means.to_string()}"
    
    elif "column names" in query or "columns" in query:
        return "Columns in the dataset:\n" + ", ".join(df.columns)

    elif "max" in query:
        numeric = df.select_dtypes(include='number')
        max_vals = numeric.max()
        return f"Maximum values:\n{max_vals.to_string()}"

    elif "min" in query:
        numeric = df.select_dtypes(include='number')
        min_vals = numeric.min()
        return f"Minimum values:\n{min_vals.to_string()}"
    
    elif "shape" in query or "rows" in query or "columns" in query:
        return f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns."

    else:
        return "Sorry, I couldn't understand your question. Try something like: 'show missing values' or 'mean of columns'."
