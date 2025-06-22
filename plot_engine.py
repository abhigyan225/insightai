import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_from_query(df: pd.DataFrame, query: str):
    query = query.lower()
    plt.clf()  # clear previous plot
    col_map = {col.lower(): col for col in df.columns}

    # Bar chart of value counts
    if "bar" in query and "of" in query:
        col = query.split("of")[-1].strip().strip('"').strip("'")
        col = col_map.get(col, col)
        if col in df.columns:
            sns.countplot(data=df, x=col)
            return True, f"Bar chart of {col}"

    # Histogram
    elif "histogram" in query or "distribution" in query:
        col = query.split("of")[-1].strip().strip('"').strip("'")
        col = col_map.get(col, col)
        if col in df.columns:
            sns.histplot(data=df, x=col, kde=True)
            return True, f"Histogram of {col}"

    # Scatter plot: "plot x vs y"
    elif "plot" in query and "vs" in query:
        parts = query.split("plot")[-1].strip().split("vs")
        if len(parts) == 2:
            x_col = parts[0].strip().strip('"').strip("'")
            y_col = parts[1].strip().strip('"').strip("'")
            x_col = col_map.get(x_col, x_col)
            y_col = col_map.get(y_col, y_col)
            if x_col in df.columns and y_col in df.columns:
                sns.scatterplot(data=df, x=x_col, y=y_col)
                return True, f"Scatter plot of {x_col} vs {y_col}"

    return False, "Sorry, I couldn't generate a plot. Try: 'plot sales vs profit' or 'histogram of age'"
