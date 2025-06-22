from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd
import os

class PDF(FPDF):
    def __init__(self):
        super().__init__(orientation='L')  # landscape mode
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Auto-Generated Data Report", ln=True, align="C")

    def add_table(self, df: pd.DataFrame, title="Data Preview"):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", size=5.5)
        self.ln(2)

        col_width = (self.w - 20) / len(df.columns)
        row_height = 5

        # Header
        for col in df.columns:
            self.cell(col_width, row_height, str(col)[:15], border=1)
        self.ln()

        # Rows
        for _, row in df.head(20).iterrows():
            for col in df.columns:
                val = str(row[col])[:15]
                self.cell(col_width, row_height, val, border=1)
            self.ln()



    def add_image(self, path, caption=""):
        self.add_page()
        self.set_font("Arial", size=12)
        self.cell(0, 10, caption, ln=True)
        self.image(path, x=10, w=180)
