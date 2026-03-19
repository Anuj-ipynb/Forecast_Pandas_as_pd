import pandas as pd

def load_all_sheets(file):
    def clean(df):
        df.columns = df.columns.str.strip()
        return df

    bookings = clean(pd.read_excel(file, sheet_name="Data Pack - Actual Bookings", header=3))
    scms = clean(pd.read_excel(file, sheet_name="SCMS", header=2))
    vms = clean(pd.read_excel(file, sheet_name="VMS", header=2))
    bigdeal = clean(pd.read_excel(file, sheet_name="Big Deal", header=0))

    return bookings, scms, vms, bigdeal