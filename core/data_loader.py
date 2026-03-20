import pandas as pd
import re

def parse_quarter(q):
    q = str(q).strip()
    if re.match(r'^FY\d{2} Q[1-4]$', q):
        return q
    match = re.match(r'^(\d{4})Q([1-4])$', q)
    if match:
        fy = int(match.group(1)) - 2000
        return f"FY{fy} Q{match.group(2)}"
    return q

def load_bookings(file_path):
    df = pd.read_excel(file_path, sheet_name="Data Pack - Actual Bookings", header=2)
    df.columns = df.columns.str.strip()
    quarter_cols = [c for c in df.columns if isinstance(c, str) and re.match(r'^FY\d{2} Q[1-4]$', c)]
    id_cols = [c for c in df.columns if c not in quarter_cols][:2]

    long = df.melt(id_vars=id_cols, value_vars=quarter_cols, var_name="Quarter", value_name="Bookings")
    long = long.rename(columns={id_cols[0]: "PLID (Masked)", id_cols[1]: "Product Category"})
    long["Bookings"] = pd.to_numeric(long["Bookings"], errors="coerce")
    long = long.dropna(subset=["PLID (Masked)"])

    # Remove summary rows
    def valid_cat(x):
        if not isinstance(x, str): return False
        return not re.search(r'^(Decline|Growth|Accuracy|Total|Grand)', x, re.I)
    long = long[long["Bookings"].notna() & long["Product Category"].apply(valid_cat)]
    long["Quarter"] = long["Quarter"].apply(parse_quarter)
    return long.sort_values(["PLID (Masked)", "Quarter"])


def load_scms(file_path):
    df = pd.read_excel(file_path, sheet_name="SCMS", header=2)
    df.columns = df.columns.str.strip()
    quarter_cols = [c for c in df.columns if isinstance(c, str) and re.match(r'^FY\d{2} Q[1-4]$', c)]
    id_cols = [c for c in df.columns if c not in quarter_cols][:2]

    long = df.melt(id_vars=id_cols, value_vars=quarter_cols, var_name="Quarter", value_name="SCMS_units")
    long = long.rename(columns={id_cols[0]: "PLID (Masked)", id_cols[1]: "Sales Coverage Code"})
    long["SCMS_units"] = pd.to_numeric(long["SCMS_units"], errors="coerce").fillna(0)
    long["Quarter"] = long["Quarter"].apply(parse_quarter)
    return long.groupby(["PLID (Masked)", "Quarter"])["SCMS_units"].sum().reset_index()


def load_vms(file_path):
    df = pd.read_excel(file_path, sheet_name="VMS", header=2)
    df.columns = df.columns.str.strip()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.rename(columns={"Unnamed: 1": "PLID (Masked)", "Unnamed: 2": "VMS Name"})

    quarter_cols = [c for c in df.columns if re.match(r'^\d{4}Q[1-4]$|^FY', str(c))]
    long = df.melt(id_vars=["PLID (Masked)", "VMS Name"], value_vars=quarter_cols,
                   var_name="Quarter", value_name="VMS_units")
    long = long[long["Quarter"].astype(str).str.match(r'^\d{4}Q[1-4]$|^FY')]
    long["Quarter"] = long["Quarter"].apply(parse_quarter)
    long["VMS_units"] = pd.to_numeric(long["VMS_units"], errors="coerce").fillna(0)
    return long.groupby(["PLID (Masked)", "Quarter"])["VMS_units"].sum().reset_index()


def load_bigdeal(file_path):
    df = pd.read_excel(file_path, sheet_name="Big Deal", header=0)
    df.columns = df.columns.str.strip()
    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # ── ROBUST PLID DETECTION (this is the fix) ──
    if "Big Deals" in df.columns:
        start_idx = df.columns.get_loc("Big Deals")
        plid_col = df.columns[start_idx - 1] if start_idx > 0 else df.columns[1]
    else:
        plid_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    df = df.rename(columns={plid_col: "PLID"})

    # Continue with melting (now using clean "PLID")
    start = df.columns.get_loc("Big Deals")
    try:
        end = df.columns.get_loc("Unnamed: 17")
    except KeyError:
        end = len(df.columns) - 1

    q_cols = df.columns[start:end+1]
    q_names = df.iloc[0, start:end+1].tolist()
    rename_map = dict(zip(q_cols, q_names))

    clean = df.iloc[1:, [df.columns.get_loc("PLID")] + list(range(start, end+1))].copy()
    clean = clean.rename(columns=rename_map)

    long = clean.melt(
        id_vars=["PLID"],
        var_name="Quarter",
        value_name="BigDeal_units"
    )

    long["BigDeal_units"] = pd.to_numeric(long["BigDeal_units"], errors="coerce").fillna(0)
    long = long.dropna(subset=["PLID"])
    long["Quarter"] = long["Quarter"].apply(parse_quarter)
    return long.sort_values(["PLID", "Quarter"])


# ==================== STANDARDIZATION ====================
def standardize_columns(bookings, scms, vms, bigdeal):
    """Rename PLID (Masked) → PLID everywhere for consistency"""
    for df in [bookings, scms, vms, bigdeal]:
        if "PLID (Masked)" in df.columns:
            df.rename(columns={"PLID (Masked)": "PLID"}, inplace=True)
    return bookings, scms, vms, bigdeal


def load_all_data(file_path):
    """Main function called from app.py"""
    b = load_bookings(file_path)
    s = load_scms(file_path)
    v = load_vms(file_path)
    bd = load_bigdeal(file_path)

    # ←←← CRITICAL: Standardize BEFORE merging
    b, s, v, bd = standardize_columns(b, s, v, bd)

    # Now merge using clean "PLID" column
    df = b.merge(s, on=["PLID", "Quarter"], how="left")
    df = df.merge(v, on=["PLID", "Quarter"], how="left")
    df = df.merge(bd, on=["PLID", "Quarter"], how="left")
    df.fillna(0, inplace=True)

    return df
# ──────────────────────────────────────────────────────────────
# COMPATIBILITY LAYER FOR YOUR app.py
# ──────────────────────────────────────────────────────────────
def load_all_sheets(file_path):
    """Returns 4 dataframes exactly as your app.py expects"""
    b = load_bookings(file_path)
    s = load_scms(file_path)
    v = load_vms(file_path)
    bd = load_bigdeal(file_path)

    # Standardize PLID name + return
    b, s, v, bd = standardize_columns(b, s, v, bd)
    return b, s, v, bd  