"""
The goal of this script is to have you set up your plate map once, copy paste the output of the absorbance reader into an excel file, and then run this script to get a tidy long table of the data. 
The output will be a csv file that can be used for HEK293_Flex_NB.py or any other analysis script.
If you copy and paste the output of the absorbance reader into an excel it will treat the cell with the first numeric value as A1, and stop at the first blank cell
Which means that if you have a blank cell in the middle of your data it will not read the rest of the data. So make sure to fill in all cells with either a value or an "x" to ignore that well.
Also, don't label the columns and rows unless there is a # in front of it. Otherwise it will read the label as a value and throw off the data.

The script will read either a singular excel or loop through a folder of excel files. 
The loop is particularly helpful because it will also output a .csv file that combines all the data into one .csv file. It will pick up on the file name *_Exp1.xlsx, *_Exp2.xlsx, etc. and label the data accordingly.
"""
import os
import sys
import pandas as pd
import numpy as np
import re

#----------------------------------
#----------Plate_Map---------------
#----------------------------------
# Build your plate maps here. It will read either a csv or excel file
 

Names_On = "Columns"
#Names_On = "Rows"

Pool_Negative_Control = True
#Match_Negative_Control = True

#Pool_Positive_Control = True
#Pool_Positive_Control_Dose = "1 ug/ml"   # label for the pooled positive control (optional)

Match_Positive_Control = True

#Columns
Col_1 = "Positive_Control"
Col_2 = "86_Apples"
Col_3 = "86_Apples"
Col_4 = "86_Blueberries"
Col_5 = "1776_0.2"
Col_6 = "1776_0.65"
Col_7 = "Animals_Puppies"
Col_8 = "Animals_Kittens"
Col_9 = "Negative_Control"
Col_10 = "381_blue"
Col_11 = "Student"
Col_12 = "381_red"

#Rows
Row_A = "10 ug/ml"
Row_B = "10 ug/ml"
Row_C = "1 ug/ml"
Row_D = "1 ug/ml"
Row_E = "0.1 ug/ml"
Row_F = "0.1 ug/ml"
Row_G = "x"
Row_H = "x"

#Ignore these wells (will recognize lables such as A1, D8, etc.)
Ignore_Wells = ["x"]

#We all make pipetting mistakes, so if you are looping this script through a folder of excel files, it will ignore a well from a specific *_Exp#.xlsx file. 
#For example, if you pipetted the wrong sample into well A1 of your *_Exp2.xlsx file, you can add "A1_Exp2" to the list below and it will ignore that well for that specific experiment.
#The script will also ignore any well in the excel file that has an "x" in it, so you can just put an "x" in the well you want to ignore and it will be ignored for that particular experiment.
Ignore_Experiment_Wells = ["x"]

#--------------------------------
#---------Functions----------------
#--------------------------------

ROWS = "ABCDEFGH"
COLS = range(1, 13)


def is_blank(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""


def is_number(v):
    if is_blank(v):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    try:
        float(str(v).strip())
        return True
    except ValueError:
        return False


def is_label(v):
    # "#" cells are labels the user added; treat them like empty space
    return isinstance(v, str) and v.strip().startswith("#")


def read_sheet(path):
    """Return the first sheet of an Excel/CSV file as a grid with no header."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, header=None, dtype=object)
    return pd.read_excel(path, header=None, dtype=object)


def find_a1(grid):
    """First numeric cell (top to bottom, left to right), skipping # labels."""
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = grid.iat[r, c]
            if is_label(v):
                continue
            if is_number(v):
                return r, c
    raise ValueError("no numeric value found - is the plate data pasted in?")


def read_plate(path):
    """
    Read the pasted plate block starting at A1. Each row is read to the right
    until the first blank cell (max 12); rows are read down until the first
    blank row (max 8). Returns {well: value}, with None for "x" wells.
    """
    grid = read_sheet(path)
    r0, c0 = find_a1(grid)
    plate = {}
    for i, row_letter in enumerate(ROWS):
        r = r0 + i
        if r >= grid.shape[0] or is_blank(grid.iat[r, c0]) or is_label(grid.iat[r, c0]):
            break
        for j, col in enumerate(COLS):
            c = c0 + j
            if c >= grid.shape[1]:
                break
            v = grid.iat[r, c]
            if is_blank(v) or is_label(v):
                break
            well = f"{row_letter}{col}"
            if str(v).strip().lower() == "x":
                plate[well] = None
            elif is_number(v):
                plate[well] = float(v)
            else:
                raise ValueError(f"{path}: well {well} holds '{v}' - "
                                 "use a number or x")
    return plate


def build_plate_map():
    """{well: (Name, Dose)} from the Col_/Row_ settings; x wells are left out."""
    col_labels = {c: globals()[f"Col_{c}"] for c in COLS}
    row_labels = {r: globals()[f"Row_{r}"] for r in ROWS}
    plate_map = {}
    for r in ROWS:
        for c in COLS:
            if Names_On == "Columns":
                name, dose = col_labels[c], row_labels[r]
            elif Names_On == "Rows":
                name, dose = row_labels[r], col_labels[c]
            else:
                raise ValueError('Names_On must be "Columns" or "Rows"')
            if name == "x" or dose == "x":
                continue
            plate_map[f"{r}{c}"] = (name, dose)
    return plate_map


def split_dose(dose):
    """'10 ug/ml' -> (10.0, 'ug/ml'); anything without a number -> (nan, dose)."""
    m = re.match(r"\s*([0-9]*\.?[0-9]+)\s*(.*)", str(dose))
    if not m:
        return np.nan, str(dose).strip()
    return float(m.group(1)), m.group(2).strip()


CONTROLS = ("Positive_Control", "Negative_Control")


def split_name(name):
    """
    'Fruit_Apples' -> ('Fruit', 'Apples'); '1588_0.65' -> ('1588', '0.65').
    Split at the FIRST underscore only ('Desert_Ginger_Bread' -> 'Desert',
    'Ginger_Bread'). No underscore -> (name, ''). Controls are never split.
    """
    if name in CONTROLS or "_" not in name:
        return name, ""
    strain, rf = name.split("_", 1)
    return strain, rf


def experiment_label(path):
    """'plate_Exp2.xlsx' -> 'Exp2'; otherwise the file name without extension."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_(Exp\d+)$", stem, re.IGNORECASE)
    return m.group(1) if m else stem


def geomean(values):
    x = np.array(values, dtype=float)
    x = x[x > 0]
    return np.exp(np.log(x).mean()) if len(x) else np.nan


def add_fold_change(df):
    """
    Fold change = signal / geometric mean of the Negative_Control wells.
    Pool:  one baseline per plate (all Negative_Control wells).
    Match: one baseline per plate and dose.
    """
    match = globals().get("Match_Negative_Control", False) is True
    pool  = globals().get("Pool_Negative_Control", False) is True
    if match == pool:
        raise ValueError("set exactly one of Pool_Negative_Control / "
                         "Match_Negative_Control to True")
    keys = ["Experiment", "Dose"] if match else ["Experiment"]
    neg = (df[df["Name"] == "Negative_Control"]
           .groupby(keys)["Signal"].apply(geomean)
           .rename("NegControl").reset_index())
    if neg.empty:
        print("  WARNING: no Negative_Control wells - FoldChange left blank")
        df["FoldChange"] = np.nan
        return df
    df = df.merge(neg, on=keys, how="left")
    df["FoldChange"] = df["Signal"] / df["NegControl"]
    return df.drop(columns="NegControl")


def tidy_plate(path, plate_map):
    exp     = experiment_label(path)
    plate   = read_plate(path)
    skip    = {w.upper() for w in Ignore_Wells if w != "x"}
    skip_ex = {w.upper() for w in Ignore_Experiment_Wells if w != "x"}
    rows = []
    for well, (name, dose) in plate_map.items():
        value = plate.get(well)
        if value is None:                          # x in the sheet, or not pasted
            continue
        if well in skip or f"{well}_{exp}".upper() in skip_ex:
            continue
        dose_value, dose_unit = split_dose(dose)
        strain, rf = split_name(name)
        rows.append({
            "Experiment": exp,
            "File":       os.path.basename(path),
            "Well":       well,
            "Name":       name,
            "Strain":     strain,
            "Rf":         rf,
            "Dose":       dose,
            "DoseValue":  dose_value,
            "DoseUnit":   dose_unit,
            "Signal":     value,
        })
    df = pd.DataFrame(rows)
    if globals().get("Pool_Positive_Control") is True:
        # Pooled positive control: one group, whatever row/column dose it sat in
        label = globals().get("Pool_Positive_Control_Dose")
        label = label if isinstance(label, str) else "pooled"
        value, unit = split_dose(label)
        pos = df["Name"] == "Positive_Control"
        df.loc[pos, "Dose"] = label
        df.loc[pos, "DoseValue"] = value
        df.loc[pos, "DoseUnit"] = unit
    df["Replicate"] = df.groupby(["Name", "Dose"]).cumcount() + 1
    return add_fold_change(df)


#--------------------------------
#---------Run--------------------
#--------------------------------

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python Parse_Flex_NB.py <plate.xlsx | folder_of_plates>")
    target    = sys.argv[1]
    plate_map = build_plate_map()

    if os.path.isdir(target):
        files = sorted(
            os.path.join(target, f) for f in os.listdir(target)
            if f.lower().endswith((".xlsx", ".xls", ".csv"))
            and not f.startswith(("~$", "._"))
            and not f.endswith("_tidy.csv")
        )
        if not files:
            sys.exit(f"no Excel/CSV files found in {target}")
    else:
        files = [target]

    tables = []
    for path in files:
        df  = tidy_plate(path, plate_map)
        out = os.path.splitext(path)[0] + "_tidy.csv"
        df.to_csv(out, index=False)
        print(f"Saved {out}  ({len(df)} wells)")
        tables.append(df)

    if os.path.isdir(target):
        combined = pd.concat(tables, ignore_index=True)
        out = os.path.join(target, "combined_tidy.csv")
        combined.to_csv(out, index=False)
        print(f"Saved {out}  ({len(combined)} wells, {len(files)} plates)")


if __name__ == "__main__":
    main()
