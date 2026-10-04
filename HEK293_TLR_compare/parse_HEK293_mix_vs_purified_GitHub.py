"""
parse_HEK293_mix_vs_purified_GitHub.py
Converts a plate-reader Excel export (absorbance) into a tidy long table for
HEK293_mix_vs_purified_GitHub.py.

Expected sheet layout (first sheet, no header row):
  column C holds a block label such as "48H_1587_ole_mouse_620nm"
    -> time (e.g. 48H), species (mouse|human), plate (e.g. p1, optional),
       wavelength (620|655 nm)
  the same row, columns D onward, holds the well labels, e.g.
    "1587_mix", "1587_68", "pos_ctrl", "neg_ctrl"   (strain_Rf [dose])
  the rows directly below hold one replicate each (absorbance values)
  a blank row in column C ends the block.

Usage:
    python parse_HEK293_mix_vs_purified_GitHub.py your_plate_export.xlsx
Output: HEK293_mix_vs_purified_tidy.csv (+ a .txt pretty-print of the same table)
"""
import sys
import pandas as pd
import numpy as np
import re

def parse_well_header(h):
    if pd.isna(h):
        return None, None, None
    h = str(h).strip()
    if h.lower().startswith("neg"):
        return "Neg", None, None
    # "Pos_cont_1ug" style (no spaces, dose embedded after last underscore)
    if h.lower().startswith("pos"):
        parts = h.split()
        strain = "Pos_cont."
        # handle "Pos_cont_1ug" with no spaces
        if len(parts) == 1 and "_" in h:
            sub = h.split("_")[-1]
            dose = sub if "ug" in sub else None
        else:
            dose = parts[-1] if "ug" in parts[-1] else None
        return strain, None, dose
    if h.startswith("EC_sigma"):
        parts = h.split()
        strain = "Pos_cont."
        dose = parts[-1] if "ug" in parts[-1] else None
        return strain, None, dose
    parts = h.split()
    strain_rf = parts[0]
    dose = parts[-1] if "ug" in parts[-1] else None
    if "_" in strain_rf:
        strain, rf = strain_rf.split("_", 1)
        # EC Rf values are stored as integers (e.g. 464) but represent 0.464
        if strain == "EC" and rf is not None:
            try:
                rf = str(round(float(rf) / 1000, 3))
            except ValueError:
                pass
    else:
        strain, rf = strain_rf, None
    return strain, rf, dose

def parse_block_label(label):
    label = str(label)
    time_match = re.search(r"(\d+)\s*H", label)
    species_match = re.search(r"(mouse|human)", label, re.IGNORECASE)
    plate_match = re.search(r"p(\d+)", label)
    wl_match = re.search(r"(620|655)\s*nm", label)
    time_h = int(time_match.group(1)) if time_match else None
    species = species_match.group(1).lower() if species_match else None
    plate = f"p{plate_match.group(1)}" if plate_match else None
    wavelength = wl_match.group(1) if wl_match else None
    return time_h, species, plate, wavelength

def parse_cleaned_excel(path):
    df = pd.read_excel(path, header=None)
    tidy_rows = []
    n_rows = df.shape[0]
    row = 0
    while row < n_rows:
        block_label = df.iloc[row, 2]
        if pd.isna(block_label):
            row += 1
            continue
        time_h, species, plate, wavelength = parse_block_label(block_label)
        headers = df.iloc[row, 3:].dropna()
        row += 1
        rep_num = 1
        while row < n_rows and df.iloc[row, 3:].notna().any():
            replicate_values = df.iloc[row, 3:3+len(headers)].values
            for col_idx, header in enumerate(headers):
                strain, rf, dose = parse_well_header(header)
                absorb = replicate_values[col_idx]
                if pd.isna(absorb):
                    continue
                tidy_rows.append({
                    "Time": time_h,
                    "Species": species,
                    "Plate": plate,
                    "Wavelength": wavelength,
                    "Strain": strain,
                    "Rf": rf,
                    "Dose": dose,
                    "Replicate": rep_num,
                    "Absorbance": absorb
                })
            rep_num += 1
            row += 1
        while row < n_rows and pd.isna(df.iloc[row, 2]):
            row += 1
    tidy_df = pd.DataFrame(tidy_rows)
    tidy_df.to_csv("HEK293_mix_vs_purified_tidy.csv", index=False)
    with open("HEK293_mix_vs_purified_tidy.txt", "w") as f:
        f.write(tidy_df.to_string(index=False))
    print("Saved HEK293_mix_vs_purified_tidy.csv and HEK293_mix_vs_purified_tidy.txt")
    return tidy_df

if __name__ == "__main__":
    parse_cleaned_excel(sys.argv[1] if len(sys.argv) > 1 else "your_plate_export.xlsx")