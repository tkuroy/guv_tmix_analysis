import re
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats


# ============================================================
# SETTINGS
# ============================================================

tmixguess = 40


# ============================================================
# SELECT INPUT FOLDER
# ============================================================

root = Tk()
root.withdraw()
folder_selected = filedialog.askdirectory(
    title="Select folder containing CSV files"
)
root.destroy()

if not folder_selected:
    raise RuntimeError("No folder was selected.")

folder_path = Path(folder_selected).resolve()


# ============================================================
# READ DATE AND DELTA C FROM FILENAMES
# ============================================================

def extract_date(filename):
    """Read a leading MMDD code, e.g. 0803 -> (8, 3)."""
    match = re.match(r"^(\d{4})", filename)
    if not match:
        return None, None

    code = match.group(1)
    return int(code[:2]), int(code[2:])


def extract_diff(filename):
    """Read diff0mM, diff20mM, diff-12.5mM, etc."""
    match = re.search(
        r"diff(-?\d+(?:\.\d+)?)mM",
        filename,
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else np.nan


def date_label(month, day):
    return f"{month:02d}/{day:02d}/2026"


# ============================================================
# SAME SIGMOID AND FITTING METHOD AS THE PROVIDED SCRIPT
# ============================================================

def sigmoid(x, x0, w, y0):
    return y0 * (1 - 1 / (1 + np.exp(-(x - x0) / w)))


def fit_one_csv(filepath, month, day, delta_c):
    """Fit one CSV and return its date, concentration, Tmix, and width."""
    df = pd.read_csv(filepath, usecols=["T", "Mixed", "Sep"])
    df = df.astype({"T": float, "Mixed": float, "Sep": float})

    denominator = df["Mixed"] + df["Sep"]
    df["fraction"] = df["Sep"] * 100 / denominator
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["T", "fraction"])

    x = df["T"].to_numpy()
    y = df["fraction"].to_numpy()

    if len(x) < 4:
        raise ValueError("Fewer than four valid temperature points")

    popt, pcov = curve_fit(
        sigmoid,
        x,
        y,
        p0=[tmixguess, 0.5, 100],
        bounds=([-np.inf, 0, 0], [np.inf, np.inf, 100]),
    )

    tmix = float(popt[0])
    width_w = float(popt[1])
    dof = len(x) - len(popt)
    tmix_se = float(np.sqrt(pcov[0, 0]))
    tmix_95ci_halfwidth = float(stats.t.ppf(0.975, dof) * tmix_se)

    if (
        not np.isfinite(tmix)
        or not np.isfinite(width_w)
        or width_w <= 0
        or not np.isfinite(tmix_95ci_halfwidth)
    ):
        raise ValueError("Fit returned a non-finite Tmix/CI or invalid width")

    return {
        "Month": month,
        "Day": day,
        "Date": date_label(month, day),
        "Delta_c_mM": delta_c,
        "Tmix_C": tmix,
        "Width_w_C": width_w,
        "Tmix_SE_C": tmix_se,
        "Tmix_95CI_halfwidth_C": tmix_95ci_halfwidth,
        "Fit_dof": dof,
        "Filename": filepath.name,
    }


# ============================================================
# FIT FILES; SKIP FILES WITHOUT DATE OR DELTA C
# ============================================================

csv_files = sorted(
    [
        path
        for path in folder_path.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and not path.stem.endswith(("_std_w", "_std_95%", "_95%"))
    ],
    key=lambda path: path.name,
)

if not csv_files:
    raise RuntimeError(f"No input CSV files were found in: {folder_path}")

fit_records = []

for filepath in csv_files:
    month, day = extract_date(filepath.name)
    delta_c = extract_diff(filepath.name)

    if month is None or day is None or not np.isfinite(delta_c):
        print(f"Skipped (date or Δc not found): {filepath.name}")
        continue

    try:
        record = fit_one_csv(filepath, month, day, delta_c)
        fit_records.append(record)
        print(
            f"Fitted: {filepath.name}  "
            f"Tmix = {record['Tmix_C']:.4f} °C, "
            f"w = {record['Width_w_C']:.4f} °C"
        )
    except (ValueError, RuntimeError, KeyError, OSError, pd.errors.ParserError) as error:
        print(f"FIT FAILED: {filepath.name}\n  {error}")

if not fit_records:
    raise RuntimeError("No CSV files could be fitted.")

fitted = pd.DataFrame(fit_records).sort_values(
    ["Month", "Day", "Delta_c_mM", "Filename"]
)


# ============================================================
# CALCULATE DAILY DELTA TMIX AND COMBINED 95% CI HALF-WIDTH
# ============================================================

def mean_95ci_halfwidth(records):
    """Approximate 95% CI half-width for a mean of independent fit estimates."""
    standard_errors = records["Tmix_SE_C"].to_numpy(dtype=float)
    dofs = records["Fit_dof"].to_numpy(dtype=float)
    count = len(standard_errors)

    # Standard error of the mean of independent estimates.
    component_variances = standard_errors**2 / count**2
    se_mean = float(np.sqrt(component_variances.sum()))

    # Welch-Satterthwaite effective degrees of freedom; for one fit this
    # reduces to that fit's residual degrees of freedom.
    dof_effective = float(
        component_variances.sum() ** 2
        / np.sum(component_variances**2 / dofs)
    )
    return float(stats.t.ppf(0.975, dof_effective) * se_mean)

output_records = []

for (month, day), day_data in fitted.groupby(["Month", "Day"], sort=True):
    baseline = day_data[np.isclose(day_data["Delta_c_mM"], 0)]
    date = date_label(month, day)

    if baseline.empty:
        print(f"Skipped date {date}: no diff0mM baseline fit.")
        continue

    # If a date has multiple diff0mM files, use the mean fit estimate. Its
    # confidence half-width is calculated from the SEs of those independent fits.
    tmix_0 = float(baseline["Tmix_C"].mean())
    ci_halfwidth_0 = mean_95ci_halfwidth(baseline)

    for _, row in day_data.iterrows():
        tmix_at_delta_c = float(row["Tmix_C"])
        ci_halfwidth_at_delta_c = float(row["Tmix_95CI_halfwidth_C"])

        if np.isclose(float(row["Delta_c_mM"]), 0):
            # The baseline is compared with itself, so show zero shift and
            # use the baseline CI half-width directly.
            tmix_at_delta_c = tmix_0
            ci_halfwidth_at_delta_c = ci_halfwidth_0
            delta_tmix = 0.0
            combined_ci_halfwidth = ci_halfwidth_0
        else:
            # Combine the two 95% CI half-widths in quadrature as requested.
            delta_tmix = tmix_at_delta_c - tmix_0
            combined_ci_halfwidth = float(
                np.sqrt(ci_halfwidth_0**2 + ci_halfwidth_at_delta_c**2)
            )

        output_records.append(
            {
                "Delta_c_mM": float(row["Delta_c_mM"]),
                "Date": date,
                "Tmix_at_Delta_c_0_C": tmix_0,
                "Tmix_95CI_halfwidth_at_Delta_c_0_C": ci_halfwidth_0,
                "Tmix_at_Delta_c_C": tmix_at_delta_c,
                "Tmix_95CI_halfwidth_at_Delta_c_C": ci_halfwidth_at_delta_c,
                "Delta_Tmix_C": delta_tmix,
                "Combined_Tmix_95CI_halfwidth_C": combined_ci_halfwidth,
            }
        )

if not output_records:
    raise RuntimeError(
        "No date had both a valid fit and a diff0mM baseline fit."
    )

output = pd.DataFrame(output_records).sort_values(["Date", "Delta_c_mM"])

# One summary CSV for the selected folder, named after that folder.
output_path = folder_path / f"{folder_path.name}_95%.csv"
output.to_csv(output_path, index=False)

print()
print(f"Saved summary CSV: {output_path}")
