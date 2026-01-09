import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def excess_kurtosis(x):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    m4 = np.mean((x - mean) ** 4)
    return (m4 / (std ** 4)) - 3.0


def _max_cross_corr_lag(x, y, max_lag):
    best_corr = -1.0
    best_lag = 0
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return best_lag, best_corr
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xs = x[:lag]
            ys = y[-lag:]
        elif lag > 0:
            xs = x[lag:]
            ys = y[:-lag]
        else:
            xs = x
            ys = y
        if xs.size < 4:
            continue
        corr = np.mean(xs * ys)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag, best_corr


def detect_segments(series, window, stride, thresholds, target_series=None):
    segments = []
    n = len(series)
    for start in range(0, n - window + 1, stride):
        seg = series[start:start + window]
        seg_std = np.std(seg)
        seg_mean = np.mean(seg)
        if seg_std == 0:
            continue

        # Heavy tails
        ek = excess_kurtosis(seg)
        if ek >= thresholds["heavy_tail_excess_kurtosis"]:
            segments.append(("heavy_tail", start, start + window, ek))

        # Heteroscedasticity: rolling std variability
        roll_win = max(4, window // 8)
        roll_stds = pd.Series(seg).rolling(roll_win).std().dropna().values
        if roll_stds.size > 0:
            cv = np.std(roll_stds) / (np.mean(roll_stds) + 1e-8)
            if cv >= thresholds["hetero_cv"]:
                segments.append(("heteroscedastic", start, start + window, cv))

        # Phase shift: high correlation with target at non-zero lag
        if target_series is not None:
            target_seg = target_series[start:start + window]
            max_lag = max(1, window // 4)
            lag, corr = _max_cross_corr_lag(target_seg, seg, max_lag)
            if abs(lag) >= thresholds["phase_min_lag"] and corr >= thresholds["phase_corr"]:
                segments.append(("phase_shift", start, start + window, corr))

        # Transient burst: max z-score in segment
        z = (seg - seg_mean) / (seg_std + 1e-8)
        max_z = np.max(np.abs(z))
        if max_z >= thresholds["burst_z"]:
            segments.append(("burst", start, start + window, max_z))

    return segments


def analyze_dataset(root_path, output_dir, window, stride, max_plots_per_type, thresholds):
    datasets = {
        "Energy": "Time-MMD/numerical/Energy/Energy.parquet",
        "Climate": "Time-MMD/numerical/Climate/Climate.parquet",
        "Health_US": "Time-MMD/numerical/Health_US/Health_US.parquet",
        "Environment": "Time-MMD/numerical/Environment/Environment.parquet",
    }
    target_series_map = {
        "Energy": "Gasoline Prices",
        "Climate": "Drought Level",
        "Health_US": "Influenza Patients Proportion",
        "Environment": "Air Quality Index",
    }

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    for name, rel_path in datasets.items():
        data_path = os.path.join(root_path, rel_path)
        df = pd.read_parquet(data_path)
        time_col = df.columns[0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)

        target_col = target_series_map[name]
        if target_col not in df.columns:
            raise ValueError(f"Target series '{target_col}' not found in {data_path}")
        target_series = df[target_col].to_numpy()

        for col in numeric_cols:
            if col == target_col:
                continue
            series = df[col].to_numpy()
            segments = detect_segments(
                series,
                window,
                stride,
                thresholds,
                target_series=target_series,
            )

            type_counts = {}
            for seg_type, start, end, score in segments:
                type_counts[seg_type] = type_counts.get(seg_type, 0) + 1
                summary_rows.append([name, col, seg_type, start, end, score])

            for seg_type in ["heavy_tail", "heteroscedastic", "phase_shift", "burst"]:
                candidates = [s for s in segments if s[0] == seg_type]
                candidates.sort(key=lambda x: x[3], reverse=True)
                for _, start, end, score in candidates[:max_plots_per_type]:
                    seg = series[start:end]
                    plt.figure(figsize=(10, 4))
                    if seg_type == "phase_shift":
                        tseg = target_series[start:end]
                        plt.plot(tseg, linewidth=1.2, label=target_col)
                        plt.plot(seg, linewidth=1.5, label=col)
                        plt.legend()
                        title = f"{name} | {col} vs {target_col} | {seg_type} | corr={score:.3f}"
                    else:
                        plt.plot(seg, linewidth=1.5)
                        title = f"{name} | {col} | {seg_type} | score={score:.3f}"
                    plt.title(title)
                    plt.xlabel("time")
                    plt.ylabel(col)
                    plt.tight_layout()
                    filename = f"{name}_{col}_{seg_type}_{start}_{end}.pdf"
                    plt.savefig(os.path.join(output_dir, filename), format="pdf", bbox_inches="tight")
                    plt.close()

            if segments:
                counts = ", ".join([f"{k}={v}" for k, v in type_counts.items()])
                print(f"{name} | {col}: {counts}")

    if summary_rows:
        summary_df = pd.DataFrame(
            summary_rows,
            columns=["dataset", "series", "type", "start_idx", "end_idx", "score"],
        )
        summary_df.to_csv(os.path.join(output_dir, "anomaly_segments.csv"), index=False)
    else:
        print("No segments matched thresholds.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Time-MMD numerical datasets for heavy tails/heteroscedasticity/phase shifts/bursts"
    )
    parser.add_argument("--root_path", type=str, default="./dataset", help="Root path of dataset")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", help="Output directory")
    parser.add_argument("--window", type=int, default=128, help="Window size")
    parser.add_argument("--stride", type=int, default=16, help="Stride size")
    parser.add_argument("--max_plots", type=int, default=5, help="Max plots per type per series")
    parser.add_argument("--heavy_tail_kurtosis", type=float, default=1.5, help="Excess kurtosis threshold")
    parser.add_argument("--hetero_cv", type=float, default=0.6, help="Rolling std CV threshold")
    parser.add_argument("--phase_corr", type=float, default=0.6, help="Phase-shift min correlation")
    parser.add_argument("--phase_min_lag", type=int, default=3, help="Phase-shift min lag (samples)")
    parser.add_argument("--burst_z", type=float, default=4.0, help="Burst z-score threshold")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"timemmd_diagnostics_{timestamp}")

    thresholds = {
        "heavy_tail_excess_kurtosis": args.heavy_tail_kurtosis,
        "hetero_cv": args.hetero_cv,
        "phase_corr": args.phase_corr,
        "phase_min_lag": args.phase_min_lag,
        "burst_z": args.burst_z,
    }

    analyze_dataset(
        args.root_path,
        output_dir,
        window=args.window,
        stride=args.stride,
        max_plots_per_type=args.max_plots,
        thresholds=thresholds,
    )
    print(f"Diagnostics saved to: {output_dir}")


if __name__ == "__main__":
    main()
