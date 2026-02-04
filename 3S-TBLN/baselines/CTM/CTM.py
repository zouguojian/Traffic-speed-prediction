# -*- coding: utf-8 -*-
"""
CTM parameter calibration + 48-step rolling prediction (merged, logic-preserving)

Important
- This script keeps the ORIGINAL computation logic unchanged.
- Only formatting and ENGLISH comments/log messages were improved for clarity.
- Workflow:
  1) Calibrate CTM parameters per segment from the first 70% of data (training split)
  2) Run 48-step rolling (recursive) prediction on the last 20% of data (test split)
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =============================================================================
# 1) CTM parameter calibration (statistical, non-iterative)
# =============================================================================

def get_statistical_params(density, velocity):
    """
    Estimate CTM/Fundamental-Diagram parameters from data distribution (no regression).

    Inputs
    - density: 1D array-like, segment density samples
    - velocity: 1D array-like, segment speed samples

    Outputs (rounded)
    - V_f   : free-flow speed (95th percentile of observed speed)
    - rho_c : critical density (here set to rho_max for simplicity under this statistical scheme)
    - w     : congestion wave speed (geometric back-calculation)
    - rho_j : jam density (here set to rho_max; may be adjusted if needed for feasibility)
    - q_max : capacity (95th percentile of observed flow q=rho*v)

    Notes
    - This is a robust, distribution-based calibration (percentiles + geometric inference).
    - It does NOT involve iterative training/optimization.
    """
    # Convert to 1D arrays
    rho = np.array(density).reshape(-1)
    v = np.array(velocity).reshape(-1)

    # Flow samples: q = rho * v
    q = rho * v

    # Keep only finite and positive samples
    mask = np.isfinite(rho) & np.isfinite(v) & (rho > 0) & (v > 0)
    rho, v, q = rho[mask], v[mask], q[mask]

    # If too few valid samples, return default parameters
    if len(rho) < 5:
        return 60.0, 150.0, 15.0, 150.0, 2000.0

    # (A) Free-flow speed: 95th percentile of observed speed
    V_f = np.percentile(v, 95)

    # (B) Critical/jam density: use maximum observed density as physical boundary
    rho_max = np.max(rho)
    rho_c = rho_max
    rho_j = rho_max

    # (C) Capacity: 95th percentile of observed flow
    q_max = np.percentile(q, 95)

    # (D) Congestion wave speed w (triangular FD geometric inference)
    # Critical density corresponding to peak flow under free-flow branch: k_crit = q_max / V_f
    k_crit = q_max / V_f

    # If feasible geometry (rho_j > k_crit), infer w = q_max / (rho_j - k_crit)
    if rho_j > k_crit + 1e-3:
        w = q_max / (rho_j - k_crit)
    else:
        # Fallback if data are degenerate: set w to an empirical fraction of V_f
        w = V_f / 3.0
        # Adjust rho_j to satisfy q_max = w * (rho_j - k_crit)
        rho_j = k_crit + q_max / w
        rho_c = rho_j

    # (E) Physical plausibility: usually w should not exceed V_f
    if w > V_f:
        w = V_f

    return round(V_f, 3), round(rho_c, 3), round(w, 3), round(rho_j, 3), round(q_max, 3)


def calibrate_from_csv_direct(csv_file, output_csv, train_ratio=0.7):
    """
    Calibrate CTM parameters for each segment from a CSV file.

    Procedure
    - Read full dataset
    - Sort by time and SegmentID (to ensure temporal consistency)
    - For each segment:
        - Split by time index (first train_ratio portion as training)
        - Estimate parameters via get_statistical_params()
    - Save per-segment parameters to output_csv
    """
    print(f"[Calibration] Reading input data: {csv_file}")

    if not os.path.exists(csv_file):
        print(f"[Calibration][Error] File does not exist: {csv_file}")
        return

    # Load CSV
    df = pd.read_csv(csv_file)

    # Ensure sorted by time if date/hour/minute columns exist
    if 'date' in df.columns:
        df = df.sort_values(by=['date', 'hour', 'minute', 'SegmentID'])

    # Unique segment list (fixed order)
    segments = df['SegmentID'].unique()
    segments.sort()

    results = []

    print(f"[Calibration] Calibrating parameters for {len(segments)} segments "
          f"(using first {train_ratio * 100:.1f}% of data)...")

    for i, sid in enumerate(segments):
        # Extract all rows for this segment
        seg_df = df[df['SegmentID'] == sid]

        # Time-based split (first train_ratio as training)
        n_total = len(seg_df)
        n_train = int(n_total * train_ratio)

        # Skip if too few samples
        if n_train < 10:
            continue

        train_df = seg_df.iloc[:n_train]

        rho_train = train_df['density'].values
        v_train = train_df['speed'].values

        # Statistical parameter estimation (logic unchanged)
        V_f, rho_c, w, rho_j, q_max = get_statistical_params(rho_train, v_train)

        results.append({
            'SegmentID': sid,
            'V_f': V_f,
            'rho_c': rho_c,
            'w': w,
            'rho_j': rho_j,
            'q_max': q_max
        })

        # Progress logging
        if (i + 1) % 100 == 0:
            print(f"[Calibration] Progress: {i + 1}/{len(segments)} segments processed...")

    # Save parameter table
    print("[Calibration] Saving calibrated parameter table to CSV...")
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv, index=False)

    print(f"[Calibration] Done. Parameters saved to: {output_csv}")
    print(f"[Calibration] Total calibrated segments: {len(df_res)}")


# =============================================================================
# 2) 48-step rolling prediction (network-level CTM-like propagation)
# =============================================================================

def run_rolling_prediction_direct_csv(
    traffic_csv_file,
    history_npz_file,
    seg_info_file,
    adj_file,
    param_file
):
    """
    Run rolling (recursive) multi-step prediction on the test portion of the data.

    High-level steps
    A) Load network segment attributes, adjacency (topology), and calibrated parameters
    B) Load full traffic CSV and pivot into matrices:
       - density matrix: [time, segment]
       - speed matrix:   [time, segment]
    C) Split test set as the last 20% of time steps
    D) Build weekday/time-slot index for lookup into historical inflow profile
    E) Load historical inflow profiles (7 x 96 x num_segments) and build entry inflow table
    F) Build downstream connectivity map and vectorized parameter arrays
    G) Rolling prediction:
       - For each test start time t_start:
         - Initialize rho_curr from the true density at t_start
         - Recursively predict 48 steps forward
         - Convert predicted density to speed and evaluate against true speed
    """
    print("[Prediction] Loading data and network topology...")

    # -----------------------------
    # (A) Load network structure and parameters
    # -----------------------------
    df_segs = pd.read_csv(seg_info_file)
    df_adj = pd.read_csv(adj_file)
    df_params = pd.read_csv(param_file)

    # Segment properties dict: SegmentID -> {in_station, out_station, distance}
    seg_props = df_segs.set_index('SegmentID')[['in_station', 'out_station', 'distance']].to_dict('index')

    # -----------------------------
    # (B) Load traffic CSV and pivot into matrices
    # -----------------------------
    print(f"[Prediction] Reading traffic CSV: {traffic_csv_file}")
    df_traffic = pd.read_csv(traffic_csv_file)

    # Ensure temporal ordering
    if 'date' in df_traffic.columns:
        df_traffic['datetime'] = pd.to_datetime(df_traffic['date'])
        df_traffic = df_traffic.sort_values(by=['date', 'hour', 'minute', 'SegmentID'])

    # Fixed segment ordering for matrix columns
    segments = df_traffic['SegmentID'].unique()
    segments.sort()
    num_segments = len(segments)

    # Unique time index
    unique_times = df_traffic[['date', 'hour', 'minute', 'datetime']].drop_duplicates().sort_values(
        ['date', 'hour', 'minute']
    )
    total_timesteps = len(unique_times)

    print(f"[Prediction] Total time steps: {total_timesteps}")
    print(f"[Prediction] Total segments: {num_segments}")

    # Pivot into [time, segment] matrices
    print("[Prediction] Building pivot matrices (density/speed)...")
    mat_density_full = df_traffic.pivot_table(
        index=['date', 'hour', 'minute'],
        columns='SegmentID',
        values='density'
    ).values
    mat_speed_full = df_traffic.pivot_table(
        index=['date', 'hour', 'minute'],
        columns='SegmentID',
        values='speed'
    ).values

    # Replace NaN with 0 to avoid numerical issues (logic unchanged)
    mat_density_full = np.nan_to_num(mat_density_full, nan=0.0)
    mat_speed_full = np.nan_to_num(mat_speed_full, nan=0.0)

    # -----------------------------
    # (C) Test split: last 20% of time steps
    # -----------------------------
    start_test_idx = int(total_timesteps * 0.8)
    print(f"[Prediction] Test split starts at index {start_test_idx} (last 20% of time steps)")

    mat_density_true = mat_density_full[start_test_idx:, :]
    mat_speed_true = mat_speed_full[start_test_idx:, :]

    test_times = unique_times.iloc[start_test_idx:].copy()
    num_timesteps_test = mat_density_true.shape[0]

    # Map SegmentID -> column index in matrices
    seg_to_col = {sid: i for i, sid in enumerate(segments)}

    # -----------------------------
    # (D) Time lookup index: weekday (0-6) and slot (0-95)
    # -----------------------------
    print("[Prediction] Building time index map (weekday, 15-min slot)...")
    test_days = test_times['datetime'].dt.dayofweek.values  # Monday=0 ... Sunday=6
    test_hours = test_times['hour'].values
    test_minutes = test_times['minute'].values
    test_slots = (test_hours * 4 + (test_minutes / 15)).astype(int)  # 15-min slots

    # Each row: [day_index, slot_index]
    time_index_map = np.column_stack((test_days, test_slots))

    # -----------------------------
    # (E) Load historical inflow profile and precompute entry inflow allocation
    # -----------------------------
    print(f"[Prediction] Loading historical inflow profile: {history_npz_file}")
    hist_data = np.load(history_npz_file)
    hist_flow_raw = hist_data['historical_flow']  # shape: [7, 96, NumSegments]

    print("[Prediction] Precomputing entrance inflow allocation table...")
    entry_inflow_table = np.zeros_like(hist_flow_raw)

    # station_to_segs maps entrance station_id -> list of segment column indices
    station_to_segs = {}
    for i, sid in enumerate(segments):
        if sid not in seg_props:
            continue
        in_st = seg_props[sid]['in_station']
        # Entrance station IDs are assumed in [0, 12]
        if 0 <= in_st <= 12:
            station_to_segs.setdefault(in_st, []).append(i)

    # For segments sharing the same entrance station, distribute total inflow equally
    for st_id, seg_indices in station_to_segs.items():
        count = len(seg_indices)
        if count == 0:
            continue

        # Sum over segments connected to the same entrance, then average
        sum_flow = np.sum(hist_flow_raw[:, :, seg_indices], axis=2)   # [7, 96]
        avg_flow = sum_flow / count

        # Assign the averaged inflow to each connected segment
        for idx in seg_indices:
            entry_inflow_table[:, :, idx] = avg_flow

    # -----------------------------
    # (F) Build topology (downstream map) and vectorize parameters
    # -----------------------------
    downstream_map = {sid: [] for sid in segments}

    # Add internal segment-to-segment downstream connections from adjacency file
    for _, row in df_adj.iterrows():
        u, v = row['src_FID'], row['nbr_FID']
        if u in downstream_map and v in seg_to_col:
            downstream_map[u].append(v)

    # Add exit connections (encoded as negative station IDs)
    for sid in segments:
        if sid not in seg_props:
            continue
        out_st = seg_props[sid]['out_station']
        # Exit station IDs are assumed in [13, 25]
        if 13 <= out_st <= 25:
            downstream_map[sid].append(-out_st)

    params = df_params.set_index('SegmentID').to_dict('index')

    # Time step in hours (15 minutes)
    dt_hours = 15 / 60.0

    # Vectorized arrays (one entry per segment column)
    factor_array = np.zeros(num_segments)  # factor = dt / dx (dx derived from segment length)
    Vf_arr = np.zeros(num_segments)
    rhoc_arr = np.zeros(num_segments)
    w_arr = np.zeros(num_segments)
    rhoj_arr = np.zeros(num_segments)
    qmax_arr = np.zeros(num_segments)

    for i, sid in enumerate(segments):
        # If missing properties or parameters, fall back to defaults
        if sid not in seg_props or sid not in params:
            factor_array[i] = 0.25
            Vf_arr[i] = 60
            rhoc_arr[i] = 200
            w_arr[i] = 15
            rhoj_arr[i] = 200
            qmax_arr[i] = 2000
            continue

        # dx (segment length) in km; factor = dt_hours / dx_km
        L_km = seg_props[sid]['distance'] / 1000.0
        factor_array[i] = dt_hours / L_km

        # Load calibrated parameters
        p = params[sid]
        Vf_arr[i] = p['V_f']
        rhoc_arr[i] = p['rho_c']
        w_arr[i] = p['w']
        rhoj_arr[i] = p['rho_j']
        qmax_arr[i] = p['q_max']

    # -----------------------------
    # (G) Rolling prediction and evaluation
    # -----------------------------
    HORIZON = 48
    end_start_idx = num_timesteps_test - HORIZON

    if end_start_idx <= 0:
        print("[Prediction][Error] Test set is too short for the requested horizon.")
        return

    print("[Prediction] Starting rolling prediction (with entrance inflow)...")
    print(f"[Prediction] Total rolling windows: {end_start_idx}")

    # Per-step accumulators (sum over all windows and segments)
    metric_accumulator = {
        step: {'sum_ae': 0.0, 'sum_se': 0.0, 'sum_ape': 0.0, 'count': 0}
        for step in range(1, HORIZON + 1)
    }

    # Overall accumulators
    global_ae = 0.0
    global_se = 0.0
    global_ape = 0.0
    global_count = 0

    start_time = time.time()

    for t_start in range(end_start_idx):
        if t_start % 100 == 0 and t_start > 0:
            print(f"[Prediction] Processing window {t_start}/{end_start_idx}...")

        # Initialize current density from ground truth at window start
        rho_curr = mat_density_true[t_start, :].copy()

        # Recursive multi-step prediction (48 steps)
        for step in range(1, HORIZON + 1):
            t_future_idx = t_start + step

            # Lookup weekday + time-slot index for historical inflow
            if t_future_idx < len(time_index_map):
                d_idx, s_idx = time_index_map[t_future_idx]
            else:
                d_idx, s_idx = time_index_map[-1]

            # 1) Supply and demand under triangular FD:
            #    Supply S = min(Vf * rho, qmax)
            #    Demand R = min(w * (rhoj - rho), qmax)
            S = np.minimum(Vf_arr * rho_curr, qmax_arr)
            R = np.minimum(w_arr * (rhoj_arr - rho_curr), qmax_arr)

            # 2) Flux calculation: Fin/Fout per segment
            Fin = np.zeros(num_segments)
            Fout = np.zeros(num_segments)

            # 2.1) Boundary inflow: use historical entrance inflow profile (scaled by 4.0)
            Fin += (entry_inflow_table[d_idx, s_idx, :] * 4.0)

            # 2.2) Internal transfers based on topology and equal split assumption
            for i in range(num_segments):
                sid = segments[i]
                branches = downstream_map.get(sid, [])
                n_branches = len(branches)
                supply = S[i]

                # If no downstream branches, treat as leaving the network
                if n_branches == 0:
                    Fout[i] += supply
                else:
                    # Equal split among downstream branches
                    potential = supply / n_branches
                    actual_out = 0.0

                    for dst_id in branches:
                        # If dst_id < 0 => exit; assume unlimited downstream demand
                        if dst_id < 0:
                            demand = 99999.0
                        else:
                            # Downstream demand from segment demand R
                            dst_idx = seg_to_col.get(dst_id)
                            demand = R[dst_idx] if dst_idx is not None else 99999.0

                        # Actual flow is constrained by split supply and downstream demand
                        flow = min(potential, demand)
                        actual_out += flow

                        # If downstream is a real segment, add to its inflow
                        if dst_id > 0:
                            Fin[seg_to_col[dst_id]] += flow

                    # Total outflow from this segment
                    Fout[i] += actual_out

            # 3) Density update: rho_next = rho_curr + factor * (Fin - Fout)
            d_rho = factor_array * (Fin - Fout)
            rho_next = rho_curr + d_rho

            # Clamp density into [0, rho_j]
            rho_next = np.clip(rho_next, 0, rhoj_arr)

            # 4) Speed prediction via Greenshields-like relationship:
            #    v = Vf * (1 - rho / rho_c)
            v_pred = Vf_arr * (1 - rho_next / rhoc_arr)
            v_pred = np.maximum(v_pred, 0)

            # Ground-truth speed at the same future time step
            v_true = mat_speed_true[t_future_idx, :]

            # Error terms (vectorized over segments)
            diff = v_pred - v_true
            abs_diff = np.abs(diff)
            sq_diff = diff ** 2
            ape = abs_diff / (np.abs(v_true) + 1e-6)

            # Accumulate metrics for this horizon step
            metric_accumulator[step]['sum_ae'] += np.sum(abs_diff)
            metric_accumulator[step]['sum_se'] += np.sum(sq_diff)
            metric_accumulator[step]['sum_ape'] += np.sum(ape)
            metric_accumulator[step]['count'] += num_segments

            # Accumulate overall metrics
            global_ae += np.sum(abs_diff)
            global_se += np.sum(sq_diff)
            global_ape += np.sum(ape)
            global_count += num_segments

            # Roll forward: next step uses the predicted density
            rho_curr = rho_next

    # -----------------------------
    # Report results
    # -----------------------------
    elapsed = time.time() - start_time
    print(f"\n[Prediction] Finished. Elapsed time: {elapsed:.2f}s")

    print("\n=== Average errors by horizon step ===")
    print(f"{'Step':<6} | {'MAE':<10} | {'RMSE':<10} | {'MAPE(%)':<10}")
    print("-" * 46)

    step_results = []
    for step in range(1, HORIZON + 1):
        m = metric_accumulator[step]
        count = m['count']
        if count == 0:
            continue

        mae = m['sum_ae'] / count
        mse = m['sum_se'] / count
        rmse = np.sqrt(mse)
        mape = (m['sum_ape'] / count) * 100

        print(f"{step:<6} | {mae:<10.4f} | {rmse:<10.4f} | {mape:<10.4f}")
        step_results.append({'Step': step, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})

    # Save per-step evaluation
    pd.DataFrame(step_results).to_csv('evaluation_by_step.csv', index=False)

    print("\n=== Overall average errors (all windows & all steps) ===")
    if global_count > 0:
        ov_mae = global_ae / global_count
        ov_rmse = np.sqrt(global_se / global_count)
        ov_mape = (global_ape / global_count) * 100

        print(f"Overall MAE : {ov_mae:.4f}")
        print(f"Overall RMSE: {ov_rmse:.4f}")
        print(f"Overall MAPE: {ov_mape:.4f}%")

        with open('evaluation_overall.txt', 'w') as f:
            f.write(f"Overall MAE: {ov_mae}\n")
            f.write(f"Overall RMSE: {ov_rmse}\n")
            f.write(f"Overall MAPE: {ov_mape}\n")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    # ============================
    # Configuration (relative paths)
    # All required files are assumed to be under ./data/
    # ============================

    # Calibration-related paths
    csv_path = os.path.join('data', 'road_segment_states_15min.csv')
    output_csv_path = os.path.join('data', 'calibrated_params.csv')

    # Prediction-related paths
    traffic_csv = os.path.join('data', 'road_segment_states_15min.csv')
    history_npz = os.path.join('data', 'historical_inflow_profile.npz')
    seg_csv = os.path.join('data', 'road_segments_with_distance.csv')
    adj_csv = os.path.join('data', 'adjacent.csv')
    param_csv = output_csv_path  # use the calibrated parameters generated above

    # ============================
    # Step 1) Parameter calibration
    # ============================
    calibrate_from_csv_direct(csv_path, output_csv_path, train_ratio=0.7)

    # ============================
    # Step 2) Rolling CTM prediction
    # ============================
    run_rolling_prediction_direct_csv(
        traffic_csv,
        history_npz,
        seg_csv,
        adj_csv,
        param_csv
    )

