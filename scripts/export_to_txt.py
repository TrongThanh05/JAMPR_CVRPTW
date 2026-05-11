"""Export fixed .pt data to simple human-readable text files.

Creates one .txt file per N value with ALL instances in a simple
tab-separated format that is easy to read and parse.

Usage:
    python scripts/export_to_txt.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def export_dataset(pt_path: str, txt_path: str):
    """Export a .pt dataset to a simple text file."""
    dataset = torch.load(pt_path, weights_only=False)

    coords_raw = dataset["coords_raw"]       # (B, N+1, 2)
    demands_raw = dataset["demands_raw"]     # (B, N+1)
    capacity = dataset["capacity"]
    n_vehicles = dataset["n_vehicles"]

    B = coords_raw.shape[0]
    N_plus_1 = coords_raw.shape[1]
    N = N_plus_1 - 1
    T = 1000  # time horizon

    tw1 = dataset["tw1"]["time_windows"]  # (B, N+1, 2)
    tw2 = dataset["tw2"]["time_windows"]
    tw3 = dataset["tw3"]["time_windows"]
    svc = dataset["tw1"]["service_times"]  # same for all tw modes

    with open(txt_path, "w", encoding="utf-8") as f:
        # Header
        f.write(f"# VRPTW Fixed Benchmark Data\n")
        f.write(f"# N={N}  INSTANCES={B}  Q={capacity}  K={n_vehicles}  T={T}  SERVICE=10\n")
        f.write(f"#\n")
        f.write(f"# Node 0 = Depot, Nodes 1..{N} = Customers\n")
        f.write(f"# Coordinates are in [0, 100], Time windows are in [0, {T}]\n")
        f.write(f"# TW1=narrow(100), TW2=medium(300), TW3=wide(500)\n")
        f.write(f"#\n")
        f.write(f"# INST\tNODE\tX\tY\tDEMAND\tTW1_A\tTW1_B\tTW2_A\tTW2_B\tTW3_A\tTW3_B\tSERVICE\n")

        for i in range(B):
            for j in range(N_plus_1):
                x = coords_raw[i, j, 0].item()
                y = coords_raw[i, j, 1].item()
                d = int(demands_raw[i, j].item())
                t1a = round(tw1[i, j, 0].item() * T)
                t1b = round(tw1[i, j, 1].item() * T)
                t2a = round(tw2[i, j, 0].item() * T)
                t2b = round(tw2[i, j, 1].item() * T)
                t3a = round(tw3[i, j, 0].item() * T)
                t3b = round(tw3[i, j, 1].item() * T)
                s = round(svc[i, j].item() * T)
                f.write(f"{i}\t{j}\t{x:.2f}\t{y:.2f}\t{d}\t"
                        f"{t1a}\t{t1b}\t{t2a}\t{t2b}\t{t3a}\t{t3b}\t{s}\n")

    file_size = os.path.getsize(txt_path) / (1024 * 1024)
    print(f"  Saved: {txt_path} ({file_size:.1f} MB, {B} instances x {N_plus_1} nodes)")


def main():
    data_dir = "data"

    for n in [20, 50]:
        pt_path = os.path.join(data_dir, f"vrptw_n{n}_fixed.pt")
        txt_path = os.path.join(data_dir, f"vrptw_n{n}_fixed.txt")

        if not os.path.exists(pt_path):
            print(f"  SKIP: {pt_path} not found")
            continue

        print(f"Exporting N={n}...")
        export_dataset(pt_path, txt_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
