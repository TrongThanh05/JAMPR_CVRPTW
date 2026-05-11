"""Export standalone .pt files for each N × TW combination."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

def load_and_split(pt_path, tw_mode):
    dataset = torch.load(pt_path, weights_only=False)
    tw_data = dataset[tw_mode]
    return {
        "coords": dataset["coords"],
        "demands": dataset["demands"],
        "time_windows": tw_data["time_windows"],
        "service_times": tw_data["service_times"],
        "capacity": dataset["capacity"],
        "n_vehicles": dataset["n_vehicles"],
        "tw_mode": tw_mode,
    }

if __name__ == "__main__":
    for n in [20, 50]:
        pt_path = f"data/vrptw_n{n}_fixed.pt"
        for tw in ["tw1", "tw2", "tw3"]:
            data = load_and_split(pt_path, tw)
            out = f"data/vrptw_n{n}_{tw}.pt"
            torch.save(data, out)
            size_mb = os.path.getsize(out) / (1024 * 1024)
            print(f"  {out}  ({size_mb:.1f} MB)  shape={data['coords'].shape}")
    print("Done!")
