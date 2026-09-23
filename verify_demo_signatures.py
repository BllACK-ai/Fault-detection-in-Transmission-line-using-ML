"""Print current RMS signatures for regenerated demo plots."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np


DEMO_TYPES = ("normal", "lg", "ll", "llg", "lll")
CURRENT_CHANNELS = ("Ia", "Ib", "Ic")


def _rms(values: np.ndarray) -> float:
    return float(sqrt(float(np.mean(np.square(values)))))


def main() -> None:
    demo_dir = Path("simulation_output") / "demo"
    for name in DEMO_TYPES:
        data = np.load(demo_dir / f"{name}_demo.npz", allow_pickle=True)
        metadata = json.loads(str(data["metadata_json"]))
        time = data["time"]
        pre_mask = time < float(metadata["t_start"])
        fault_mask = (time >= float(metadata["t_start"])) & (time < float(metadata["t_end"]))
        print(
            f"{name.upper()}: type={metadata['fault_type']}, "
            f"phases={metadata['faulted_phases'] or 'None'}, "
            f"Rf={metadata['fault_resistance_ohm']}"
        )
        for channel in CURRENT_CHANNELS:
            values = data[channel]
            pre_rms = _rms(values[pre_mask]) if np.any(pre_mask) else _rms(values)
            fault_rms = _rms(values[fault_mask]) if np.any(fault_mask) else None
            ratio = None if fault_rms is None else fault_rms / max(pre_rms, 1e-9)
            fault_text = "None" if fault_rms is None else f"{fault_rms:.1f}"
            ratio_text = "None" if ratio is None else f"{ratio:.1f}"
            print(f"  {channel}: pre_rms={pre_rms:.1f} A, fault_rms={fault_text} A, ratio={ratio_text}")


if __name__ == "__main__":
    main()
