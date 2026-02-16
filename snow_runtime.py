#!/usr/bin/env python3
# snow_runtime.py
#
# Lightweight snow-cover attenuation model for PV forecast.
# Uses Open-Meteo hourly variables: temperature_2m, snowfall, snow_depth, rain, irradiance.
#
# Idea:
# - Maintain a snow_index (0..~) representing effective coverage/occlusion.
# - Snowfall increases snow_index (scaled).
# - Melt (temperature above threshold), rain and sun/irradiance reduce snow_index.
# - Convert snow_index -> multiplicative factor f in [k, 1] (k = minimum fraction when fully covered).
#
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional


@dataclass(frozen=True)
class SnowParams:
    # Minimum multiplicative factor (fully covered)
    k: float = 0.85
    # Temp threshold where snow starts to accumulate (°C)
    t_acc: float = 0.0
    # Temp threshold where melt accelerates (°C)
    t_melt: float = 2.0
    # Snowfall -> snow_index increase (per mm)
    a_snowfall: float = 0.12
    # Melt rate per hour per (T - t_melt)
    b_melt: float = 0.03
    # Rain wash-off rate per mm
    b_rain: float = 0.12
    # Sun clean-off rate per kWh/m² equivalent (using irradiance Wh/m² per hour)
    b_sun: float = 0.02
    # Optional: snow depth direct effect (cm)
    a_depth: float = 0.0
    # Optional passive decay per hour
    b_passive: float = 0.0

    @staticmethod
    def from_json(path: str | Path) -> "SnowParams":
        p = Path(path)
        j = json.loads(p.read_text(encoding="utf-8"))
        # allow file to contain metrics too; ignore unknown keys
        allowed = {f.name for f in SnowParams.__dataclass_fields__.values()}
        kwargs = {k: float(v) for k, v in j.items() if k in allowed}
        return SnowParams(**kwargs)


class SnowModel:
    def __init__(self, params: SnowParams, snow_index0: float = 0.0):
        self.p = params
        self.snow_index = max(0.0, float(snow_index0))

    def update(
        self,
        temp_c: Optional[float],
        snowfall_mm: Optional[float],
        snow_depth_cm: Optional[float],
        rain_mm: Optional[float],
        irr_wm2: Optional[float],
        prev_snow_depth_cm: Optional[float] = None,
    ) -> None:
        # Accumulation from snowfall when cold-ish
        if snowfall_mm is not None and temp_c is not None:
            if temp_c <= self.p.t_acc:
                self.snow_index += self.p.a_snowfall * max(0.0, float(snowfall_mm))
        elif snowfall_mm is not None and temp_c is None:
            self.snow_index += self.p.a_snowfall * max(0.0, float(snowfall_mm))

        # Optional depth-based adjustment (can help when snowfall data is sparse)
        if self.p.a_depth and snow_depth_cm is not None:
            # use delta depth if available; otherwise absolute
            depth = max(0.0, float(snow_depth_cm))
            if prev_snow_depth_cm is not None:
                delta = depth - max(0.0, float(prev_snow_depth_cm))
                if delta > 0:
                    self.snow_index += self.p.a_depth * delta
            else:
                self.snow_index += self.p.a_depth * depth

        # Melt when warm
        if temp_c is not None:
            t = float(temp_c)
            if t > self.p.t_melt:
                self.snow_index -= self.p.b_melt * (t - self.p.t_melt)

        # Rain wash-off
        if rain_mm is not None:
            self.snow_index -= self.p.b_rain * max(0.0, float(rain_mm))

        # Sun/irradiance wash-off
        if irr_wm2 is not None:
            # per-hour energy in Wh/m² is ~ W/m² * 1h
            whm2 = max(0.0, float(irr_wm2))
            # scale to ~kWh/m²
            self.snow_index -= self.p.b_sun * (whm2 / 1000.0)

        # Passive decay
        if self.p.b_passive:
            self.snow_index -= float(self.p.b_passive)

        # clamp
        if self.snow_index < 0.0:
            self.snow_index = 0.0

    def factor(self) -> float:
        # Convert snow_index to multiplicative factor in [k, 1]
        # Using an exponential approach: f = k + (1-k)*exp(-snow_index)
        import math
        k = min(1.0, max(0.0, float(self.p.k)))
        f = k + (1.0 - k) * math.exp(-max(0.0, self.snow_index))
        # clamp
        return min(1.0, max(k, f))
