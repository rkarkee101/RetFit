# -*- coding: utf-8 -*-
"""
Retrofit-Focused Building Energy Simulation (Advanced, Hourly)
- Weather: hourly T + humidity (RH or dewpoint if available) via Meteostat
- Typical-Year: average of last 3 full years (8760 hours), UTC index
- Schedules: applied in LOCAL TIME; exact holiday dates/ranges per month
- Holidays: unoccupied all day; HVAC off; only minimal baseload & exterior lights at night
- HVAC: hourly sensible + latent cooling (from infiltration humidity); COPs by scenario
- Lighting: off when unoccupied/holiday except exterior/parking lights at night; daylight dimming when occupied
- Multi-zone; envelope, infiltration, internal gains; shading option
- Scenarios: Baseline, HVAC/Lighting/Envelope/Infiltration/Fan, All Combined
- Outputs: summary.txt + seaborn charts

Dependencies:
  pip install meteostat pandas numpy matplotlib seaborn pytz
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, List, Union
import math
import re

import numpy as np
import pandas as pd
from meteostat import Point, Hourly
import pytz
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# 1) USER INPUTS
# =========================================================

# ---- Location (Los Angeles by default; change to Texas, etc.) ----
LATITUDE   = 34.0522
LONGITUDE  = -118.2437
ELEV_M     = 71
LOCAL_TZ   = "America/Los_Angeles"  # schedules/holidays applied in this zone

# ---- Unoccupied behavior ----
# "off": HVAC off (no active heating/cooling) when unoccupied/holiday
# "setback": use unoccupied setpoints (minimal conditioning)
UNOCCUPIED_HVAC_MODE = "off"

# ---- School-like schedule with exact holidays per month ----
# start/end hours apply to NON-holiday days only.
# Holidays: either ints (e.g., 7, 8) or ranges like "1-8" separated by commas.
# Example: 'holidays': "1-8, 14, 21-22"  (whole days are unoccupied)
SCHEDULE: Dict[int, Dict[str, Union[int, str]]] = {
    1:  {'start_hour': 7, 'end_hour': 17, 'holidays': "1, 6, 20"},
    2:  {'start_hour': 8, 'end_hour': 16, 'holidays': "17"},
    3:  {'start_hour': 8, 'end_hour': 16, 'holidays': ""},
    4:  {'start_hour': 8, 'end_hour': 16, 'holidays': ""},
    5:  {'start_hour': 7, 'end_hour': 17, 'holidays': ""},
    6:  {'start_hour': 7, 'end_hour': 17, 'holidays': ""},          # keep open in summer if you want cooling
    7:  {'start_hour': 7, 'end_hour': 17, 'holidays': "1-7"},       # partial closure example
    8:  {'start_hour': 8, 'end_hour': 16, 'holidays': ""},
    9:  {'start_hour': 7, 'end_hour': 17, 'holidays': ""},
    10: {'start_hour': 7, 'end_hour': 17, 'holidays': ""},
    11: {'start_hour': 8, 'end_hour': 16, 'holidays': "28-29"},
    12: {'start_hour': 8, 'end_hour': 16, 'holidays': "24-31"}
}

# ---- Indoor setpoints (Â°C) ----
HEAT_SET_OCC   = 21.0
COOL_SET_OCC   = 24.0
HEAT_SET_UNOCC = 16.0
COOL_SET_UNOCC = 28.0
DEADBAND = 0.5  # neutral zone

# ---- Indoor humidity target (for dehumidification, latent cooling) ----
INDOOR_RH_TARGET_OCC   = 0.50  # 50% RH when occupied
INDOOR_RH_TARGET_UNOCC = 0.60  # 60% RH when unoccupied (rarely active if HVAC=off)

# ---- Building-wide minimal baseloads (when unoccupied or holiday) ----
# Always-on plug loads (e.g., servers, fridges) as W/m2 of zone area
UNOCC_EQUIP_W_PER_M2 = 0.5
# Exterior/parking lights at night (kW, building-level)
EXTERIOR_PARKING_LIGHT_KW = 5.0
# Night definition: sun_fraction < this, or local hour in [0..5] or [20..23]
NIGHT_SUN_FRAC_THRESH = 0.05

# ---- Daylighting & lighting control when occupied ----
DAYLIGHT_DIM_FACTOR = 0.5     # windowed lights to 50% when sunny
OCC_SENSOR_FACTOR   = 0.9     # 10% savings when sensors present
SHADING_REDUCTION   = 0.35    # apply to solar through south windows when shaded (35% remains)

# ---- Fans & HVAC efficiencies ----
BASE_FAN_W_PER_M2 = 0.5  # when HVAC is running (occupied & calling)
BASE_HEAT_COP = 1.0
BASE_COOL_COP = 3.0

# ---- Costs ----
ELECTRIC_RATE_USD_PER_KWH = 0.15
CAPEX = {
    "HVAC Upgrade": 120_000,
    "Lighting Upgrade": 60_000,
    "Envelope Upgrade": 80_000,
    "Infiltration Reduction": 40_000,
    "Fan Upgrade": 25_000,
    "All Combined": 120_000 + 60_000 + 80_000 + 40_000 + 25_000
}

# =========================================================
# 2) WEATHER: FETCH + TYPICAL YEAR
# =========================================================

def last_three_full_years() -> Tuple[int, int]:
    now = datetime.now()  # naive to avoid tz mix in meteostat
    end_year = now.year - 1
    start_year = end_year - 2
    return start_year, end_year

def fetch_hourly_weather(lat: float, lon: float, elev_m: float) -> pd.DataFrame:
    """Fetch hourly weather (T, RH or DP if available) for last 3 full years; return UTC-indexed DataFrame."""
    sy, ey = last_three_full_years()
    start = datetime(sy, 1, 1)
    end   = datetime(ey, 12, 31, 23, 59)
    loc = Point(lat, lon, elev_m)
    df = Hourly(loc, start, end).fetch()

    if 'temp' not in df.columns:
        raise RuntimeError("Meteostat weather missing 'temp' (Â°C). Try nearby station.")

    # Ensure columns exist
    for col in ['tsun', 'cldc', 'rhum', 'dwpt']:
        if col not in df.columns:
            df[col] = pd.NA

    # UTC index
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    else:
        df = df.tz_convert('UTC')

    return df

def saturation_vapor_pressure_Pa(T_C: float) -> float:
    """Magnus-Tetens approximation (Pa)."""
    return 610.94 * math.exp((17.625 * T_C) / (T_C + 243.04))

def humidity_ratio_from_T_RH(T_C: float, RH_frac: float, P_Pa: float = 101325.0) -> float:
    """w [kg/kg dry air] from dry-bulb T and RH."""
    RH = max(0.0, min(1.0, RH_frac))
    Psat = saturation_vapor_pressure_Pa(T_C)
    Pv   = RH * Psat
    return 0.62198 * Pv / max(P_Pa - Pv, 1.0)

def humidity_ratio_from_T_and_dewpoint(T_C: float, Tdp_C: float, P_Pa: float = 101325.0) -> float:
    """w from dry-bulb T and dewpoint."""
    Pv = saturation_vapor_pressure_Pa(Tdp_C)
    return 0.62198 * Pv / max(P_Pa - Pv, 1.0)

def build_typical_year(df_utc: pd.DataFrame) -> pd.DataFrame:
    """
    Build a typical-year (8760) in UTC by averaging same (month, day, hour) across 3 years.
    Outputs: temp_C, sun_fraction, rh_frac (if available), dewpoint_C (if available).
    """
    REF_YEAR = 2021
    hours_utc = pd.date_range(f"{REF_YEAR}-01-01 00:00", f"{REF_YEAR}-12-31 23:00", freq="h", tz="UTC")
    out = pd.DataFrame(index=hours_utc, columns=['temp_C', 'sun_fraction', 'rh_frac', 'dewpoint_C'], dtype=float)

    def solar_daylight_flag(ts_utc: pd.Timestamp, lat: float, lon: float) -> float:
        # cheap solar elevation proxy: local solar time ~ UTC + lon/15
        day = ts_utc.timetuple().tm_yday
        decl = 23.45 * math.pi/180.0 * math.sin(2*math.pi*(284 + day)/365.0)
        solar_time = (ts_utc.hour + ts_utc.minute/60.0) + lon/15.0
        ha = math.radians(15.0 * (solar_time - 12.0))
        latr = math.radians(lat)
        sin_alt = math.sin(latr)*math.sin(decl) + math.cos(latr)*math.cos(decl)*math.cos(ha)
        alt = math.asin(max(-1.0, min(1.0, sin_alt)))
        return 1.0 if (alt > math.radians(5.0)) else 0.0

    for ts in hours_utc:
        m, d, h = ts.month, ts.day, ts.hour
        match = df_utc[(df_utc.index.month == m) & (df_utc.index.day == d) & (df_utc.index.hour == h)]
        if match.empty:
            # fallback from previous hour
            if ts > hours_utc[0]:
                out.loc[ts] = out.loc[ts - pd.Timedelta(hours=1)]
            else:
                out.loc[ts] = [22.0, 0.0, np.nan, np.nan]
            continue

        # Temperature
        tempC = pd.to_numeric(match['temp'], errors='coerce')
        out.loc[ts, 'temp_C'] = float(np.nanmean(tempC))

        # Sun fraction: tsun (minutes) + cloud cover + geometry
        tsun = pd.to_numeric(match.get('tsun', pd.Series([pd.NA]*len(match))), errors='coerce')
        tsun_frac = tsun.fillna(0.0) / 60.0
        cldc = pd.to_numeric(match.get('cldc', pd.Series([pd.NA]*len(match))), errors='coerce')
        inv_cloud = (1.0 - cldc).clip(lower=0.0, upper=1.0).replace({np.inf: np.nan})
        dayflag = solar_daylight_flag(ts, LATITUDE, LONGITUDE)
        vals, wts = [], []
        if not tsun_frac.isna().all():
            vals.append(float(np.nanmean(tsun_frac))); wts.append(0.7)
        if not inv_cloud.isna().all():
            vals.append(float(np.nanmean(inv_cloud))); wts.append(0.2)
        vals.append(dayflag); wts.append(0.1)
        out.loc[ts, 'sun_fraction'] = max(0.0, min(1.0, float(np.average(vals, weights=wts))))

        # Humidity: rhum (%) and/or dewpoint (Â°C)
        rh = pd.to_numeric(match.get('rhum', pd.Series([pd.NA]*len(match))), errors='coerce')
        dp = pd.to_numeric(match.get('dwpt', pd.Series([pd.NA]*len(match))), errors='coerce')
        out.loc[ts, 'rh_frac']     = float(np.nanmean(rh) / 100.0) if not rh.isna().all() else np.nan
        out.loc[ts, 'dewpoint_C'] = float(np.nanmean(dp)) if not dp.isna().all() else np.nan

    return out

# =========================================================
# 3) SCHEDULES & HOLIDAYS (LOCAL TIME)
# =========================================================

def days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (datetime(year, month+1, 1) - datetime(year, month, 1)).days

def parse_holiday_string(holidays: str, year: int, month: int) -> set:
    """
    Parse strings like "1-5, 7, 14-16" into a set of day integers.
    Empty or None returns empty set.
    """
    out = set()
    if not holidays:
        return out
    # allow spaces
    tokens = [t.strip() for t in re.split(r'[,\s]+', str(holidays)) if t.strip()]
    dim = days_in_month(year, month)
    for tok in tokens:
        if '-' in tok:
            a, b = tok.split('-', 1)
            try:
                a = max(1, min(dim, int(a)))
                b = max(1, min(dim, int(b)))
                if a <= b:
                    out.update(range(a, b+1))
                else:
                    out.update(range(b, a+1))
            except:
                continue
        else:
            try:
                d = int(tok)
                if 1 <= d <= dim:
                    out.add(d)
            except:
                continue
    return out

def is_occupied_local(ts_local: pd.Timestamp, schedule: Dict[int, Dict[str, Union[int, str]]]) -> bool:
    """
    Return True if the given local timestamp is OCCUPIED (non-holiday within start..end hours).
    Holidays: unoccupied all day.
    """
    year = ts_local.year
    m, d, h = ts_local.month, ts_local.day, ts_local.hour
    sch = schedule[m]
    start_h = int(sch['start_hour'])
    end_h   = int(sch['end_hour'])
    holi_set = parse_holiday_string(str(sch.get('holidays', "")), year, m)

    if d in holi_set:
        return False
    if start_h == end_h:  # closed month (0..0)
        return False
    return start_h <= h < end_h

def is_night(ts_local: pd.Timestamp, sun_fraction: float) -> bool:
    """Night flag for exterior lights."""
    return (sun_fraction < NIGHT_SUN_FRAC_THRESH) or (ts_local.hour <= 5) or (ts_local.hour >= 20)

# =========================================================
# 4) ZONES, SCENARIOS, CONSTANTS
# =========================================================

@dataclass
class Zone:
    name: str
    area_m2: float
    volume_m3: float
    # Envelope (baseline)
    U_wall: float      # W/m2-K
    U_roof: float
    U_window: float
    A_wall: float
    A_roof: float
    A_window: float
    A_window_south: float
    # Infiltration (baseline)
    infiltration_ach: float
    # Internal gains (baseline densities)
    occupants: int
    equip_w_per_m2: float
    LPD_w_per_m2: float
    # Shading flags
    shading_existing: bool = False
    shading_new: bool = True

def default_zones() -> List[Zone]:
    classroom = Zone(
        name="Classroom", area_m2=100.0, volume_m3=300.0,
        U_wall=0.5, U_roof=0.3, U_window=2.5,
        A_wall=150.0, A_roof=100.0, A_window=20.0, A_window_south=12.0,
        infiltration_ach=1.0,
        occupants=30, equip_w_per_m2=5.0, LPD_w_per_m2=10.0,
        shading_existing=False, shading_new=True
    )
    office = Zone(
        name="Office", area_m2=60.0, volume_m3=180.0,
        U_wall=0.45, U_roof=0.28, U_window=2.2,
        A_wall=90.0, A_roof=60.0, A_window=12.0, A_window_south=8.0,
        infiltration_ach=0.8,
        occupants=8, equip_w_per_m2=8.0, LPD_w_per_m2=12.0,
        shading_existing=True, shading_new=True
    )
    gym = Zone(
        name="Gym", area_m2=200.0, volume_m3=900.0,
        U_wall=0.6, U_roof=0.35, U_window=3.0,
        A_wall=350.0, A_roof=200.0, A_window=30.0, A_window_south=20.0,
        infiltration_ach=1.2,
        occupants=50, equip_w_per_m2=3.0, LPD_w_per_m2=6.0,
        shading_existing=False, shading_new=True
    )
    cafeteria = Zone(
        name="Cafeteria", area_m2=250.0, volume_m3=750.0,
        U_wall=0.55, U_roof=0.35, U_window=2.8,
        A_wall=300.0, A_roof=250.0, A_window=25.0, A_window_south=15.0,
        infiltration_ach=1.5,
        occupants=120, equip_w_per_m2=15.0, LPD_w_per_m2=12.0,
        shading_existing=False, shading_new=True
    )
    lab_computer = Zone(
        name="Lab_Computer", area_m2=120.0, volume_m3=360.0,
        U_wall=0.45, U_roof=0.28, U_window=2.2,
        A_wall=150.0, A_roof=120.0, A_window=18.0, A_window_south=10.0,
        infiltration_ach=1.0,
        occupants=35, equip_w_per_m2=18.0, LPD_w_per_m2=12.0,
        shading_existing=True, shading_new=True
    )
    lab_science = Zone(
        name="Lab_Science", area_m2=150.0, volume_m3=450.0,
        U_wall=0.5, U_roof=0.3, U_window=2.5,
        A_wall=180.0, A_roof=150.0, A_window=20.0, A_window_south=10.0,
        infiltration_ach=2.0,  # higher due to fume hood make-up etc.
        occupants=25, equip_w_per_m2=12.0, LPD_w_per_m2=12.0,
        shading_existing=False, shading_new=True
    )
    return [classroom, office, gym, cafeteria, lab_computer, lab_science]

# Scenario parameters
SCENARIOS = {
    "Baseline": {
        "heat_COP": BASE_HEAT_COP, "cool_COP": BASE_COOL_COP,
        "LPD_factor": 1.0, "occ_sensor": False, "daylight": False,
        "U_wall_factor": 1.0, "U_roof_factor": 1.0, "U_window_factor": 1.0,
        "ACH_factor": 1.0, "fan_w_per_m2": BASE_FAN_W_PER_M2,
        "use_new_shading": False
    },
    "HVAC Upgrade": {
        "heat_COP": 3.0, "cool_COP": 4.0,
        "LPD_factor": 1.0, "occ_sensor": False, "daylight": False,
        "U_wall_factor": 1.0, "U_roof_factor": 1.0, "U_window_factor": 1.0,
        "ACH_factor": 1.0, "fan_w_per_m2": BASE_FAN_W_PER_M2,
        "use_new_shading": False
    },
    "Lighting Upgrade": {
        "heat_COP": BASE_HEAT_COP, "cool_COP": BASE_COOL_COP,
        "LPD_factor": 0.5, "occ_sensor": True, "daylight": True,
        "U_wall_factor": 1.0, "U_roof_factor": 1.0, "U_window_factor": 1.0,
        "ACH_factor": 1.0, "fan_w_per_m2": BASE_FAN_W_PER_M2,
        "use_new_shading": False
    },
    "Envelope Upgrade": {
        "heat_COP": BASE_HEAT_COP, "cool_COP": BASE_COOL_COP,
        "LPD_factor": 1.0, "occ_sensor": False, "daylight": False,
        "U_wall_factor": 0.5, "U_roof_factor": 0.5, "U_window_factor": 0.5,
        "ACH_factor": 1.0, "fan_w_per_m2": BASE_FAN_W_PER_M2,
        "use_new_shading": True
    },
    "Infiltration Reduction": {
        "heat_COP": BASE_HEAT_COP, "cool_COP": BASE_COOL_COP,
        "LPD_factor": 1.0, "occ_sensor": False, "daylight": False,
        "U_wall_factor": 1.0, "U_roof_factor": 1.0, "U_window_factor": 1.0,
        "ACH_factor": 0.5, "fan_w_per_m2": BASE_FAN_W_PER_M2,
        "use_new_shading": False
    },
    "Fan Upgrade": {
        "heat_COP": BASE_HEAT_COP, "cool_COP": BASE_COOL_COP,
        "LPD_factor": 1.0, "occ_sensor": False, "daylight": False,
        "U_wall_factor": 1.0, "U_roof_factor": 1.0, "U_window_factor": 1.0,
        "ACH_factor": 1.0, "fan_w_per_m2": 0.3,
        "use_new_shading": False
    },
    "All Combined": {
        "heat_COP": 3.0, "cool_COP": 4.0,
        "LPD_factor": 0.5, "occ_sensor": True, "daylight": True,
        "U_wall_factor": 0.5, "U_roof_factor": 0.5, "U_window_factor": 0.5,
        "ACH_factor": 0.5, "fan_w_per_m2": 0.3,
        "use_new_shading": True
    }
}

# =========================================================
# 5) SIMULATION CORE (hourly)
# =========================================================

@dataclass
class EndUse:
    Heating: float = 0.0
    Cooling: float = 0.0
    Lighting: float = 0.0
    Equipment: float = 0.0
    Fans: float = 0.0
    @property
    def Total(self) -> float:
        return self.Heating + self.Cooling + self.Lighting + self.Equipment + self.Fans

def simulate_scenario(typ: pd.DataFrame, zones: List[Zone], scen_name: str, params: dict) -> Dict:
    """
    typ: typical-year DataFrame with UTC index and columns:
         ['temp_C','sun_fraction','rh_frac','dewpoint_C']
    Returns overall dict with per-zone EndUse totals, building totals, and monthly (local) breakdown.
    """
    tz_local = pytz.timezone(LOCAL_TZ)
    REF_YEAR = typ.index[0].year

    per_zone = {z.name: EndUse() for z in zones}
    building = EndUse()
    monthly = {m: EndUse() for m in range(1, 13)}

    # Scenario params
    heat_COP = params["heat_COP"]; cool_COP = params["cool_COP"]
    LPD_factor = params["LPD_factor"]; occ_sensor = params["occ_sensor"]; daylight = params["daylight"]
    U_wall_factor = params["U_wall_factor"]; U_roof_factor = params["U_roof_factor"]; U_window_factor = params["U_window_factor"]
    ACH_factor = params["ACH_factor"]; fan_w_per_m2 = params["fan_w_per_m2"]; use_new_shading = params["use_new_shading"]

    # Constants
    rho_air = 1.2     # kg/m3
    cp_air  = 1005.0  # J/kg-K
    h_fg_kJ_per_kg = 2500.0  # latent heat ~ kJ/kg

    for ts_utc, row in typ.iterrows():
        # Local time for schedules & holidays
        ts_local = ts_utc.tz_convert(tz_local)
        m_local = ts_local.month

        To = float(row['temp_C'])
        sun_frac = float(row['sun_fraction'])
        rh_frac = row['rh_frac'] if not pd.isna(row['rh_frac']) else np.nan
        Tdp = row['dewpoint_C'] if not pd.isna(row['dewpoint_C']) else np.nan

        # Outdoor humidity ratio (w_out)
        if not np.isnan(rh_frac):
            w_out = humidity_ratio_from_T_RH(To, rh_frac)
        elif not np.isnan(Tdp):
            w_out = humidity_ratio_from_T_and_dewpoint(To, Tdp)
        else:
            w_out = np.nan  # latent effects off if we lack humidity

        # Occupancy (non-holiday within schedule)
        occ = is_occupied_local(ts_local, SCHEDULE)

        # Setpoints and indoor humidity target
        T_heat = HEAT_SET_OCC if occ else HEAT_SET_UNOCC
        T_cool = COOL_SET_OCC if occ else COOL_SET_UNOCC
        RH_in_target = INDOOR_RH_TARGET_OCC if occ else INDOOR_RH_TARGET_UNOCC
        w_in_target = humidity_ratio_from_T_RH(T_cool if occ else T_heat, RH_in_target)

        # Night flag for exterior lights
        night = is_night(ts_local, sun_frac)

        for z in zones:
            # Envelope UA
            UA = (z.U_wall * U_wall_factor) * z.A_wall + \
                 (z.U_roof * U_roof_factor) * z.A_roof + \
                 (z.U_window * U_window_factor) * z.A_window

            # Infiltration mass flow (kg/s of moist air ~ dry air)
            ach = z.infiltration_ach * ACH_factor
            vol_flow_m3_s = (ach / 3600.0) * z.volume_m3
            m_dot = vol_flow_m3_s * rho_air
            H_inf = m_dot * cp_air  # W/K (sensible infiltration conductance)

            # Solar gains through south glazing
            shade_factor = SHADING_REDUCTION if ((use_new_shading and z.shading_new) or ((not use_new_shading) and z.shading_existing)) else 1.0
            Q_solar_W = sun_frac * 150.0 * z.A_window_south * shade_factor  # 150 W/m2 at sun_fraction=1

            # Internal gains (occupied only)
            people_W = 75.0 * z.occupants if occ else 0.0
            equip_W_occ = z.equip_w_per_m2 * z.area_m2 if occ else 0.0
            light_W_occ = z.LPD_w_per_m2 * LPD_factor * z.area_m2 if occ else 0.0
            if occ and occ_sensor:
                light_W_occ *= OCC_SENSOR_FACTOR
            if occ and daylight and sun_frac > 0.2:
                light_W_occ *= DAYLIGHT_DIM_FACTOR
            internal_W = people_W + equip_W_occ + light_W_occ

            # Minimal baseloads when unoccupied/holiday
            equip_W_unocc = 0.0 if occ else (UNOCC_EQUIP_W_PER_M2 * z.area_m2)
            light_ext_kW  = (EXTERIOR_PARKING_LIGHT_KW if (not occ and night) else 0.0)

            # HVAC mode selection
            if UNOCCUPIED_HVAC_MODE.lower() == "off" and not occ:
                heating_mode = False
                cooling_mode = False
            else:
                heating_mode = To < (T_heat - DEADBAND)
                cooling_mode = To > (T_cool + DEADBAND)

            # Sensible loads
            heating_load_W = 0.0
            cooling_sensible_W = 0.0
            if heating_mode:
                dT = max(T_heat - To, 0.0)
                env_loss_W = (UA + H_inf) * dT
                heating_load_W = max(env_loss_W - (internal_W + Q_solar_W), 0.0)
            elif cooling_mode:
                dT = max(To - T_cool, 0.0)
                env_gain_W = (UA + H_inf) * dT
                cooling_sensible_W = max(env_gain_W + internal_W + Q_solar_W, 0.0)

            # Latent load: only when cooling and outdoor is more humid than target
            cooling_latent_kW = 0.0
            if cooling_mode and not np.isnan(w_out):
                dw = max(w_out - w_in_target, 0.0)
                if dw > 0 and m_dot > 0:
                    # m_dot (kg/s dry air) * dw (kg/kg) * h_fg (kJ/kg) = kJ/s = kW
                    cooling_latent_kW = m_dot * dw * h_fg_kJ_per_kg

            # Convert to kWh for this hour
            heat_kWh      = (heating_load_W / 1000.0) / max(heat_COP, 0.1)
            cool_sens_kWh = (cooling_sensible_W / 1000.0) / max(cool_COP, 0.1)
            cool_lat_kWh  = cooling_latent_kW / max(cool_COP, 0.1)
            cool_kWh      = cool_sens_kWh + cool_lat_kWh

            light_kWh     = (light_W_occ / 1000.0)
            equip_kWh     = (equip_W_occ / 1000.0)
            equip_unocc_kWh = (equip_W_unocc / 1000.0)
            ext_light_kWh = light_ext_kW  # already kW

            # Fans only when HVAC is actually conditioning
            fans_kWh = 0.0
            if heating_mode or cooling_mode:
                fans_kWh = (fan_w_per_m2 * z.area_m2) / 1000.0

            # Accumulate
            ez = per_zone[z.name]
            ez.Heating   += heat_kWh
            ez.Cooling   += cool_kWh
            ez.Lighting  += light_kWh
            ez.Equipment += equip_kWh + equip_unocc_kWh
            ez.Fans      += fans_kWh

            building.Heating   += heat_kWh
            building.Cooling   += cool_kWh
            building.Lighting  += light_kWh + ext_light_kWh
            building.Equipment += equip_kWh + equip_unocc_kWh
            building.Fans      += fans_kWh

            mm = monthly[m_local]
            mm.Heating   += heat_kWh
            mm.Cooling   += cool_kWh
            mm.Lighting  += light_kWh + ext_light_kWh
            mm.Equipment += equip_kWh + equip_unocc_kWh
            mm.Fans      += fans_kWh

    # Monthly dataframe
    mon_df = pd.DataFrame({
        "Month": range(1, 13),
        "Heating": [monthly[m].Heating for m in range(1,13)],
        "Cooling": [monthly[m].Cooling for m in range(1,13)],
        "Lighting": [monthly[m].Lighting for m in range(1,13)],
        "Equipment": [monthly[m].Equipment for m in range(1,13)],
        "Fans": [monthly[m].Fans for m in range(1,13)]
    }).set_index("Month")
    mon_df["Total"] = mon_df.sum(axis=1)

    return {"per_zone": per_zone, "building": building, "building_monthly": mon_df}

# =========================================================
# 6) REPORTING & PLOTS (Seaborn)
# =========================================================

def write_summary(results: Dict[str, Dict], rate: float, capex: Dict[str, float]):
    base = results["Baseline"]["building"]
    base_total = base.Total
    base_cost = base_total * rate
    lines = []
    lines.append(f"Baseline Total: {base_total:.0f} kWh  (${base_cost:,.0f})")
    lines.append(f"  Heating={base.Heating:.0f}, Cooling={base.Cooling:.0f}, Lighting={base.Lighting:.0f}, "
                 f"Equipment={base.Equipment:.0f}, Fans={base.Fans:.0f}\n")

    # Monthly snapshot (baseline)
    lines.append("Baseline Monthly (local months):")
    bmon = results["Baseline"]["building_monthly"]
    for m, row in bmon.iterrows():
        lines.append(f"  {m:02d}: Heat={row['Heating']:.0f} kWh, Cool={row['Cooling']:.0f} kWh, "
                     f"Light={row['Lighting']:.0f} kWh, Equip={row['Equipment']:.0f} kWh, "
                     f"Fans={row['Fans']:.0f} kWh, Total={row['Total']:.0f} kWh")
    lines.append("")

    for scen, out in results.items():
        if scen == "Baseline":
            continue
        b = out["building"]
        tot = b.Total
        cost = tot * rate
        save_kWh = base_total - tot
        save_pct = (save_kWh / base_total * 100.0) if base_total > 0 else 0.0
        save_cost = base_cost - cost
        payback  = (capex.get(scen, 0.0) / save_cost) if save_cost > 0 else float('inf')
        lines.append(f"{scen}: {tot:.0f} kWh  (${cost:,.0f})")
        lines.append(f"  Savings vs Base: {save_kWh:.0f} kWh ({save_pct:.1f}%), ${save_cost:,.0f}/yr")
        lines.append(f"  Simple Payback: CAPEX=${capex.get(scen,0):,.0f} â†’ {payback:.1f} years")
        lines.append(f"  Breakdown: H={b.Heating:.0f}, C={b.Cooling:.0f}, L={b.Lighting:.0f}, "
                     f"E={b.Equipment:.0f}, F={b.Fans:.0f}\n")

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def plot_by_scenario_and_zone(results: Dict[str, Dict]):
    rows = []
    for scen, data in results.items():
        for zname, eu in data["per_zone"].items():
            rows.append({"Scenario": scen, "Zone": zname, "kWh": eu.Total})
    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12,6), dpi=130)
    ax = sns.barplot(data=df, x="Scenario", y="kWh", hue="Zone", palette="Set2")
    ax.set_title("Annual Energy by Scenario and Zone")
    ax.set_xlabel("")
    ax.set_ylabel("Energy (kWh)")
    plt.xticks(rotation=20, ha="right")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.0f}", (p.get_x()+p.get_width()/2, h), ha='center', va='bottom',
                    fontsize=8, xytext=(0,3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig("energy_by_scenario_zone.png")
    plt.close()

def plot_by_scenario_and_enduse(results: Dict[str, Dict]):
    rows = []
    for scen, data in results.items():
        b = data["building"]
        rows += [
            {"Scenario": scen, "End Use": "Heating",   "kWh": b.Heating},
            {"Scenario": scen, "End Use": "Cooling",   "kWh": b.Cooling},
            {"Scenario": scen, "End Use": "Lighting",  "kWh": b.Lighting},
            {"Scenario": scen, "End Use": "Equipment", "kWh": b.Equipment},
            {"Scenario": scen, "End Use": "Fans",      "kWh": b.Fans},
        ]
    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12,6), dpi=130)
    ax = sns.barplot(data=df, x="Scenario", y="kWh", hue="End Use", palette="Paired")
    ax.set_title("Annual Energy by Scenario and End Use")
    ax.set_xlabel("")
    ax.set_ylabel("Energy (kWh)")
    plt.xticks(rotation=20, ha="right")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.0f}", (p.get_x()+p.get_width()/2, h), ha='center', va='bottom',
                    fontsize=8, xytext=(0,3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig("energy_by_scenario_enduse.png")
    plt.close()

def plot_monthly_baseline(results: Dict[str, Dict]):
    bmon = results["Baseline"]["building_monthly"].reset_index()
    bmon_melt = bmon.melt(id_vars="Month",
                          value_vars=["Heating","Cooling","Lighting","Equipment","Fans"],
                          var_name="End Use", value_name="kWh")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12,5), dpi=130)
    ax = sns.barplot(data=bmon_melt, x="Month", y="kWh", hue="End Use", palette="icefire")
    ax.set_title("Baseline Monthly End Uses (Local Months)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Energy (kWh)")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.0f}", (p.get_x()+p.get_width()/2, h), ha='center', va='bottom',
                        fontsize=8, xytext=(0,3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig("baseline_monthly_enduses.png")
    plt.close()

# =========================================================
# 7) MAIN
# =========================================================

def main():
    print("Fetching hourly weather (last 3 full years) ...")
    wx = fetch_hourly_weather(LATITUDE, LONGITUDE, ELEV_M)  # UTC

    print("Building typical-year (8760h, UTC) ...")
    typ = build_typical_year(wx)

    zones = default_zones()

    print(f"Simulating scenarios (local tz: {LOCAL_TZ}, unocc HVAC: {UNOCCUPIED_HVAC_MODE}) ...")
    results = {}
    for scen, params in SCENARIOS.items():
        results[scen] = simulate_scenario(typ, zones, scen, params)

    write_summary(results, ELECTRIC_RATE_USD_PER_KWH, CAPEX)
    print("summary.txt written")

    plot_by_scenario_and_zone(results)
    plot_by_scenario_and_enduse(results)
    plot_monthly_baseline(results)
    print("Charts saved: energy_by_scenario_zone.png, energy_by_scenario_enduse.png, baseline_monthly_enduses.png")

    base = results["Baseline"]["building"]
    allc = results["All Combined"]["building"]
    print("\nAnnual Energy (kWh):")
    print(f"  Baseline:     {base.Total:.0f} (H={base.Heating:.0f}, C={base.Cooling:.0f}, "
          f"L={base.Lighting:.0f}, E={base.Equipment:.0f}, F={base.Fans:.0f})")
    print(f"  All Combined: {allc.Total:.0f} (H={allc.Heating:.0f}, C={allc.Cooling:.0f}, "
          f"L={allc.Lighting:.0f}, E={allc.Equipment:.0f}, F={allc.Fans:.0f})")
    print(f"  Savings:      {base.Total - allc.Total:.0f} "
          f"({(100.0*(base.Total-allc.Total)/base.Total if base.Total>0 else 0):.1f}%)")

if __name__ == "__main__":
    main()