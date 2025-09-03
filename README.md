# PyRetFit


# Retrofit-Focused Building Energy Simulation (Hourly)

**Baseline vs. Retrofit** building energy model for existing facilities (e.g., schools).  
Pulls **hourly weather** (T, RH/Dewpoint, sunshine) via Meteostat, builds a **typical 8760**, applies **local-time schedules with exact holiday dates**, and simulates **HVAC (sensible + latent)**, **lighting (with daylight/occupancy controls)**, **fans**, **plug baseloads**, **solar/shading**, and **envelope + infiltration**.  
Outputs a **summary** and **Seaborn plots**. Scenarios include **HVAC**, **Lighting**, **Envelope**, **Infiltration**, **Fans**, and **All Combined**.

> This is a lightweight alternative for retrofit analyses—**not** a full EnergyPlus/eQuest replacement—designed for transparency and quick iteration.

---

## Quickstart

```bash
pip install meteostat pandas numpy matplotlib seaborn pytz
python main.py
```

Outputs:
- `summary.txt`
- `energy_by_scenario_zone.png`
- `energy_by_scenario_enduse.png`
- `baseline_monthly_enduses.png`

---

## What to Edit (User Inputs)

Open `main.py` and edit the **“1) USER INPUTS”** block.

### Site & Time Zone
```python
LATITUDE, LONGITUDE, ELEV_M
LOCAL_TZ = "America/Los_Angeles"  # schedules in local time
```

### Unoccupied Behavior
```python
UNOCCUPIED_HVAC_MODE = "off"     # HVAC off when unoccupied/holiday
# or
UNOCCUPIED_HVAC_MODE = "setback" # minimal conditioning via unoccupied setpoints
```

### Schedules & Exact Holidays
Per-month dict (local time). `start_hour`/`end_hour` apply to **non-holiday** days.  
Holidays are **full days**: strings like `"1-8, 14, 21-22"`.
```python
SCHEDULE = {
  1: {'start_hour': 7, 'end_hour': 17, 'holidays': "1, 6, 20"},
  2: {'start_hour': 8, 'end_hour': 16, 'holidays': "17"},
  ...
}
```
- Fully closed month? Use `"1-31"` (or exact length) or set `start_hour == end_hour` (e.g. `0`, `0`).  
- On **holidays** the building is **unoccupied all day**: HVAC off (if `"off"`), interior lights off, only parking/exterior lights at night + minimal plug baseload.

### Indoor Setpoints & Humidity Targets
```python
HEAT_SET_OCC, COOL_SET_OCC
HEAT_SET_UNOCC, COOL_SET_UNOCC  # used only if UNOCCUPIED_HVAC_MODE="setback"
INDOOR_RH_TARGET_OCC, INDOOR_RH_TARGET_UNOCC  # for latent estimate
```

### Minimal Baseloads & Exterior Lighting
```python
UNOCC_EQUIP_W_PER_M2 = 0.5          # always-on plugs in unoccupied/holiday
EXTERIOR_PARKING_LIGHT_KW = 5.0     # only at night when unoccupied/holiday
NIGHT_SUN_FRAC_THRESH = 0.05
```

### Daylighting & Lighting Controls
```python
DAYLIGHT_DIM_FACTOR = 0.5   # 50% when sunny
OCC_SENSOR_FACTOR   = 0.9   # 10% savings when occupied
```

### Fans & HVAC Efficiencies
```python
BASE_FAN_W_PER_M2 = 0.5
BASE_HEAT_COP = 1.0
BASE_COOL_COP = 3.0
```

### Costs
```python
ELECTRIC_RATE_USD_PER_KWH = 0.15
CAPEX = { "HVAC Upgrade": 120000, ... }
```

### Zones (Edit `default_zones()`)
Add/modify zones with geometry, envelope, infiltration, gains, glazing, shading.
```python
Zone(
  name="Cafeteria", area_m2=250.0, volume_m3=750.0,
  U_wall=0.55, U_roof=0.35, U_window=2.8,
  A_wall=300.0, A_roof=250.0, A_window=25.0, A_window_south=15.0,
  infiltration_ach=1.5,
  occupants=120, equip_w_per_m2=15.0, LPD_w_per_m2=12.0,
  shading_existing=False, shading_new=True
)
```
Included defaults: `Classroom`, `Office`, `Gym`, **`Cafeteria`**, **`Lab_Computer`**, **`Lab_Science`**.

### Scenarios (Edit `SCENARIOS`)
Each scenario adjusts efficiency & envelope/control factors:
- `heat_COP`, `cool_COP`
- `LPD_factor`, `occ_sensor`, `daylight`
- `U_wall_factor`, `U_roof_factor`, `U_window_factor`
- `ACH_factor`
- `fan_w_per_m2`
- `use_new_shading`

---

## Method & Theory (Hourly)

### Weather
- Meteostat hourly data (last **3 full years**): dry-bulb `temp` (°C), `rhum` (% RH) and/or `dwpt` (°C), sunshine `tsun` (min), cloud cover `cldc`.
- Builds a **typical 8760** by averaging the same `(month, day, hour)` across the 3 years (UTC).
- `sun_fraction` ∈ [0,1] from a weighted mix: `tsun` (0.7), `1 - cldc` (0.2), and a simple solar-geometry day/night flag (0.1).

### Local-Time Schedules & Holidays
- Convert each hour to `LOCAL_TZ` before schedule checks.
- **Occupied** iff **not** a holiday and `start_hour ≤ hour < end_hour`.
- **Holidays** are full unoccupied days from explicit lists/ranges (per local month).
- When **unoccupied/holiday**:
  - If `UNOCCUPIED_HVAC_MODE="off"`: no active heating/cooling.
  - Interior lights off; exterior/parking lights at night only.
  - Minimal plug baseload remains.

### Zone Loads (per hour)
1. **Envelope Sensible**  
   `UA = Σ(U_i * A_i)` (walls, roof, windows; scaled by scenario factors)  
   Infiltration sensible conductance: `H_inf = m_dot * c_p`, with `m_dot = ρ * (ACH/3600) * Volume`  
   - Heating (when `T_out < T_heat - deadband`):  
     `EnvLoss = (UA + H_inf) * max(T_heat - T_out, 0)`  
     `HeatingLoad = max(EnvLoss - (Internal + Solar), 0)`  
   - Cooling (when `T_out > T_cool + deadband`):  
     `EnvGain = (UA + H_inf) * max(T_out - T_cool, 0)`  
     `CoolingSensible = max(EnvGain + Internal + Solar, 0)`

2. **Internal Gains** (occupied only)  
   - People ~ **75 W/person**  
   - Equipment: `equip_w_per_m2 * area`  
   - Lighting: `LPD_w_per_m2 * LPD_factor * area`  
     - `occ_sensor=True` → × `0.9`; `daylight=True` and `sun_fraction > 0.2` → × `0.5`

3. **Solar Gains** (simplified)  
   - `Solar = sun_fraction * 150 W/m² * A_window_south * shade_factor`  
   - `shade_factor = 0.35` when (existing or new) shading is active in the scenario, else `1.0`.

4. **Latent Cooling** (from infiltration humidity)  
   - Compute outdoor humidity ratio `w_out` (from RH or dewpoint).  
   - Define indoor **target** ratio `w_in_target` from setpoint & target RH.  
   - When cooling: `CoolingLatent (kW) = m_dot * max(w_out - w_in_target, 0) * h_fg (≈2500 kJ/kg)`.

5. **HVAC Electricity**  
   - `HeatElec (kWh) = HeatingLoad (kW_th) / COP_heat`  
   - `CoolElec (kWh) = (CoolingSensible + CoolingLatent) (kW_th) / COP_cool`  
   - **Fans** accrue only when actively heating/cooling: `fan_w_per_m2 * area`.

6. **Lighting & Plug Loads**  
   - Occupied: interior lights/equipment as above.  
   - Unoccupied/holiday: interior lights **off**; **exterior lights** at night; **UNOCC_EQUIP_W_PER_M2** plug baseload.

### Scenarios – What Changes
- **Baseline**: Zone inputs “as is”; baseline COPs, fans; no daylight/sensors; existing shading flags.  
- **HVAC Upgrade**: Improves COPs (3.0/4.0).  
- **Lighting Upgrade**: `LPD_factor=0.5`, `occ_sensor=True`, `daylight=True`.  
- **Envelope Upgrade**: U-factors × 0.5, enable `use_new_shading=True`.  
- **Infiltration Reduction**: ACH × 0.5.  
- **Fan Upgrade**: `fan_w_per_m2` reduced (e.g., 0.3).  
- **All Combined**: All the above changes together.

---

## Tips & Validation

- **Climate sanity**: LA shows little heating; Texas in summer shows dominant cooling—ensure June–Aug are **occupied** to reflect reality.
- **Holidays**: Use exact day lists/ranges; on holidays HVAC is off (default) and only minimal loads remain.
- **Humidity gaps**: If RH/dewpoint missing at your station, latent is skipped (sensible still modeled). Try nearby stations if needed.
- **Zones**: Add `A_window_south=0` for interior rooms; increase `infiltration_ach` for labs with fume hoods.
- **Check monthly**: See `baseline_monthly_enduses.png` and the monthly table in `summary.txt`.

---


## Acknowledgements
Weather data by [Meteostat](https://meteostat.net/).
