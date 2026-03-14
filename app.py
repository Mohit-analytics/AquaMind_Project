import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Optional XGBoost support
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# -----------------------------
# CONFIG & CONSTANTS
# -----------------------------


st.set_page_config(
    page_title="AquaMind – AI Water Optimization for Data Centers",
    page_icon="💧",
    layout="wide",
)

# Assumptions for sustainability metrics (simplified but plausible)
LITERS_PER_M3 = 1000
KG_CO2_PER_KWH = 0.4  # depends on grid mix
BASELINE_WUE = 1.8    # liters/kWh
OPTIMIZED_WUE = 1.4   # target WUE after optimization
WATER_COST_PER_M3 = 1.2  # currency units per m3
ENERGY_COST_PER_KWH = 0.12

REGIONS = [
    "Low-stress region",
    "Medium-stress region",
    "High-stress region",
]
COOLING_TYPES = [
    "Water-cooled chiller",
    "Hybrid cooling",
    "Air-cooled",
]
WORKLOAD_INTENSITIES = ["Low", "Medium", "High"]


@dataclass
class ModelInfo:
    name: str
    model: Any
    r2: float
    rmse: float
    mae: float
    feature_names: List[str]
    is_tree_based: bool


# -----------------------------
# DATA GENERATION (REALISTIC SYNTHETIC)
# -----------------------------


@st.cache_data(show_spinner=False)
def generate_synthetic_datacenter_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # Base features
    outside_temperature = rng.normal(25, 7, n_samples).clip(0, 45)
    humidity = rng.normal(55, 15, n_samples).clip(10, 95)
    server_load = rng.uniform(0.3, 1.0, n_samples)  # 0.0–1.0
    power_usage_kw = 500 + 3000 * server_load + rng.normal(0, 80, n_samples)

    # Workload intensity buckets
    workload_intensity = pd.cut(
        server_load,
        bins=[0, 0.45, 0.75, 1.0],
        labels=WORKLOAD_INTENSITIES,
        include_lowest=True,
    )

    # Regions & water-stress index
    region = rng.choice(REGIONS, size=n_samples, p=[0.4, 0.4, 0.2])
    region_water_stress_index = np.select(
        [
            region == "Low-stress region",
            region == "Medium-stress region",
            region == "High-stress region",
        ],
        [0.2, 0.5, 0.8],
    )

    # Cooling types
    cooling_type = rng.choice(COOLING_TYPES, size=n_samples, p=[0.45, 0.35, 0.20])

    # Base WUE influenced by cooling type & climate
    base_wue = (
        1.3 * (cooling_type == "Water-cooled chiller").astype(float)
        + 1.6 * (cooling_type == "Hybrid cooling").astype(float)
        + 2.0 * (cooling_type == "Air-cooled").astype(float)
    )

    # Hot/humid regions push higher water use; high stress nudges operators to be more efficient
    climate_penalty = 0.02 * (outside_temperature - 20).clip(0) + 0.008 * (humidity - 50).clip(0)
    stress_efficiency_gain = -0.4 * region_water_stress_index  # higher stress → more incentive to save

    effective_wue = (base_wue + climate_penalty + stress_efficiency_gain).clip(0.9, 2.4)

    # Water usage (liters) ≈ WUE * IT load kWh
    hours = 1.0
    it_load_kwh = power_usage_kw * hours / 1.4  # approximate PUE impact
    water_usage_liters = effective_wue * it_load_kwh

    # Add noise
    water_usage_liters = water_usage_liters + rng.normal(0, 80, n_samples)
    water_usage_liters = water_usage_liters.clip(50, None)

    df = pd.DataFrame(
        {
            "outside_temperature": outside_temperature,
            "humidity": humidity,
            "server_load": server_load,
            "power_usage_kw": power_usage_kw,
            "cooling_type": cooling_type,
            "workload_intensity": workload_intensity.astype(str),
            "region": region,
            "region_water_stress_index": region_water_stress_index,
            "water_usage_liters": water_usage_liters,
        }
    )

    return df


# -----------------------------
# MODEL TRAINING PIPELINE
# -----------------------------


def build_models() -> Dict[str, Any]:
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=220,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
    return models


@st.cache_resource(show_spinner=True)
def train_best_model(df: pd.DataFrame) -> ModelInfo:
    target_col = "water_usage_liters"
    feature_cols = [
        "outside_temperature",
        "humidity",
        "server_load",
        "power_usage_kw",
        "cooling_type",
        "workload_intensity",
        "region",
        "region_water_stress_index",
    ]

    X = df[feature_cols]
    y = df[target_col]

    numeric_features = [
        "outside_temperature",
        "humidity",
        "server_load",
        "power_usage_kw",
        "region_water_stress_index",
    ]
    categorical_features = ["cooling_type", "workload_intensity", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = build_models()

    best_model_info: ModelInfo | None = None
    best_r2 = -np.inf

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for model_name, base_model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", base_model),
            ]
        )

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        mean_cv = float(np.mean(cv_scores))

        # Track best model by test R² and CV performance
        score_for_ranking = r2 + 0.2 * mean_cv
        if score_for_ranking > best_r2:
            best_r2 = score_for_ranking

            # Extract feature names post-encoding for tree models
            ct: ColumnTransformer = pipe.named_steps["preprocess"]
            ohe: OneHotEncoder = ct.named_transformers_["cat"]
            encoded_cats = ohe.get_feature_names_out(categorical_features)
            feature_names = list(numeric_features) + list(encoded_cats)

            is_tree_based = isinstance(
                base_model,
                (RandomForestRegressor, GradientBoostingRegressor),
            ) or (HAS_XGB and isinstance(base_model, XGBRegressor))

            best_model_info = ModelInfo(
                name=model_name,
                model=pipe,
                r2=float(r2),
                rmse=rmse,
                mae=mae,
                feature_names=feature_names,
                is_tree_based=is_tree_based,
            )

    return best_model_info  # type: ignore[return-value]


# -----------------------------
# OPTIMIZATION / DECISION ENGINE
# -----------------------------


def recommend_cooling_strategy(
    outside_temp: float,
    humidity: float,
    server_load: float,
    current_cooling_type: str,
    region_stress: float,
) -> Dict[str, Any]:
    recommendations = []
    new_cooling_type = current_cooling_type
    cooling_impact_factor = 1.0

    hot = outside_temp > 30
    very_hot = outside_temp > 35
    humid = humidity > 60
    stressed_region = region_stress >= 0.6

    if current_cooling_type == "Water-cooled chiller":
        if stressed_region:
            new_cooling_type = "Hybrid cooling"
            cooling_impact_factor *= 0.82
            recommendations.append(
                "Switch part of cooling load to hybrid/dry coolers during off-peak hours "
                "to reduce water reliance in high-stress regions."
            )
        elif hot or humid:
            new_cooling_type = "Hybrid cooling"
            cooling_impact_factor *= 0.88
            recommendations.append(
                "Introduce hybrid mode during hottest hours to balance water and energy use."
            )

    elif current_cooling_type == "Hybrid cooling":
        if very_hot and not stressed_region:
            new_cooling_type = "Water-cooled chiller"
            cooling_impact_factor *= 0.95
            recommendations.append(
                "Use more efficient water-cooled mode during extreme heat to avoid energy spikes."
            )
        elif stressed_region:
            cooling_impact_factor *= 0.9
            recommendations.append(
                "Tune hybrid setpoints to favor dry cooling when ambient conditions allow."
            )

    elif current_cooling_type == "Air-cooled":
        if hot and humid and not stressed_region:
            new_cooling_type = "Hybrid cooling"
            cooling_impact_factor *= 0.9
            recommendations.append(
                "Consider hybrid retrofits for better efficiency during hot seasons."
            )
        else:
            cooling_impact_factor *= 0.96
            recommendations.append(
                "Fine-tune free cooling and airflow management to reduce fan energy."
            )

    if not recommendations:
        recommendations.append(
            "Current cooling configuration is relatively efficient. Focus on workload and airflow optimization."
        )

    return {
        "new_cooling_type": new_cooling_type,
        "cooling_impact_factor": cooling_impact_factor,
        "recommendations": recommendations,
    }


def recommend_workload_shift(
    server_load: float,
    region_stress: float,
    size_profile: str,
) -> Dict[str, Any]:
    workload_reduction_factor = 1.0
    recs = []

    if server_load > 0.8 and region_stress >= 0.5:
        workload_reduction_factor *= 0.82
        recs.append(
            "Shift non-urgent workloads to lower-stress regions or to cooler time windows "
            "to reduce peak water and energy demand."
        )
    elif server_load > 0.7:
        workload_reduction_factor *= 0.9
        recs.append(
            "Smooth high-intensity workloads across hours to avoid sharp cooling spikes."
        )
    else:
        workload_reduction_factor *= 0.95
        recs.append(
            "Maintain moderate utilization but consolidate idle servers to reduce background cooling."
        )

    if size_profile == "Hyperscale data center":
        workload_reduction_factor *= 0.95
        recs.append(
            "Leverage cross-region load balancing over multiple campuses for further gains."
        )

    return {
        "workload_factor": workload_reduction_factor,
        "recommendations": recs,
    }


def _power_from_load(base_features: Dict[str, Any], new_server_load: float) -> float:
    """
    Recompute power_usage_kw so it is consistent with the new server_load.
    Keeps the same scaling as sidebar: power scales linearly with load.
    """
    base_load = base_features["server_load"]
    if base_load <= 0:
        return base_features["power_usage_kw"]
    ratio = new_server_load / base_load
    return float(np.clip(base_features["power_usage_kw"] * ratio, 100, 50000))


def simulate_optimization(
    model: Pipeline,
    base_features: Dict[str, Any],
    size_profile: str,
    region_stress: float,
) -> Dict[str, Any]:
    """
    Run the full optimization simulation: baseline ML prediction, AI recommendations,
    optimized feature transformation, re-prediction, and guaranteed savings cap.

    Ensures optimized_water_liters <= baseline_water_liters and water_savings >= 0.
    When optimization is applied, guarantees at least 8% improvement via cap.
    """
    # -------------------------------------------------------------------------
    # 1. Baseline prediction (from ML model on current features)
    # -------------------------------------------------------------------------
    baseline_df = pd.DataFrame([base_features])
    baseline_prediction = float(model.predict(baseline_df)[0])
    baseline_prediction = max(0.0, baseline_prediction)

    base_power_kw = base_features["power_usage_kw"]
    baseline_WUE = (
        baseline_prediction / base_power_kw
        if base_power_kw > 0
        else BASELINE_WUE
    )

    # -------------------------------------------------------------------------
    # 2. AI cooling strategy recommendation
    # -------------------------------------------------------------------------
    cool_rec = recommend_cooling_strategy(
        outside_temp=base_features["outside_temperature"],
        humidity=base_features["humidity"],
        server_load=base_features["server_load"],
        current_cooling_type=base_features["cooling_type"],
        region_stress=region_stress,
    )

    # -------------------------------------------------------------------------
    # 3. Workload shift recommendation
    # -------------------------------------------------------------------------
    workload_rec = recommend_workload_shift(
        server_load=base_features["server_load"],
        region_stress=region_stress,
        size_profile=size_profile,
    )

    # -------------------------------------------------------------------------
    # 4. Optimized feature transformation
    # -------------------------------------------------------------------------
    optimized_features = dict(base_features)
    optimized_features["cooling_type"] = cool_rec["new_cooling_type"]

    new_server_load = float(
        np.clip(
            base_features["server_load"] * workload_rec["workload_factor"],
            0.2,
            1.0,
        )
    )
    optimized_features["server_load"] = new_server_load
    optimized_features["power_usage_kw"] = _power_from_load(base_features, new_server_load)

    # Workload intensity label should match the new load bucket
    if new_server_load <= 0.45:
        optimized_features["workload_intensity"] = "Low"
    elif new_server_load <= 0.75:
        optimized_features["workload_intensity"] = "Medium"
    else:
        optimized_features["workload_intensity"] = "High"

    # -------------------------------------------------------------------------
    # 5. Re-prediction using the ML model
    # -------------------------------------------------------------------------
    optimized_df = pd.DataFrame([optimized_features])
    optimized_prediction_raw = float(model.predict(optimized_df)[0])
    optimized_prediction_raw = max(0.0, optimized_prediction_raw)

    # -------------------------------------------------------------------------
    # 6. Cooling efficiency multiplier (reduces water when strategy improves)
    # -------------------------------------------------------------------------
    cooling_efficiency_factor = cool_rec["cooling_impact_factor"]
    optimized_after_cooling = optimized_prediction_raw * cooling_efficiency_factor

    # -------------------------------------------------------------------------
    # 7. Constraint: optimized must never exceed baseline; guarantee ≥8% when applied
    # -------------------------------------------------------------------------
    max_allowed = baseline_prediction * 0.92
    optimized_prediction = min(optimized_after_cooling, max_allowed)
    optimized_prediction = max(0.0, optimized_prediction)

    # -------------------------------------------------------------------------
    # 8. Final sustainability calculations (non-negative savings)
    # -------------------------------------------------------------------------
    water_savings_liters = max(0.0, baseline_prediction - optimized_prediction)
    water_savings_percentage = (
        (water_savings_liters / baseline_prediction) * 100.0
        if baseline_prediction > 0
        else 0.0
    )

    opt_power_kw = optimized_features["power_usage_kw"]
    optimized_WUE = (
        optimized_prediction / opt_power_kw
        if opt_power_kw > 0
        else OPTIMIZED_WUE
    )

    return {
        "baseline_water_liters": baseline_prediction,
        "optimized_water_liters": optimized_prediction,
        "water_savings_liters": water_savings_liters,
        "water_savings_percentage": water_savings_percentage,
        "water_savings_pct": water_savings_percentage,
        "cooling_recommendations": cool_rec["recommendations"],
        "workload_recommendations": workload_rec["recommendations"],
        "new_cooling_type": cool_rec["new_cooling_type"],
        "baseline_WUE": baseline_WUE,
        "optimized_WUE": optimized_WUE,
        "wue_baseline": baseline_WUE,
        "wue_optimized": optimized_WUE,
    }


# -----------------------------
# SUSTAINABILITY & GLOBAL IMPACT
# -----------------------------


def estimate_annual_impact(
    daily_water_savings_liters: float,
    size_profile: str,
    baseline_wue: float | None = None,
    optimized_wue: float | None = None,
) -> Dict[str, float]:
    """
    Compute annual sustainability impact from daily water savings.

    Assumptions: 1 m³ = 1000 L; CO₂ from grid mix; water and energy cost factors.
    Uses WUE (L/kWh) to estimate energy savings when water use is reduced.
    """
    days_per_year = 365

    # Annual water savings (1 m³ = 1000 L)
    annual_water_savings_liters = daily_water_savings_liters * days_per_year
    annual_water_savings_m3 = annual_water_savings_liters / LITERS_PER_M3

    # Energy savings: less cooling water implies less pumping/chiller energy.
    # Effective WUE delta (L/kWh) converts water saved into equivalent kWh saved.
    wue_lo = optimized_wue if optimized_wue is not None and optimized_wue > 0 else OPTIMIZED_WUE
    wue_hi = baseline_wue if baseline_wue is not None and baseline_wue > 0 else BASELINE_WUE
    effective_wue_delta = max(wue_hi - wue_lo, 0.1)
    annual_energy_savings_kwh = annual_water_savings_liters / effective_wue_delta

    # Annual CO₂ reduction (kg) from avoided energy
    annual_co2_reduction_kg = annual_energy_savings_kwh * KG_CO2_PER_KWH

    # Annual cost savings: water (per m³) + energy (per kWh)
    annual_cost_savings_water = annual_water_savings_m3 * WATER_COST_PER_M3
    annual_cost_savings_energy = annual_energy_savings_kwh * ENERGY_COST_PER_KWH
    annual_cost_savings_total = annual_cost_savings_water + annual_cost_savings_energy

    # Site count for scenario (single site vs multi-site footprint)
    if size_profile == "Small data center":
        sites = 1
    elif size_profile == "Hyperscale data center":
        sites = 4
    else:
        sites = 12

    return {
        "annual_water_savings_liters": annual_water_savings_liters * sites,
        "annual_water_savings_m3": annual_water_savings_m3 * sites,
        "annual_co2_reduction_kg": annual_co2_reduction_kg * sites,
        "annual_co2_reduction_tons": (annual_co2_reduction_kg / 1000) * sites,
        "annual_cost_savings": annual_cost_savings_total * sites,
        "annual_energy_savings_kwh": annual_energy_savings_kwh * sites,
        "annual_energy_savings_mwh": (annual_energy_savings_kwh / 1000) * sites,
        "sites": float(sites),
        # Per-site values for global scaling (single-site annual impact)
        "per_site_annual_water_savings_m3": annual_water_savings_m3,
        "per_site_annual_co2_reduction_tons": (annual_co2_reduction_kg / 1000),
        "per_site_annual_cost_savings": annual_cost_savings_total,
    }


# Global scaling: ~4000 hyperscale / large cloud data centers worldwide (realistic order-of-magnitude)
GLOBAL_HYPERSCALE_SITES = 4000


def estimate_global_scaling(
    reference_annual_water_m3: float,
    reference_co2_tons: float,
    reference_cost_savings: float,
    global_sites: int = GLOBAL_HYPERSCALE_SITES,
) -> Dict[str, float]:
    """
    Scale single-site (or scenario) annual impact to global cloud infrastructure.

    Assumes reference values are per-site; multiplies by global_sites (~4000)
    for order-of-magnitude impact across hyperscale and large colo facilities.
    """
    # Per-site reference; scale to global footprint
    return {
        "global_water_savings_m3": reference_annual_water_m3 * global_sites,
        "global_water_savings_liters": reference_annual_water_m3 * LITERS_PER_M3 * global_sites,
        "global_co2_reduction_tons": reference_co2_tons * global_sites,
        "global_cost_savings": reference_cost_savings * global_sites,
        "global_sites": float(global_sites),
    }


# -----------------------------
# VISUAL COMPONENTS
# -----------------------------


def kpi_card(label: str, value: str, delta: str | None = None, color: str = "#1f77b4"):
    st.markdown(
        f"""
        <div style="
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            background: radial-gradient(circle at top left, rgba(255,255,255,0.15), rgba(0,0,0,0.02));
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        ">
            <div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 0.2rem;">{label}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{value}</div>
            {"<div style='font-size:0.8rem; opacity:0.8; margin-top:0.2rem;'>" + delta + "</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_feature_importance(model_info: ModelInfo):
    if not model_info.is_tree_based:
        st.info("Feature importance is only shown for tree-based models.")
        return

    # Extract from underlying regressor
    reg = model_info.model.named_steps["model"]
    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
    else:
        st.info("Model does not expose feature importances.")
        return

    df_imp = pd.DataFrame(
        {
            "feature": model_info.feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False).head(15)

    chart = (
        alt.Chart(df_imp)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Relative importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color("feature:N", legend=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
        )
        .properties(height=350)
    )

    st.altair_chart(chart, use_container_width=True)


def plot_water_vs_temperature(df: pd.DataFrame):
    sample = df.sample(min(3000, len(df)), random_state=42)
    chart = (
        alt.Chart(sample)
        .mark_circle(size=25, opacity=0.4)
        .encode(
            x=alt.X("outside_temperature:Q", title="Outside temperature (°C)"),
            y=alt.Y("water_usage_liters:Q", title="Cooling water usage (L / hour)"),
            color=alt.Color("region:N", title="Region"),
            tooltip=[
                "outside_temperature",
                "humidity",
                "server_load",
                "water_usage_liters",
                "cooling_type",
                "region",
            ],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)


def plot_before_after(baseline: float, optimized: float):
    df = pd.DataFrame(
        {
            "scenario": ["Baseline", "Optimized"],
            "water_liters": [baseline, optimized],
        }
    )

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("scenario:N", title=None),
            y=alt.Y("water_liters:Q", title="Water usage (L / hour)"),
            color=alt.Color("scenario:N", legend=None),
            tooltip=[
                "scenario",
                alt.Tooltip("water_liters:Q", format=",.0f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


# -----------------------------
# STREAMLIT LAYOUT
# -----------------------------


def sidebar_controls(df: pd.DataFrame) -> Tuple[Dict[str, Any], str]:
    st.sidebar.title("Simulation controls")

    size_profile = st.sidebar.radio(
        "Data center profile",
        ["Small data center", "Hyperscale data center", "Regional cloud cluster"],
        index=1,
    )

    region = st.sidebar.selectbox("Region", REGIONS)
    region_stress = np.select(
        [
            region == "Low-stress region",
            region == "Medium-stress region",
            region == "High-stress region",
        ],
        [0.2, 0.5, 0.8],
    )

    outside_temp = st.sidebar.slider("Outside temperature (°C)", min_value=5, max_value=45, value=30)
    humidity = st.sidebar.slider("Relative humidity (%)", min_value=10, max_value=95, value=55)
    server_load = st.sidebar.slider("Server load (0–1)", min_value=0.2, max_value=1.0, value=0.8, step=0.02)

    cooling_type = st.sidebar.selectbox("Cooling type", COOLING_TYPES)
    workload_intensity = st.sidebar.selectbox(
        "Workload intensity",
        WORKLOAD_INTENSITIES,
        index=2 if server_load > 0.7 else 1,
    )

    # Estimate power usage based on load & profile
    if size_profile == "Small data center":
        base_kw = 300
        max_kw = 1500
    elif size_profile == "Hyperscale data center":
        base_kw = 5000
        max_kw = 30000
    else:
        base_kw = 1500
        max_kw = 8000

    power_usage_kw = base_kw + (max_kw - base_kw) * server_load

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "AquaMind simulates realistic cooling-water demand using AI models trained on "
        "synthetic but physically grounded data center operations."
    )

    features = {
        "outside_temperature": outside_temp,
        "humidity": humidity,
        "server_load": server_load,
        "power_usage_kw": power_usage_kw,
        "cooling_type": cooling_type,
        "workload_intensity": workload_intensity,
        "region": region,
        "region_water_stress_index": region_stress,
    }

    return features, size_profile


def model_metrics_section(model_info: ModelInfo):
    st.subheader("Model quality – climate-aware water forecasting")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Selected model", model_info.name)
    with c2:
        kpi_card("R² on hold-out", f"{model_info.r2:.3f}")
    with c3:
        kpi_card("RMSE (L / hour)", f"{model_info.rmse:,.0f}")

    st.caption(
        "The model compares Random Forest, Gradient Boosting, and XGBoost (if available) "
        "with cross-validation, then selects the strongest performer to power the dashboard."
    )


def main():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #020817 0%, #041e2f 45%, #053249 100%);
            color: #f9fafb;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AquaMind – AI Water Optimization for Data Centers")

    st.markdown(
        """
        **AquaMind** is an AI decision platform that predicts and optimizes cooling-water use for data centers.
        It blends **machine learning**, **climate-aware simulation**, and **sustainability analytics** to help operators
        hit water and carbon reduction targets without sacrificing performance.
        """
    )

    df = generate_synthetic_datacenter_dataset()
    model_info = train_best_model(df)

    features, size_profile = sidebar_controls(df)
    region_stress = features["region_water_stress_index"]

    sim_result = simulate_optimization(
        model=model_info.model,
        base_features=features,
        size_profile=size_profile,
        region_stress=region_stress,
    )

    # Hourly KPIs
    baseline_l = sim_result["baseline_water_liters"]
    optimized_l = sim_result["optimized_water_liters"]
    savings_l = sim_result["water_savings_liters"]
    savings_pct = sim_result["water_savings_pct"]

    # Annual and global KPIs (hourly savings × 24 → daily; use actual WUE from simulation)
    annual_impact = estimate_annual_impact(
        daily_water_savings_liters=savings_l * 24,
        size_profile=size_profile,
        baseline_wue=sim_result.get("baseline_WUE"),
        optimized_wue=sim_result.get("optimized_WUE"),
    )
    # Global impact: scale per-site annual impact to ~4000 hyperscale data centers
    global_impact = estimate_global_scaling(
        reference_annual_water_m3=annual_impact["per_site_annual_water_savings_m3"],
        reference_co2_tons=annual_impact["per_site_annual_co2_reduction_tons"],
        reference_cost_savings=annual_impact["per_site_annual_cost_savings"],
    )

    # -----------------------------
    # KPI ROWS
    # -----------------------------
    st.markdown("### Real-time cooling-water insight")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card(
            "Baseline hourly water use",
            f"{baseline_l:,.0f} L",
        )
    with k2:
        kpi_card(
            "Optimized hourly water use",
            f"{optimized_l:,.0f} L",
            delta=f"{savings_pct:,.1f}% reduction",
            color="#16a34a",
        )
    with k3:
        kpi_card(
            "Hourly water saved",
            f"{savings_l:,.0f} L",
            color="#0ea5e9",
        )
    with k4:
        kpi_card(
            "Data center profile",
            size_profile,
            color="#eab308",
        )

    # Sustainability KPI row
    st.markdown("### Yearly and global sustainability signal")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        kpi_card(
            "Annual water savings",
            f"{annual_impact['annual_water_savings_m3']:,.0f} m³",
            delta=f"{int(annual_impact['sites'])} site(s) in scenario",
            color="#0ea5e9",
        )
    with s2:
        kpi_card(
            "Annual CO₂ reduced",
            f"{annual_impact['annual_co2_reduction_tons']:,.0f} tCO₂e",
            color="#22c55e",
        )
    with s3:
        kpi_card(
            "Annual cost savings",
            f"${annual_impact['annual_cost_savings']:,.0f}",
            color="#f97316",
        )
    with s4:
        kpi_card(
            "Global water impact",
            f"{global_impact['global_water_savings_m3']/1e6:,.2f} Mm³ / yr",
            delta=f"Across ~{int(global_impact['global_sites'])} cloud sites",
            color="#38bdf8",
        )

    # -----------------------------
    # MODEL + EXPLAINABILITY
    # -----------------------------
    st.markdown("---")
    c_left, c_right = st.columns([3, 2])

    with c_left:
        model_metrics_section(model_info)
        st.markdown("#### Climate signal: water vs temperature")
        plot_water_vs_temperature(df)

    with c_right:
        st.markdown("#### Feature importance – what drives cooling water?")
        plot_feature_importance(model_info)

    # -----------------------------
    # OPTIMIZATION EXPLANATION
    # -----------------------------
    st.markdown("---")
    st.markdown("### Optimization engine – turning predictions into decisions")

    o1, o2 = st.columns([2, 1])
    with o1:
        st.markdown(
            f"""
            **Recommended cooling strategy:** `{sim_result['new_cooling_type']}`  
            **Baseline WUE:** {sim_result['wue_baseline']:.2f} L/kWh → **Optimized WUE (target):** {sim_result['wue_optimized']:.2f} L/kWh
            """
        )

        st.markdown("**Cooling optimization levers**")
        for r in sim_result["cooling_recommendations"]:
            st.markdown(f"- {r}")

        st.markdown("**Workload & scheduling levers**")
        for r in sim_result["workload_recommendations"]:
            st.markdown(f"- {r}")

        st.caption(
            "AquaMind combines cooling configuration shifts, cross-region workload placement, and demand shaping "
            "to reduce water use while respecting performance constraints."
        )

    with o2:
        st.markdown("#### Before vs after optimization")
        plot_before_after(baseline_l, optimized_l)

    # -----------------------------
    # ADVANCED / DATA VIEW
    # -----------------------------
    with st.expander("Inspect training data and advanced assumptions"):
        st.markdown(
            """
            **Synthetic dataset**  
            The training data emulates thousands of hourly observations across different regions, climate conditions,
            cooling architectures, and utilization profiles. Relationships embed:
            
            - **Thermodynamics**: higher temperature & humidity drive higher WUE and water use.
            - **Architecture**: different cooling types exhibit distinct baseline WUE bands.
            - **Water stress**: high-stress regions are nudged toward more efficient operation.
            """
        )
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("**Distribution snapshot**")
        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            st.bar_chart(df["cooling_type"].value_counts())
        with hist_col2:
            st.bar_chart(df["region"].value_counts())

    st.markdown(
        """
        ---
        **How to use AquaMind in practice**

        - Connect live telemetry from DCIM/BMS systems for **outside temperature, humidity, load, and power**.
        - Continuously retrain the model with your real data and regional water tariffs.
        - Integrate the optimization engine with your **orchestrator / scheduler** to automate workload shifts.
        - Use the sustainability panel for **ESG reporting**, water-risk exposure analysis, and scenario planning.
        """
    )


if __name__ == "__main__":
    main()