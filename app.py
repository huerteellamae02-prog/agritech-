import streamlit as st
import pandas as pd

def predict_crop(features):
    """Simple rule-based crop prediction based on soil parameters."""
    try:
        n = features.get('N', 0)
        p = features.get('P', 0)
        k = features.get('K', 0)
        ph = features.get('pH', 7.0)

        # Simple rule-based prediction
        if 5.0 <= ph <= 6.5:
            if n > 80 and k > 40:
                return "rice"
            elif p > 60:
                return "potato"
            else:
                return "soybean"
        elif 6.5 < ph <= 7.5:
            if n > 100:
                return "maize"
            elif k > 80:
                return "cotton"
            else:
                return "wheat"
        elif ph > 7.5:
            return "sorghum"
        else:
            return "potato"  # Most tolerant to acidic soil
            
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "wheat"  # Safe default

st.set_page_config(page_title="Agritech AI", layout="wide")
st.title("🌱 Agritech AI Platform")
st.caption("Empowering agriculture with AI-driven insights.")

# --- Simple knowledge base for crop recommendations ---
PH_RANGES = {
    "rice": (5.5, 6.5),
    "maize": (5.8, 7.0),
    "wheat": (6.0, 7.5),
    "soybean": (6.0, 7.0), 
    "potato": (5.0, 6.5),
    "tomato": (6.0, 6.8),
    "sorghum": (5.8, 7.5),
    "cotton": (5.8, 8.0),
}

SPECIFIC_ADVICE = {
    "rice": ["Prefers puddled fields and consistent water supply", "Apply nitrogen in splits (basal, tillering, panicle initiation)"],
    "maize": ["Ensure adequate phosphorus at planting for root vigor", "Nitrogen top-dress at V6–V8 stages"],
    "wheat": ["Avoid late nitrogen; apply before heading", "Good drainage is important"],
    "soybean": ["Inoculate seeds with Rhizobium in low-organic soils", "Avoid excessive nitrogen fertilization"],
    "potato": ["Maintain slightly acidic soil; avoid fresh manure", "Hilling is beneficial to tuber quality"],
    "tomato": ["Requires steady irrigation; avoid waterlogging", "Provide staking/trellising and pest monitoring"],
    "sorghum": ["Drought tolerant; suitable for marginal rainfall", "Monitor for stem borers and birds"],
    "cotton": ["Warm-season crop; ensure long frost-free period", "Balanced K is critical for fiber quality"],
}

def nutrient_advice(n: int, p: int, k: int) -> list[str]:
    tips: list[str] = []
    if n < 40:
        tips.append("Nitrogen is low: consider urea or ammonium sulfate (split applications)")
    elif n > 100:
        tips.append("Nitrogen is high: reduce N to avoid lodging and leaching")
    if p < 30:
        tips.append("Phosphorus is low: apply DAP/TSP at planting, banded near seed")
    if k < 30:
        tips.append("Potassium is low: apply MOP (KCl) to improve stress tolerance")
    return tips

def recommend_alternatives(ph_value: float, predicted: str) -> list[str]:
    # Suggest up to 2 alternative crops by closest pH fit
    distances = []
    for crop, (lo, hi) in PH_RANGES.items():
        mid = (lo + hi) / 2.0
        distances.append((abs(ph_value - mid), crop))
    distances.sort()
    alts = [c for _, c in distances if c.lower() != str(predicted).lower()]
    return alts[:2]

# --- Simple authentication ---
def bootstrap_users():
    # Initialize a mutable users store in session_state from secrets or defaults
    if "users" in st.session_state and isinstance(st.session_state.users, dict):
        return
    users_from_secrets = {}
    try:
        users_from_secrets = st.secrets.get("auth", {}).get("users", {})
    except Exception:
        users_from_secrets = {}
    if not isinstance(users_from_secrets, dict) or not users_from_secrets:
        users_from_secrets = {"admin": "admin123", "farmer": "farm@2025"}
    st.session_state.users = dict(users_from_secrets)


def get_credentials():
    bootstrap_users()
    return st.session_state.users


def require_login():
    if "auth_user" in st.session_state:
        return True

    creds = get_credentials()

    # Centered login card in the main area
    st.markdown("\n\n")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.subheader("Sign in")
        with st.form("login_form", clear_on_submit=False, border=True):
            username = st.text_input("Username", placeholder="yourname")
            show_pw = st.checkbox("Show password", value=False)
            password = st.text_input("Password", type="default" if show_pw else "password", placeholder="••••••••")
            remember = st.checkbox("Remember me", value=True)
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
                return False
            # Exact match against provided credentials
            if username in creds and creds[username] == password:
                st.session_state.auth_user = username
                if remember:
                    st.session_state.remember_me = True
                st.success("Logged in successfully")
                st.rerun()
                return True
            st.error("Invalid username or password.")
            return False

        # Simple create-account form
        with st.expander("Create account"):
            with st.form("register_form", clear_on_submit=True, border=False):
                r_user = st.text_input("New username")
                r_pass1 = st.text_input("New password", type="password")
                r_pass2 = st.text_input("Confirm password", type="password")
                r_submit = st.form_submit_button("Register", use_container_width=True)
            if r_submit:
                if not r_user or not r_pass1 or not r_pass2:
                    st.warning("All fields are required.")
                elif r_pass1 != r_pass2:
                    st.warning("Passwords do not match.")
                elif r_user in st.session_state.users:
                    st.warning("Username already exists.")
                else:
                    st.session_state.users[r_user] = r_pass1
                    st.success("Account created. You can now sign in.")

    st.stop()


if not require_login():
    st.stop()

with st.sidebar:
    st.markdown(f"Signed in as: **{st.session_state.get('auth_user', '')}**")
    # Admin-only simple user management (in-memory for this session)
    if st.session_state.get("auth_user") == "admin":
        with st.expander("User management"):
            with st.form("add_user_form", clear_on_submit=True):
                new_user = st.text_input("New username")
                new_pass = st.text_input("New password", type="password")
                add_sub = st.form_submit_button("Add user")
            if add_sub:
                if not new_user or not new_pass:
                    st.warning("Provide username and password.")
                elif new_user in st.session_state.users:
                    st.warning("User already exists.")
                else:
                    st.session_state.users[new_user] = new_pass
                    st.success(f"User '{new_user}' added for this session.")

            if st.session_state.users:
                del_user = st.selectbox(
                    "Remove user", [u for u in st.session_state.users.keys()], index=0
                )
                if st.button("Delete selected user"):
                    if del_user == "admin":
                        st.warning("Cannot delete admin.")
                    else:
                        st.session_state.users.pop(del_user, None)
                        st.success(f"User '{del_user}' removed.")
    if st.button("Log out"):
        st.session_state.pop("auth_user", None)
        st.rerun()

tab_overview, tab_advisory, tab_weather, tab_notebook = st.tabs([
    "Overview",
    "Crop Advisory",
    "Weather Insights",
    "Field Notebook",
])

with tab_overview:
    st.subheader("Welcome")
    st.write(
        "Use the tabs to predict suitable crops, review weather conditions, and keep a field notebook."
    )
    st.markdown(
        "- Provide soil nutrients and pH in Crop Advisory to get an AI recommendation.\n"
        "- Use Weather Insights to quickly assess current conditions.\n"
        "- Log observations, actions, and costs in Field Notebook and export to CSV."
    )

with tab_advisory:
    st.subheader("Crop Advisory")
    col1, col2 = st.columns(2)
    with col1:
        nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
        phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=140, value=30)
    with col2:
        potassium = st.number_input("Potassium (K)", min_value=0, max_value=140, value=40)
        ph = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1)

    if st.button("Predict Crop", type="primary"):
        try:
            features = {
                "N": nitrogen,
                "P": phosphorus,
                "K": potassium,
                "pH": ph,
            }
            crop = predict_crop(features)
            if not crop:
                st.error("No crop prediction available")
            else:
                st.success(f"Recommended Crop: {crop}")
                # Contextual recommendations
                crop_key = str(crop).lower()
                ph_range = PH_RANGES.get(crop_key)
                if ph_range:
                    st.caption(f"Ideal pH for {crop}: {ph_range[0]}–{ph_range[1]}")
                extra = SPECIFIC_ADVICE.get(crop_key, [])
                tips = nutrient_advice(nitrogen, phosphorus, potassium)
                if extra or tips:
                    st.markdown("**Recommendations:**")
                    for t in extra + tips:
                        st.markdown(f"- {t}")
                # Alternatives by pH fit
                alts = recommend_alternatives(ph, crop_key)
                if alts:
                    st.markdown("**Also consider:** " + ", ".join(a.title() for a in alts))
            st.caption("Model source: `model/predict.py`")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

with tab_weather:
    st.subheader("Weather Insights")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        temperature_c = st.number_input("Air Temp (°C)", min_value=-20.0, max_value=60.0, value=28.0, step=0.5)
    with c2:
        humidity = st.number_input("Relative Humidity (%)", min_value=0, max_value=100, value=60, step=1)
    with c3:
        rainfall_mm = st.number_input("Rainfall (mm/24h)", min_value=0.0, max_value=500.0, value=5.0, step=0.5)
    with c4:
        wind_ms = st.number_input("Wind (m/s)", min_value=0.0, max_value=30.0, value=2.0, step=0.1)

    # Optional advanced inputs
    with st.expander("Advanced inputs"):
        colA, colB, colC = st.columns(3)
        with colA:
            tmin = st.number_input("Tmin (°C)", min_value=-30.0, max_value=50.0, value=max(-5.0, temperature_c - 8.0), step=0.5)
        with colB:
            tmax = st.number_input("Tmax (°C)", min_value=-10.0, max_value=60.0, value=min(50.0, temperature_c + 8.0), step=0.5)
        with colC:
            solar_mj = st.number_input("Sun (MJ/m²/day)", min_value=0.0, max_value=35.0, value=18.0, step=0.5)

    # Calculations
    import math
    # Dew point (Magnus formula)
    a, b = 17.27, 237.7
    gamma = (a * temperature_c) / (b + temperature_c) + math.log(max(1e-6, humidity/100.0))
    dew_point = (b * gamma) / (a - gamma)

    # Saturation vapor pressure (kPa) and VPD (kPa)
    es = 0.6108 * math.exp(17.27 * temperature_c / (temperature_c + 237.3))
    ea = es * (humidity / 100.0)
    vpd = max(0.0, es - ea)

    # Simple heat index (approx) for shade conditions
    heat_index = temperature_c + (humidity/100.0) * 2.0

    # Very simple evapotranspiration proxy (not FAO-56):
    # ETp_mm approximated from VPD, wind, and solar radiation
    etp_mm = max(0.0, 0.35 * vpd * (1.0 + 0.1 * wind_ms) + 0.05 * solar_mj)

    # Heuristics for operations
    good_temp = 15 <= temperature_c <= 32
    good_rh = 40 <= humidity <= 80
    low_rain = rainfall_mm < 5
    low_wind_for_spray = wind_ms <= 3.0

    ops_ok = good_temp and good_rh and low_rain and low_wind_for_spray
    if ops_ok:
        st.success("Good window for field operations (tillage/planting/spraying).")
    else:
        st.warning("Caution: suboptimal conditions for some operations.")

    # Irrigation suggestion
    irrigation_need = 0.0
    if (vpd > 1.2 or temperature_c > 32 or humidity < 40) and rainfall_mm < 5:
        # Target irrigation scaled by ETp, floor 5mm, cap 25mm
        irrigation_need = min(25.0, max(5.0, round(etp_mm + 5, 1)))
        st.info(f"Irrigation suggested: ~{irrigation_need} mm within 24h.")

    # Disease risk (very simple fungal risk proxy)
    fungal_risk = "low"
    if humidity >= 85 and 18 <= temperature_c <= 26:
        fungal_risk = "high"
    elif humidity >= 75 and 16 <= temperature_c <= 30:
        fungal_risk = "moderate"

    # Wind advisory for spraying
    if not low_wind_for_spray:
        st.error("Avoid pesticide spraying now: wind too strong.")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Heat Index (°C)", f"{heat_index:.1f}")
    with m2:
        st.metric("Dew Point (°C)", f"{dew_point:.1f}")
    with m3:
        st.metric("VPD (kPa)", f"{vpd:.2f}")
    with m4:
        st.metric("ET proxy (mm/day)", f"{etp_mm:.1f}")

    # Quick badges
    st.caption(
        (
            "Spray: OK" if low_wind_for_spray and low_rain else "Spray: NO" 
        ) + "  |  " +
        (f"Fungal risk: {fungal_risk}") + "  |  " +
        ("Irrigation: needed" if irrigation_need > 0 else "Irrigation: not urgent")
    )

with tab_notebook:
    st.subheader("Field Notebook")

    if "notebook_df" not in st.session_state:
        st.session_state.notebook_df = pd.DataFrame(
            [
                {
                    "Date": pd.NaT,
                    "Field": "",
                    "Crop": "",
                    "Operation": "Observation",
                    "Observation": "",
                    "Action": "",
                    "Quantity": 0.0,
                    "Unit": "",
                    "Cost": 0.0,
                    "Revenue": 0.0,
                }
            ]
        )

    # Ensure Date column is datetime64[ns] for compatibility with DateColumn
    if "Date" in st.session_state.notebook_df.columns:
        st.session_state.notebook_df["Date"] = pd.to_datetime(
            st.session_state.notebook_df["Date"], errors="coerce"
        )

    # Quick add form
    with st.expander("Quick add entry"):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            q_date = st.date_input("Date", value=None, format="YYYY-MM-DD")
        with f2:
            q_field = st.text_input("Field", placeholder="Plot A")
        with f3:
            q_crop = st.text_input("Crop", placeholder="Maize")
        with f4:
            q_op = st.selectbox(
                "Operation",
                [
                    "Observation",
                    "Tillage",
                    "Planting",
                    "Fertilizer",
                    "Irrigation",
                    "Pesticide",
                    "Harvest",
                    "Sale",
                    "Other",
                ],
                index=0,
            )

        o1, o2 = st.columns(2)
        with o1:
            q_obs = st.text_input("Observation/Notes")
            q_qty = st.number_input("Quantity", min_value=0.0, value=0.0, step=0.1)
        with o2:
            q_action = st.text_input("Action taken")
            q_unit = st.text_input("Unit", placeholder="kg, L, hr, ...")

        c1, c2, c3 = st.columns(3)
        with c1:
            q_cost = st.number_input("Cost", min_value=0.0, value=0.0, step=0.1)
        with c2:
            q_rev = st.number_input("Revenue", min_value=0.0, value=0.0, step=0.1)
        with c3:
            if st.button("Add entry", type="primary"):
                new_row = {
                    "Date": pd.to_datetime(q_date) if q_date else pd.NaT,
                    "Field": q_field,
                    "Crop": q_crop,
                    "Operation": q_op,
                    "Observation": q_obs,
                    "Action": q_action,
                    "Quantity": q_qty,
                    "Unit": q_unit,
                    "Cost": float(q_cost),
                    "Revenue": float(q_rev),
                }
                st.session_state.notebook_df = pd.concat([
                    st.session_state.notebook_df,
                    pd.DataFrame([new_row])
                ], ignore_index=True)
                st.success("Entry added.")

    # Filters
    with st.expander("Filters"):
        fl1, fl2, fl3, fl4 = st.columns(4)
        with fl1:
            start_date = st.date_input("Start date", value=None, format="YYYY-MM-DD", key="nb_start")
        with fl2:
            end_date = st.date_input("End date", value=None, format="YYYY-MM-DD", key="nb_end")
        with fl3:
            field_filter = st.text_input("Field contains", key="nb_field")
        with fl4:
            crop_filter = st.text_input("Crop contains", key="nb_crop")

    df_view = st.session_state.notebook_df.copy()
    if start_date:
        df_view = df_view[df_view["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df_view = df_view[df_view["Date"] <= pd.to_datetime(end_date)]
    if field_filter:
        df_view = df_view[df_view["Field"].str.contains(field_filter, case=False, na=False)]
    if crop_filter:
        df_view = df_view[df_view["Crop"].str.contains(crop_filter, case=False, na=False)]

    edited_df = st.data_editor(
        df_view,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Field": st.column_config.TextColumn("Field"),
            "Crop": st.column_config.TextColumn("Crop"),
            "Operation": st.column_config.SelectboxColumn("Operation", options=["Observation", "Tillage", "Planting", "Fertilizer", "Irrigation", "Pesticide", "Harvest", "Sale", "Other"]),
            "Quantity": st.column_config.NumberColumn("Quantity"),
            "Unit": st.column_config.TextColumn("Unit"),
            "Cost": st.column_config.NumberColumn("Cost", help="Input in your local currency"),
            "Revenue": st.column_config.NumberColumn("Revenue", help="Input in your local currency"),
        },
        hide_index=True,
        key="notebook_editor",
    )

    # Merge edits back into main dataframe (respecting filters)
    if len(edited_df) == len(df_view):
        # Align indices to original for replacement
        st.session_state.notebook_df.loc[df_view.index] = edited_df.values
    else:
        st.session_state.notebook_df = edited_df

    totals = {
        "Total Cost": float(pd.to_numeric(st.session_state.notebook_df.get("Cost", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "Total Revenue": float(pd.to_numeric(st.session_state.notebook_df.get("Revenue", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
    }
    st.write(
        f"Net: {round(totals['Total Revenue'] - totals['Total Cost'], 2)}  |  "
        f"Cost: {round(totals['Total Cost'], 2)}  |  Revenue: {round(totals['Total Revenue'], 2)}"
    )

    # Aggregations
    with st.expander("Summaries"):
        g1, g2 = st.columns(2)
        with g1:
            by_field = st.session_state.notebook_df.groupby("Field", dropna=False)[["Cost", "Revenue"]].sum(numeric_only=True)
            by_field["Net"] = by_field["Revenue"] - by_field["Cost"]
            st.write("By Field")
            st.dataframe(by_field)
        with g2:
            by_op = st.session_state.notebook_df.groupby("Operation", dropna=False)[["Cost", "Revenue"]].sum(numeric_only=True)
            by_op["Net"] = by_op["Revenue"] - by_op["Cost"]
            st.write("By Operation")
            st.dataframe(by_op)

    # Simple monthly cash flow chart
    with st.expander("Cash flow (monthly)"):
        df_chart = st.session_state.notebook_df.copy()
        if "Date" in df_chart.columns:
            df_chart["Month"] = pd.to_datetime(df_chart["Date"], errors="coerce").dt.to_period("M").astype(str)
            monthly = df_chart.groupby("Month")[['Cost','Revenue']].sum(numeric_only=True)
            monthly["Net"] = monthly["Revenue"] - monthly["Cost"]
            st.bar_chart(monthly, use_container_width=True)

    # Import/Export & actions
    a1, a2, a3 = st.columns([1,1,1])
    csv_bytes = st.session_state.notebook_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="field_notebook.csv",
        mime="text/csv",
    )

    up = st.file_uploader("Import CSV", type=["csv"], accept_multiple_files=False, key="nb_upload")
    if up is not None:
        try:
            imported = pd.read_csv(up)
            if "Date" in imported.columns:
                imported["Date"] = pd.to_datetime(imported["Date"], errors="coerce")
            st.session_state.notebook_df = imported
            st.success("CSV imported.")
        except Exception as e:
            st.error(f"Failed to import CSV: {e}")

    colsave, colclear = st.columns([1,1])
    with colsave:
        if st.button("Save to local CSV"):
            try:
                st.session_state.notebook_df.to_csv("field_notebook.csv", index=False)
                st.success("Saved to field_notebook.csv")
            except Exception as e:
                st.error(f"Save failed: {e}")
    with colclear:
        if st.button("Clear notebook"):
            st.session_state.notebook_df = st.session_state.notebook_df.iloc[0:0]
            st.warning("Notebook cleared for this session.")

st.info("This demo uses manual inputs. Connect sensors/APIs and enhance `model/predict.py` for production.")
