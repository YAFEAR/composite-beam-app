import streamlit as st
import matplotlib.pyplot as plt
import json
import os

from helpers import (
    run_fem_simulation,
    plot_cross_section,
    plot_laminate_stackup,
)
from manual_correction import apply_manual_correction
from laminate_mechanics import plot_laminate_stackup_mechanics_compare
from beam_qlearning_case1 import update_result_with_mass_breakdown

st.set_page_config(page_title="Beam Design UI", layout="wide")
st.title("🛠️ Composite Beam Design Optimization")

# -----------------------------------------------------------------------------
# Friendly labels for case IDs --------------------------------------------------
# -----------------------------------------------------------------------------
LABELS = {
    1: "I‑beam",
    2: "Filled I‑beam",
    3: "Rectangular foam‑core beam",
}

# -----------------------------------------------------------------------------
# Tabs -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

tabs = st.tabs(["Input", "Geometry", "Ply Plots", "FEM"])

# =============================================================================
# 1 ─ INPUT TAB
# =============================================================================
with tabs[0]:
    st.header("Design Constraints")

    case = st.selectbox("Beam type", options=[1, 2, 3], format_func=LABELS.get)
    beam_length = st.number_input("Beam Length (mm)", value=450.0, min_value=100.0)
    h_max = st.number_input("Max Height H (mm)", value=55.0, min_value=10.0)
    b_max = st.number_input("Max Width B (mm)", value=40.0, min_value=5.0)
    mass_limit = st.number_input("Max Mass (g)", value=170.0, min_value=10.0)
    defl_limit = st.number_input("Max Deflection (mm)", value=3.8, min_value=0.1)

    force_max = st.number_input(
        "Maximum design force (N)", value=5_000.0, min_value=0.0, step=100.0
    )
    fem_resolution = st.slider("FEM Mesh Resolution", 50, 500, 100)

    st.markdown("---")

    if st.button("Run Q-learning Optimization"):
        os.environ["FORCE_N"] = str(force_max)  # pass to optimiser
        os.system("python beam_qlearning_case1.py")
        st.success("Optimization complete. Results saved to opt_result.json")

    # ── Manual correction inputs -------------------------------------------
    manual_inputs = {}
    manual_enable = False
    if os.path.exists("opt_result.json"):
        with open("opt_result.json", "r") as f:
            optimized_result = json.load(f)

        optimized_result = update_result_with_mass_breakdown(
            optimized_result,
            h=optimized_result["h"],
            b=optimized_result["b"],
            t_flange=optimized_result["t_flange"],
            t_foam=optimized_result["t_foam"],
        )

        st.markdown("### ✍️ Manual Correction (optional)")
        manual_enable = st.checkbox("Enable Manual Correction")

        if manual_enable:
            manual_inputs["flange_orientation"] = st.text_input(
                "Flange Ply Orientations", "90,0,0,45,-45,0,0,90"
            )
            manual_inputs["web_orientation"] = st.text_input(
                "Web Ply Orientations", "45,90,-45"
            )
            manual_inputs["flange_ply_count"] = st.number_input(
                "Flange Ply Count", 1, 20, 8
            )
            manual_inputs["web_ply_count"] = st.number_input(
                "Web Ply Count", 1, 20, 3
            )
            manual_inputs["h"] = st.number_input(
                "Corrected Height h (mm)", value=optimized_result["h"]
            )
            manual_inputs["b"] = st.number_input(
                "Corrected Width b (mm)", value=optimized_result["b"]
            )
            manual_inputs["t_core"] = st.number_input(
                "Corrected Core Thickness t_core (mm)",
                value=optimized_result["t_foam"],
                min_value=1.0,
                max_value=50.0,
                step=0.5,
            )

# =============================================================================
# 2 ─ GEOMETRY TAB
# =============================================================================
corrected_result = None
corrected_fem = None

if os.path.exists("opt_result.json"):
    with open("opt_result.json", "r") as f:
        optimized_result = json.load(f)

    optimized_result = update_result_with_mass_breakdown(
        optimized_result,
        h=optimized_result["h"],
        b=optimized_result["b"],
        t_flange=optimized_result["t_flange"],
        t_foam=optimized_result["t_foam"],
    )

    if manual_inputs:
        corrected_result = apply_manual_correction(optimized_result, manual_inputs)
        if corrected_result:
            corrected_fem = run_fem_simulation(corrected_result, fem_resolution)

    # ── GEOMETRY TAB -------------------------------------------------------
    with tabs[1]:
        st.header("Cross Section Comparison")
        col1, col2 = st.columns(2)

        # Optimized ---------------------------------------------------------
        with col1:
            st.subheader("Optimized Cross Section")
            st.pyplot(plot_cross_section(optimized_result))

        # Corrected ---------------------------------------------------------
        with col2:
            if corrected_result:
                st.subheader("Corrected Cross Section")
                st.pyplot(plot_cross_section(corrected_result))

    # ── PLY PLOT TAB -------------------------------------------------------
    with tabs[2]:
        st.header("Laminate Stackups")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optimized Flange Layup")
            st.pyplot(plot_laminate_stackup(optimized_result["flange_layup"]))
            st.subheader("Optimized Web Layup")
            st.pyplot(plot_laminate_stackup(optimized_result["web_layup"]))
        if corrected_result:
            with col2:
                st.subheader("Corrected Flange Layup")
                st.pyplot(plot_laminate_stackup(corrected_result["flange_layup"]))
                st.subheader("Corrected Web Layup")
                st.pyplot(plot_laminate_stackup(corrected_result["web_layup"]))

            st.subheader("A‑matrix / E‑property Comparison")
            col3, col4 = st.columns(2)
            with col3:
                fig_f = plot_laminate_stackup_mechanics_compare(
                    optimized_result["flange_layup"],
                    corrected_result["flange_layup"],
                    ["Optimized", "Corrected"],
                    "Flange Stiffness Comparison",
                )
                st.pyplot(fig_f)
            with col4:
                fig_w = plot_laminate_stackup_mechanics_compare(
                    optimized_result["web_layup"],
                    corrected_result["web_layup"],
                    ["Optimized", "Corrected"],
                    "Web Stiffness Comparison",
                )
                st.pyplot(fig_w)

    # ── FEM TAB ------------------------------------------------------------
    with tabs[3]:
        st.header("FEM Deflection Comparison")
        fem_result = run_fem_simulation(optimized_result, fem_resolution)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(fem_result["x"], fem_result["u"], label="Optimized", linewidth=2)
        if corrected_fem:
            ax.plot(
                corrected_fem["x"],
                corrected_fem["u"],
                label="Corrected",
                linestyle="--",
                linewidth=2,
            )
        ax.set_xlabel("Beam Length [mm]")
        ax.set_ylabel("Deflection [mm]")
        ax.set_title("Deflection Curve Comparison")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
else:
    with tabs[1]:
        st.warning("No optimization result found. Please run the optimizer in the Input tab.")
