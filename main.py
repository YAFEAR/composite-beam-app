import streamlit as st
import matplotlib.pyplot as plt
import json
import os

from helpers import (
    run_fem_simulation,
    plot_cross_section,
    plot_laminate_stackup,
    plot_laminate_stackup_compare,
)
from manual_correction import apply_manual_correction
from laminate_mechanics import plot_laminate_stackup_mechanics_compare
from beam_qlearning_case1 import update_result_with_mass_breakdown

st.set_page_config(page_title="Beam Design UI", layout="wide")
st.title("🛠️ Composite Beam Design Optimization")

# -----------------------------------------------------------------------------
# Mapping keeps the internal integer IDs (1/2/3) intact while showing friendly
# names in the GUI. Down‑stream modules that key on the case number therefore
# remain untouched.
# -----------------------------------------------------------------------------
LABELS = {1: "I‑beam", 2: "Filled I‑beam", 3: "Rectangular foam‑core beam"}

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
        "Maximum design force (N)", value=50_000.0, min_value=0.0, step=100.0
    )

    fem_resolution = st.slider("FEM Mesh Resolution", 10, 100, 50)

    st.markdown("---")

    if st.button("Run Q-learning Optimization"):
        os.environ["FORCE_N"] = str(force_max)
        os.system("python beam_qlearning_case1.py")
        st.success("Optimization complete. Results saved to opt_result.json")

    # Manual correction -------------------------------------------------------
    manual_inputs: dict = {}
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
            manual_inputs["web_ply_count"] = st.number_input("Web Ply Count", 1, 20, 3)
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
            corrected_fem = run_fem_simulation(
                corrected_result, mesh_resolution=fem_resolution
            )

    with tabs[1]:
        st.header("Cross Section Comparison")
        col1, col2 = st.columns(2)

        # Optimized -----------------------------------------------------------
        with col1:
            st.subheader("Optimized Cross Section")
            fig_opt = plot_cross_section(optimized_result)
            st.pyplot(fig_opt)
            st.markdown(f"**h** = {optimized_result['h']} mm")
            st.markdown(f"**b** = {optimized_result['b']} mm")
            st.markdown(f"**t_flange** = {optimized_result['t_flange']} mm")
            st.markdown(f"**t_web** = {optimized_result['t_web']} mm")
            st.markdown(f"**t_foam** = {optimized_result['t_foam']} mm")
            st.markdown(f"**Mass** = {optimized_result['mass']} g")
            st.markdown(
                f"**Max Deflection** = {optimized_result['max_deflection']} mm"
            )
            st.markdown("**Mass Breakdown:**")
            st.markdown(f"- Flange: {optimized_result['mass_flange']} g")
            st.markdown(f"- Core: {optimized_result['mass_core']} g")
            st.markdown(f"- Side Panels: {optimized_result['mass_side']} g")

        # Corrected -----------------------------------------------------------
        with col2:
            if corrected_result:
                st.subheader("Corrected Cross Section")
                fig_corr = plot_cross_section(corrected_result)
                st.pyplot(fig_corr)
                st.markdown(f"**h** = {corrected_result['h']} mm")
                st.markdown(f"**b** = {corrected_result['b']} mm")
                st.markdown(f"**t_flange** = {corrected_result['t_flange']} mm")
                st.markdown(f"**t_web** = {corrected_result['t_web']} mm")
                st.markdown(f"**t_foam** = {corrected_result['t_foam']} mm")
                st.markdown(f"**Mass** = {corrected_result['mass']} g")
                st.markdown(
                    f"**Max Deflection** = {corrected_result['max_deflection']} mm"
                )
                st.markdown("**Mass Breakdown:**")
                st.markdown(f"- Flange: {corrected_result['mass_flange']} g")
                st.markdown(f"- Core: {corrected_result['mass_core']} g")
                st.markdown(f"- Side Panels: {corrected_result['mass_side']} g")

    # =============================================================================
    # 3 ─ PLY PLOT TAB
    # =============================================================================
    with tabs[2]:
        st.header("Laminate Stackups")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Optimized Flange Layup")
            fig1 = plot_laminate_stackup(optimized_result["flange_layup"])
            st.pyplot(fig1)

            st.subheader("Optimized Web Layup")
            fig2 = plot_laminate_stackup(optimized_result["web_layup"])
            st.pyplot(fig2)

        if corrected_result:
            with col2:
                st.subheader("Corrected Flange Layup")
                fig3 = plot_laminate_stackup(corrected_result["flange_layup"])
                st.pyplot(fig3)

                st.subheader("Corrected Web Layup")
                fig4 = plot_laminate_stackup(corrected_result["web_layup"])
                st.pyplot(fig4)

            st.subheader("A-matrix / E-property Comparison")
            col3, col4 = st.columns(2)

            with col3:
                st.markdown("**Flange Layup Stiffness**")
                fig_flange = plot_laminate_stackup_mechanics_compare(
                    opt_layup=optimized_result["flange_layup"],
                    corr_layup=corrected_result["flange_layup"],
                    labels=["Optimized", "Corrected"],
                    title="Flange Stiffness Comparison",
                )
                st.pyplot(fig_flange)

            with col4:
                st.markdown("**Side Panel (Web) Layup Stiffness**")
                fig_web = plot_laminate_stackup_mechanics_compare(
                    opt_layup=optimized_result["web_layup"],
                    corr_layup=corrected_result["web_layup"],
                    labels=["Optimized", "Corrected"],
                    title="Web Stiffness Comparison",
                )
                st.pyplot(fig_web)

    # =============================================================================
    # 4 ─ FEM TAB
    # =============================================================================
    with tabs[3]:
        st.header("FEM Deflection Comparison")
        fem_result = run_fem_simulation(
            optimized
