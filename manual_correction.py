"""Manual correction utilities

This module lets the user tweak the optimiser’s result and immediately see
updated performance metrics.  No core beam‑sizing logic is changed here – we
simply pass the corrected dimensions / lay‑ups through the same analytics
pipeline used elsewhere.
"""

import os
import matplotlib.pyplot as plt
from beam_analytics import calculate_beam_properties
from laminate_mechanics import (
    compute_laminate_A_matrix,
    plot_laminate_stackup_mechanics_compare,
)
from fem_solver import fem_beam_solver_correct
from cross_section import draw_cross_section

# -----------------------------------------------------------------------------
# Helper – pull design load from optimiser export or environment --------------
# -----------------------------------------------------------------------------

def _get_design_force(opt_result):
    """Return the force value that was used in optimisation (fallback 4000 N)."""
    # New field added by Streamlit after v2025‑05‑05; keep fallback for old JSONs
    if isinstance(opt_result, dict) and "F" in opt_result:
        return float(opt_result["F"])
    # Env variable set by UI before launching the optimiser
    return float(os.getenv("FORCE_N", "4000"))


# -----------------------------------------------------------------------------
# Streamlit‑friendly correction routine ---------------------------------------
# -----------------------------------------------------------------------------

def apply_manual_correction(opt_result, manual_inputs):
    """Return a corrected result dict based on user tweaks from the UI."""

    # ------------------------------------------------------------------
    # Use optimiser’s design load unless the UI has changed it further
    # ------------------------------------------------------------------
    F_design = _get_design_force(opt_result)

    t_core = manual_inputs.get("t_core", opt_result.get("t_foam", 5.0))
    h = manual_inputs.get("h", opt_result.get("h", 40.0))
    b = manual_inputs.get("b", opt_result.get("b", 20.0))

    flange_layup = [int(angle.strip()) for angle in manual_inputs["flange_orientation"].split(",")]
    web_layup = [int(angle.strip()) for angle in manual_inputs["web_orientation"].split(",")]

    flange_thickness = compute_laminate_A_matrix(flange_layup)["thickness"]
    web_thickness = compute_laminate_A_matrix(web_layup)["thickness"]

    props_corr = calculate_beam_properties(
        case_type=1,
        h=h,
        b=b,
        flange_layup=flange_layup,
        web_layup=web_layup,
        t_side=web_thickness,
        t_core_web=t_core,
        t_core_side=web_thickness,
        F=F_design,  # ← unified with optimisation phase
        L=450,
    )

    corrected_result = {
        "h": h,
        "b": b,
        "t_foam": t_core,
        "flange_layup": flange_layup,
        "web_layup": web_layup,
        "t_flange": flange_thickness,
        "t_side": web_thickness,
        "t_web": props_corr["t_web [mm]"],
        "mass": props_corr["Mass [g]"],
        "max_deflection": props_corr["Max Deflection [mm]"],
        "EI": props_corr["EI [Nmm²]"],
        "stiffness": props_corr["EI [Nmm²]"] / 1e3,
        "mass_flange": props_corr["mass_flange"],
        "mass_core": props_corr["mass_core"],
        "mass_side": props_corr["mass_side"],  # side panels
        "F": F_design,  # propagate for downstream calls
    }

    return corrected_result


# -----------------------------------------------------------------------------
# CLI‑style interactive correction (kept for debug / batch use) ---------------
# -----------------------------------------------------------------------------

def run_manual_correction(opt_props, opt_flange, opt_web):
    """Legacy interactive console workflow (outside Streamlit)."""

    F_design = _get_design_force(opt_props)

    choice = input("Do you want to manually correct the design? (y/n): ").lower()
    if choice != "y":
        return

    print("\nEnter corrected dimensions and layups:")
    t_core = float(input("  Foam core thickness t_core [mm]: "))
    h = float(input("  Beam height h [mm]: "))
    b = float(input("  Beam width b [mm]: "))
    flange_layup = eval(
        input("  Flange Layup (e.g. [0,45,-45,90,45,0]): ")
    )
    web_layup = eval(input("  Web Layup (e.g. [0,45,-45,0]): "))

    t_flange = compute_laminate_A_matrix(flange_layup)["thickness"]
    t_web = compute_laminate_A_matrix(web_layup)["thickness"]

    props_corr = calculate_beam_properties(
        case_type=1,
        h=h,
        b=b,
        flange_layup=flange_layup,
        web_layup=web_layup,
        t_side=t_web,
        t_core_web=t_core,
        t_core_side=t_web,
        F=F_design,
        L=450,
    )

    # ─ Plot layup stiffness comparison --------------------------------
    plot_laminate_stackup_mechanics_compare(
        opt_flange, flange_layup, title="Flange Layup Comparison"
    )
    plot_laminate_stackup_mechanics_compare(
        opt_web, web_layup, title="Side Panel Layup Comparison"
    )

    # ─ Compare deflections --------------------------------------------
    EI_opt = opt_props["EI [Nmm²]"]
    EI_corr = props_corr["EI [Nmm²]"]

    x_opt, u_opt = fem_beam_solver_correct(EI=EI_opt, L=450, F=F_design, num_elements=100)
    x_corr, u_corr = fem_beam_solver_correct(EI=EI_corr, L=450, F=F_design, num_elements=100)

    plt.figure(figsize=(8, 5))
    plt.plot(x_opt, u_opt, label="Optimized", color="blue")
    plt.plot(x_corr, u_corr, label="Corrected", color="red", linestyle="--")
    plt.xlabel("Beam Length [mm]")
    plt.ylabel("Deflection [mm]")
    plt.title("Deflection Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # ─ Draw corrected cross‑section -----------------------------------
    draw_cross_section(
        case_type=1,
        h=h,
        b=b,
        t_flange=t_flange,
        t_web=props_corr["t_web [mm]"],
        t_side=t_web,
        t_core_web=t_core,
        t_core_side=t_web,
    )

    print(
        """
=== Manual Correction Summary ===
Parameters: t_foam={:.2f} mm, h={:.2f} mm, b={:.2f} mm
Mass: {:.2f} g | Deflection: {:.3f} mm | Safety Factor: {:.2f}
Skin thickness = {:.2f} mm (flange layup)
Side thickness = {:.2f} mm (side panel layup)
Flange Layup: {}
Side Panel Layup: {}
""".format(
            t_core,
            h,
            b,
            props_corr["Mass [g]"],
            props_corr["Max Deflection [mm]"],
            props_corr["Safety Factor (Bending)"],
            t_flange,
            t_web,
            flange_layup,
            web_layup,
        )
    )
