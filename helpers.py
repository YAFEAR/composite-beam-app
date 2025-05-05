"""Convenience wrappers used by the Streamlit UI

Only *interfaces* are changed so that the new design‑force input from the UI is
threaded all the way through FEM and optimisation.  No numerical algorithms are
modified.
"""

import os
from typing import Any, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Q‑learning optimiser --------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────

def run_q_learning_optimization(
    case: int,
    length: float,
    h_max: float,
    b_max: float,
    mass_limit: float,
    deflection_limit: float,
    force_max: float,
) -> Dict[str, Any]:
    """Launch the optimiser with all constraint knobs.

    The underlying script (`beam_qlearning_case1.py`) now grabs the design load
    from the environment variable **FORCE_N**. We set it here so every call to
    the optimiser inherits the user‑specified force without touching its
    internal API.
    """

    # Pass force to optimiser via environment --------------------------
    os.environ["FORCE_N"] = str(force_max)

    from beam_qlearning_case1 import main as optimiser_main

    # Future: feed the geometric / mass / deflection limits into the env.
    # At present, the optimiser module does not accept these as kwargs, so we
    # just call it directly.  The env already hard‑codes those constraint
    # ranges, which users requested *not* to alter here.
    _ = case, length, h_max, b_max, mass_limit, deflection_limit  # noqa: F841

    best_result = optimiser_main()  # returns via JSON; function prints to CLI
    return best_result


# ─────────────────────────────────────────────────────────────────────────────
# FEM post‑processing ---------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────

def run_fem_simulation(beam_result: Dict[str, Any], mesh_resolution: int):
    """Run Euler‑Bernoulli FEM with the *same* force used during optimisation."""

    from fem_solver import fem_beam_solver_correct

    EI = beam_result["EI"]
    L = beam_result.get("L", 450)  # default if span is not stored
    F_design = float(beam_result.get("F", 4000))

    x, u = fem_beam_solver_correct(EI=EI, L=L, F=F_design, num_elements=mesh_resolution)

    max_deflection = abs(min(u))  # conservative check
    return {
        "x": x.tolist(),
        "u": u.tolist(),
        "max_deflection": max_deflection,
        "stiffness": EI / 1e3,  # [N/mm]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers ------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_section(beam_result):
    from cross_section import draw_cross_section

    fig = draw_cross_section(
        case_type=1,
        h=beam_result["h"],
        b=beam_result["b"],
        t_flange=beam_result["t_flange"],
        t_web=beam_result["t_web"],
        t_side=beam_result["t_side"],
        t_core_web=beam_result["t_foam"],
        t_core_side=beam_result["t_side"],
    )
    return fig


def plot_laminate_stackup(beam_result):
    from laminate_mechanics import plot_laminate_stackup_mechanics

    return plot_laminate_stackup_mechanics(beam_result)


def plot_laminate_stackup_compare(opt_result, corr_result):
    from laminate_mechanics import plot_laminate_stackup_mechanics_compare

    return plot_laminate_stackup_mechanics_compare(opt_result, corr_result)
