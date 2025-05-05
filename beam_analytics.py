import numpy as np
from laminate_mechanics import compute_laminate_A_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Material properties (SI‑consistent within mm / N units)
# ─────────────────────────────────────────────────────────────────────────────
E_skin = 115e3       # [MPa]
rho_skin = 1.57e-3   # [g mm⁻³]
rho_core = 0.052e-3  # [g mm⁻³]
sigma_skin_max = 1670  # [MPa]


def calculate_beam_properties(
    case_type,
    h,
    b,
    flange_layup,
    web_layup,
    t_side,
    t_core_web,
    t_core_side,
    F=4000.0,
    L=450.0,
    override_thickness=False,
    t_flange=None,
    t_web=None,
):
    """Return mass, stiffness and strength metrics for a composite beam.

    Parameters
    ----------
    case_type : int
        1 = **I‑beam** (open),
        2 = **Filled I‑beam** (semi‑closed),
        3 = **Rectangular foam‑core beam** (full box).
        The numeric codes are kept for backward compatibility across modules.
    F : float, optional
        Design point load applied at mid‑span [N].  Default 4000 N.
    L : float, optional
        Beam span [mm].  Default 450 mm.
    override_thickness : bool, optional
        If *True* use *t_flange* and *t_web* supplied by caller instead of the
        thickness returned by the laminate calculator (handy for manual
        corrections).
    """

    # ── Laminate effective thicknesses -------------------------------------
    flange_props = compute_laminate_A_matrix(flange_layup)
    web_props = compute_laminate_A_matrix(web_layup)

    if override_thickness:
        t_flange_actual = t_flange
        t_web_actual = t_web
    else:
        t_flange_actual = flange_props["thickness"]
        t_web_actual = web_props["thickness"]

    # ── Flange contributions ----------------------------------------------
    I_flange = 2 * (b * t_flange_actual) * (h / 2) ** 2
    A_flange = 2 * (b * t_flange_actual)

    # ── Shear‑web / side‑panel contributions ------------------------------
    if case_type == 1:
        # I‑beam (open web)
        I_web = 2 * (t_web_actual * (h - 2 * t_flange_actual) ** 3) / 12
        A_web = 2 * (t_web_actual * (h - 2 * t_flange_actual))
        A_core = (b - 2 * t_web_actual) * (h - 2 * t_flange_actual)

    elif case_type == 2:
        # Filled I‑beam (webs + side fillers)
        I_web = 2 * (t_web_actual * (h - 2 * t_flange_actual) ** 3) / 12
        A_web = 2 * (t_web_actual * (h - 2 * t_flange_actual))
        A_core = (b - 2 * t_web_actual) * (h - 2 * t_flange_actual)

    elif case_type == 3:
        # Rectangular foam‑core beam (full box)
        I_outer = (b * h ** 3) / 12
        I_inner = ((b - 2 * t_side) * (h - 2 * t_flange_actual) ** 3) / 12
        I_web = I_outer - I_inner
        A_web = 2 * (b * t_flange_actual) + 2 * ((h - 2 * t_flange_actual) * t_side)
        A_core = (b - 2 * t_side) * (h - 2 * t_flange_actual)

    else:
        raise ValueError("Invalid case_type. Choose 1, 2, or 3.")

    # ── Global sectional properties ---------------------------------------
    I_total = I_flange + I_web
    A_skin = A_flange + A_web
    EI = E_skin * I_total

    # ── Simple‑span, mid‑span point‑load deflection -----------------------
    delta_max = (F * L ** 3) / (48 * EI)

    # ── Bending stress at extreme fibre -----------------------------------
    M_max = F * L / 4
    sigma_max = (M_max * (h / 2)) / I_total

    # ── Mass breakdown -----------------------------------------------------
    mass_total = (A_skin * rho_skin + A_core * rho_core) * L

    return {
        "I [mm^4]": I_total,
        "Mass [g]": mass_total,
        "Max Deflection [mm]": delta_max,
        "Max Stress [MPa]": sigma_max,
        "Safety Factor (Bending)": sigma_skin_max / sigma_max,
        "EI [Nmm²]": EI,
        "t_flange [mm]": t_flange_actual,
        "t_web [mm]": t_web_actual,
        "core_area [mm²]": A_core,
        "mass_flange": round(A_flange * rho_skin * L, 2),
        "mass_core": round(A_core * rho_core * L, 2),
        "mass_side": round(A_web * rho_skin * L, 2),  # side panels (was T‑whip)
    }
