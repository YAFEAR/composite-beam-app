import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from beam_analytics import calculate_beam_properties
from laminate_mechanics import (
    plot_laminate_stackup_mechanics,
    compute_laminate_A_matrix,
    plot_laminate_stackup_mechanics_compare,
)
from cross_section import draw_cross_section
from fem_solver import fem_beam_solver_correct
from manual_correction import run_manual_correction
from beam_analytics import rho_skin, rho_core


def update_result_with_mass_breakdown(result, h, b, t_flange, t_foam):
    """Append flange / core / side‑panel masses to the result dict.

    Pure post‑processing helper; does *not* affect any optimisation maths.
    """
    A_flange = 2 * (b * t_flange)
    A_web = 2 * (result["t_web"] * (h - 2 * t_flange))
    A_core = (b - 2 * result["t_web"]) * (h - 2 * t_flange)
    L = 450  # beam length (mm)

    result["mass_flange"] = round(A_flange * rho_skin * L, 2)
    result["mass_core"] = round(A_core * rho_core * L, 2)
    result["mass_side"] = round(A_web * rho_skin * L, 2)  # side‑panels (was T‑whip)
    return result


class BeamQEnv:
    def __init__(self, case_type=1, layup_pool=None, F=4000.0, L=450.0):
        # ---------------------------------------------------------------
        # Allow external scripts (e.g. the Streamlit UI) to inject the
        # user‑specified load *without touching the optimisation code*.
        # ---------------------------------------------------------------
        self.case_type = case_type
        self.F = F  # <-- user‑defined design load [N]
        self.L = L  # beam span [mm]

        # ---- default geometry seeds ----------------------------------
        self.t_core = 4.0
        self.h = 35.0
        self.b = 20.0

        # ---- layup search pool ---------------------------------------
        self.layup_pool = [
    # ── ❶ Classic quasi‑isotropic 8‑ply variants ────────────────────────────
    [0,  45,  90, -45, -45,  90, 45, 0],
    [0,  45, -45,  90,  90, -45, 45, 0],
    [45, 0,  90, -45, -45, 90,  0, 45],

    # ── ❷ Thin face‑skin (4–6 plies) variants ───────────────────────────────
    [0, 45, -45, 0],               # already had
    [45, -45],                     # already had
    [0, 90, 0, 90],                # already had
    [0, 0, 45, -45, 0, 0],
    [0, 0, 90, 0, 0],
    [0, 45, -45, 45, -45, 0],
    [45, -45, 90, -45, 45],        # already had
    [45, -45, 45, -45],            # already had
    [0, 90, 90, 0],                # already had

    # ── ❸ Heavier face‑skins (7–9 plies) for stiffness ----------------------
    [0, 0, 45, -45, 90, -45, 45, 0, 0],  # already had
    [0, 0, 0, -45, 90, -45, 0, 0, 0],    # already had
    [0, 0, 0, -45, 0, -45, 0, 0, 0],     # already had
    [0, 0, 45, -45, 45, -45, 0, 0],      # already had
    [0, 0, 0, 90, 0, 90, 0, 0],
    [45, -45, 0, 0, -45, 45],            # already had
    [0, 90, 0, 0, 90, 0],                # already had
    [0, 0, 0, 0, 90, 0, 0, 0, 0],        # already had
    [0, 45, -45, 90, -45, 45, 0],        # already had
    [0, 45, 90, -45, 0],                 # already had
    [0, 45, 90, 90, -45, 0],             # already had

    # ── ❹ Bias‑dominated skins (shear‑friendly) -----------------------------
    [45, -45, 45, -45, 45, -45],
    [45, -45, -45, 45],
    [45, 45, -45, -45, 45, 45, -45, -45],
    [45,-45,45,-45,45,-45,45,-45],

    # ── ❺ 0‑/±45 hybrids (tension‑dominated) --------------------------------
    [0, 45, 0, -45, 0, 45, 0, -45],
    [0, 0, 45, 0, -45, 0, 45, 0, -45],
    [0, 0, 0, 45, -45, -45, 45, 0, 0],

    # ── ❻ 90‑rich skins (buckling‑resistant) ---------------------------------
    [90, 0, 0, 0, 90],
    [90, 45, -45, 90],
    [0, 90, 0, 90, 0, 90],

    # ── ❼ Box‑beam side‑panel specific (thin) --------------------------------
    [0, 90, -45, 45],      # good torsion‑stiff web
    [0, 0, 90, 90],
    [45, -45, 0, 0],

    # ── ❽ Box‑beam side‑panel (thicker) --------------------------------------
    [0, 45, 90, -45, 0, 90],
    [0, 0, 45, 90, -45, 0, 0],

    # ── ❾ Mixed non‑symmetric (for testing instability) ----------------------
    [0, 0, 0, 45, -45, 90],    # intentionally asymmetric
    [45, 0, -45, 0, 90],

    # ── ❿ Ultra‑thin demo (2–3 ply) ------------------------------------------
    [0, 0, 0],
    [0, 90, 0],
    [45, -45, 45],

    # ── ⓫ Quasi‑iso but 6‑ply (lighter) --------------------------------------
    [0, 45, 90, -45, 45, 0],
    [45, -45, 90, -45, 45, -45],

    # ── ⓬ Pure 0°/90° wraps (torsion weak) -----------------------------------
    [0, 0, 0, 90, 90, 90],
    [0, 90, 90, 90, 0],
    [0, 0, 90, 90, 0, 0],

    # ── ⓭ High‑torsion balanced ±45 only -------------------------------------
    [45, -45, 45, -45, 45, -45, 45, -45],

    # ── ⓮ Alternating 0/±45 stack -------------------------------------------
    [0, 45, 0, -45, 0, 45],
    [0, -45, 0, 45, 0, -45, 0, 45],
]
        

        # ---- pick best initial flange / web layups -------------------
        self.flange_layup, self.web_layup, self.best_reward = self.find_best_layup_pair()

        # ---- fixed skin / side‑panel thicknesses from layups ----------
        self.t_flange_fixed = compute_laminate_A_matrix(self.flange_layup)[
            "thickness"
        ]
        self.t_side_fixed = compute_laminate_A_matrix(self.web_layup)["thickness"]

        # ---- design variable grids -----------------------------------
        self.t_foam_values = np.linspace(4.0, 14.0, 6)
        self.h_values = np.linspace(30.0, 55.0, 6)
        self.b_values = np.linspace(20.0, 40.0, 5)

        self.n_states = (
            len(self.t_foam_values),
            len(self.h_values),
            len(self.b_values),
        )

        self.q_table = np.random.uniform(low=-1, high=1, size=self.n_states)

    # ---------------------------------------------------------------------
    # Remaining methods are UNCHANGED (logic untouched) --------------------
    # ---------------------------------------------------------------------

    def find_best_layup_pair(self):
        best_reward = -np.inf
        best_flange = None
        best_web = None

        for flange in self.layup_pool:
            for web in self.layup_pool:
                flange_thick = compute_laminate_A_matrix(flange)["thickness"]
                web_thick = compute_laminate_A_matrix(web)["thickness"]

                props = calculate_beam_properties(
                    case_type=self.case_type,
                    h=self.h,
                    b=self.b,
                    flange_layup=flange,
                    web_layup=web,
                    t_side=web_thick,
                    t_core_web=self.t_core,
                    t_core_side=web_thick,
                    F=self.F,
                    L=self.L,
                )
                reward = self.custom_reward(
                    props["Mass [g]"], props["Max Deflection [mm]"]
                )
                if reward > best_reward and props["Max Deflection [mm]"] <= 3.8:
                    best_reward = reward
                    best_flange = flange
                    best_web = web

        plot_laminate_stackup_mechanics(best_flange, title="Initial Flange Layup")
        plot_laminate_stackup_mechanics(best_web, title="Initial Web Layup")
        return best_flange, best_web, best_reward

    def reset(self):
        return tuple(np.random.randint(0, dim) for dim in self.n_states)

    def custom_reward(self, mass, deflection):
        # --- unchanged reward logic -----------------------------------
        if mass <= 70:
            mass_reward = +10
        elif mass <= 80:
            mass_reward = +7
        elif mass <= 100:
            mass_reward = +4
        elif mass <= 120:
            mass_reward = +2
        elif mass <= 170:
            mass_reward = 0
        else:
            mass_reward = -20

        if deflection <= 2:
            deflection_reward = +10
        elif deflection <= 3:
            deflection_reward = +5
        elif deflection <= 4:
            deflection_reward = +2
        else:
            deflection_reward = -15

        return 0.75 * mass_reward + 0.25 * deflection_reward

    def step(self, idx):
        t_foam = self.t_foam_values[idx[0]]
        h = self.h_values[idx[1]]
        b = self.b_values[idx[2]]

        props = calculate_beam_properties(
            case_type=self.case_type,
            h=h,
            b=b,
            flange_layup=self.flange_layup,
            web_layup=self.web_layup,
            t_side=self.t_side_fixed,
            t_core_web=t_foam,
            t_core_side=self.t_side_fixed,
            F=self.F,
            L=self.L,
        )

        reward = self.custom_reward(props["Mass [g]"], props["Max Deflection [mm]"])
        return reward, props

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.reset()
        else:
            return np.unravel_index(np.argmax(self.q_table), self.q_table.shape)

    def update_q(self, state, reward, next_state, alpha, gamma):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state] = (1 - alpha) * self.q_table[state] + alpha * (
            reward + gamma * best_next
        )


# =============================================================================
# Main optimisation routine ----------------------------------------------------
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # Read user‑defined force from environment (set by Streamlit or CLI)
    #   ‑ If FORCE_N is not provided, fall back to the historical 4 kN
    # ------------------------------------------------------------------
    F_user = float(os.getenv("FORCE_N", "4000"))

    env = BeamQEnv(case_type=1, F=F_user)

    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.99

    num_episodes = 5000
    best_reward = -np.inf
    best_props = None
    best_state = None

    print(
        f"Starting Q-learning optimization for I‑beam (Case 1) with F = {F_user:.0f} N...\n"
    )

    for episode in range(num_episodes):
        state = env.reset()
        reward, props = env.step(state)
        next_state = env.choose_action(state, epsilon)
        env.update_q(state, reward, next_state, alpha, gamma)

        if reward > best_reward:
            best_reward = reward
            best_props = props
            best_state = state

        t_foam = env.t_foam_values[state[0]]
        h = env.h_values[state[1]]
        b = env.b_values[state[2]]

        print(
            f"Step {episode+1:04} | t_foam={t_foam:.1f} mm | h={h:.1f} mm | b={b:.1f} mm"
        )
        print(
            f"          Mass = {props['Mass [g]']:.2f} g | Deflection = {props['Max Deflection [mm]']:.3f} mm | Reward = {reward:.2f}\n"
        )

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    t_foam = env.t_foam_values[best_state[0]]
    h = env.h_values[best_state[1]]
    b = env.b_values[best_state[2]]
    t_flange = env.t_flange_fixed
    t_side = env.t_side_fixed

    print(
        f"""
=== Optimization Complete ===
Best Reward: {best_reward:.2f}
Best Parameters: t_foam={t_foam:.2f} mm, h={h:.2f} mm, b={b:.2f} mm
Mass: {best_props['Mass [g]']:.2f} g | Deflection: {best_props['Max Deflection [mm]']:.3f} mm | Safety Factor: {best_props['Safety Factor (Bending)']:.2f}
t_flange = {t_flange:.2f} mm (from flange layup)
t_side   = {t_side:.2f} mm (from side panel layup)
Flange Layup: {env.flange_layup}
Side Panel Layup: {env.web_layup}
"""
    )

    print("\nDrawing best beam cross-section and FEM deflection...")
    draw_cross_section(
        case_type=1,
        h=h,
        b=b,
        t_flange=t_flange,
        t_web=best_props["t_web [mm]"],
        t_side=t_side,
        t_core_web=t_foam,
        t_core_side=t_side,
    )

    EI = best_props["EI [Nmm²]"]
    delta_analytical = best_props["Max Deflection [mm]"]
    x, u = fem_beam_solver_correct(EI=EI, L=env.L, F=env.F, num_elements=100)

    plt.figure(figsize=(8, 5))
    plt.plot(x, u, label="FEM Deflection")
    plt.hlines(
        -delta_analytical,
        0,
        env.L,
        colors="red",
        linestyles="--",
        label="Analytical Max Deflection",
    )
    plt.xlabel("Beam Length [mm]")
    plt.ylabel("Deflection [mm]")
    plt.title("Best Beam Deflection – I‑beam (Case 1)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # ✅ Save optimised result to JSON for Streamlit UI -----------------
    result = {
        "t_foam": t_foam,
        "h": h,
        "b": b,
        "t_flange": t_flange,
        "t_side": t_side,
        "t_web": best_props["t_web [mm]"],
        "flange_layup": env.flange_layup,
        "web_layup": env.web_layup,
        "mass": best_props["Mass [g]"],
        "max_deflection": best_props["Max Deflection [mm]"],
        "EI": best_props["EI [Nmm²]"],
        "stiffness": best_props["EI [Nmm²]"] / 1e3,
    }

    # --- mass breakdown fields --------------------------------------
    result = update_result_with_mass_breakdown(result, h, b, t_flange, t_foam)

    with open("opt_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # run_manual_correction(best_props, env.flange_layup, env.web_layup)


if __name__ == "__main__":
    main()
