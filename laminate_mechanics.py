import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def compute_laminate_A_matrix(layup, E1=115000, E2=7600, G12=3500, v12=0.3, ply_thickness=0.14):
    n = len(layup)
    total_thickness = n * ply_thickness
    A = np.zeros((3, 3))

    Q11 = E1 / (1 - v12**2 * E2 / E1)
    Q22 = E2 / (1 - v12**2 * E2 / E1)
    Q12 = v12 * E2 / (1 - v12**2 * E2 / E1)
    Q66 = G12
    Q = np.array([[Q11, Q12, 0],
                  [Q12, Q22, 0],
                  [0,   0,   Q66]])

    for angle in layup:
        theta = np.radians(float(angle))
        m = np.cos(theta)
        n = np.sin(theta)

        T1 = np.array([[m**2, n**2, 2*m*n],
                       [n**2, m**2, -2*m*n],
                       [-m*n, m*n, m**2 - n**2]])

        T2 = np.array([[m**2, n**2, m*n],
                       [n**2, m**2, -m*n],
                       [-2*m*n, 2*m*n, m**2 - n**2]])

        Q_bar = T1 @ Q @ T2.T
        A += Q_bar * ply_thickness

    return {
        "A_matrix": A,
        "thickness": total_thickness,
        "E1": E1,
        "E2": E2,
        "G12": G12,
        "v12": v12
    }


def plot_laminate_stackup_mechanics(layup, title="Laminate Mechanics", ply_thickness=0.14):
    angles = np.linspace(0, 360, 361)
    E_theta = []
    G_theta = []
    E2_theta = []

    A_dict = compute_laminate_A_matrix(layup, ply_thickness=ply_thickness)
    A = A_dict["A_matrix"]
    thickness = A_dict["thickness"]
    a = np.linalg.inv(A)

    for theta_deg in angles:
        theta = np.radians(theta_deg)
        m = np.cos(theta)
        n = np.sin(theta)

        T_sigma = np.array([[m**2, n**2, m*n],
                            [n**2, m**2, -m*n],
                            [-2*m*n, 2*m*n, m**2 - n**2]])

        a_rotated = T_sigma @ a @ T_sigma.T

        E_eff = 1 / a_rotated[0, 0] * thickness if a_rotated[0, 0] > 0 else 0
        G_eff = 1 / a_rotated[2, 2] * thickness if a_rotated[2, 2] > 0 else 0
        E2_eff = 1 / a_rotated[1, 1] * thickness if a_rotated[1, 1] > 0 else 0

        E_theta.append(E_eff / 1000)  # Convert to GPa
        G_theta.append(G_eff / 1000)  # Convert to GPa
        E2_theta.append(E2_eff / 1000)  # Convert to GPa

    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(211, polar=True)
    ax.plot(np.radians(angles), E_theta, label='E1 (GPa)', color='blue')
    ax.plot(np.radians(angles), G_theta, label='G12 (GPa)', color='green')
    ax.plot(np.radians(angles), E2_theta, label='E2 (GPa)', color='red')
    ax.fill(np.radians(angles), E_theta, alpha=0.1, color='blue')
    ax.fill(np.radians(angles), G_theta, alpha=0.1, color='green')
    ax.fill(np.radians(angles), E2_theta, alpha=0.1, color='red')
    ax.set_title("Polar Properties (GPa)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    # Plot ply stackup
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, len(layup) * ply_thickness)
    ax2.set_title("Ply Stackup (Top to Bottom)")
    ax2.set_xlabel("Orientation [°]")
    ax2.set_ylabel("Through-thickness [mm]")

    for i, angle in enumerate(reversed(layup)):
        rect = Rectangle((0.1, i * ply_thickness), 0.8, ply_thickness, edgecolor='black', facecolor='lightgray')
        ax2.add_patch(rect)
        ax2.text(0.5, i * ply_thickness + ply_thickness / 2, f"{angle}°",
                 ha='center', va='center', fontsize=10)

    ax2.set_xticks([])
    ax2.set_yticks(np.arange(0, len(layup) * ply_thickness + 0.01, ply_thickness))
    ax2.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

def plot_laminate_stackup_mechanics_compare(opt_layup, corr_layup, labels=None, title="Layup Comparison", ply_thickness=0.14):
    angles = np.linspace(0, 360, 361)

    def compute_E_theta(layup):
        A_dict = compute_laminate_A_matrix(layup, ply_thickness=ply_thickness)
        A = A_dict["A_matrix"]
        thickness = A_dict["thickness"]
        a = np.linalg.inv(A)

        E_theta = []
        for theta_deg in angles:
            theta = np.radians(theta_deg)
            m, n = np.cos(theta), np.sin(theta)
            T_sigma = np.array([
                [m**2, n**2, m*n],
                [n**2, m**2, -m*n],
                [-2*m*n, 2*m*n, m**2 - n**2]
            ])
            a_rot = T_sigma @ a @ T_sigma.T
            E = 1 / a_rot[0, 0] * thickness if a_rot[0, 0] > 0 else 0
            E_theta.append(E / 1000)  # Convert to GPa
        return np.array(E_theta)

    E_opt = compute_E_theta(opt_layup)
    E_corr = compute_E_theta(corr_layup)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    label_opt = labels[0] if labels else 'Optimized'
    label_corr = labels[1] if labels else 'Corrected'

    ax.plot(np.radians(angles), E_opt, label=label_opt, color='blue')
    ax.plot(np.radians(angles), E_corr, label=label_corr, color='red', linestyle='--')
    ax.fill(np.radians(angles), E_opt, alpha=0.1, color='blue')
    ax.fill(np.radians(angles), E_corr, alpha=0.1, color='red')
    ax.set_title(f"{title} - E(θ) [GPa]")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
