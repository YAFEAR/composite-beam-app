# fem_solver.py

import numpy as np

def fem_beam_solver_correct(EI, L=450, F=4000, num_elements=100):
    """
    Solve beam deflection using FEM (Euler-Bernoulli Beam Theory) correctly.
    
    Parameters:
    - EI: bending stiffness [Nmm²]
    - L: beam length [mm]
    - F: total load [N] (applied at midspan)
    - num_elements: number of beam elements
    
    Returns:
    - x: positions along beam [mm]
    - u: vertical displacements [mm] at nodes
    """
    le = L / num_elements  # element length
    n_nodes = num_elements + 1
    dof_per_node = 2
    total_dofs = n_nodes * dof_per_node  # displacement + rotation at each node

    # Initialize global stiffness matrix and load vector
    K_global = np.zeros((total_dofs, total_dofs))
    F_global = np.zeros(total_dofs)

    # Local stiffness matrix for Euler-Bernoulli beam element
    k_local = (EI / le**3) * np.array([
        [12, 6*le, -12, 6*le],
        [6*le, 4*le**2, -6*le, 2*le**2],
        [-12, -6*le, 12, -6*le],
        [6*le, 2*le**2, -6*le, 4*le**2]
    ])

    # Assemble the global stiffness matrix
    for e in range(num_elements):
        idx = np.array([2*e, 2*e+1, 2*e+2, 2*e+3])
        for i in range(4):
            for j in range(4):
                K_global[idx[i], idx[j]] += k_local[i, j]

    # Apply force at midspan node (only displacement DOF)
    mid_node = n_nodes // 2
    F_global[2*mid_node] = -F

    # Apply boundary conditions (simple supports: displacement = 0 at node 0 and node n)
    fixed_dofs = [0, 2*(n_nodes-1)]  # Displacement DOFs at both ends
    free_dofs = list(set(range(total_dofs)) - set(fixed_dofs))

    # Solve reduced system
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = F_global[free_dofs]
    d_reduced = np.linalg.solve(K_reduced, F_reduced)

    # Build full solution vector
    d = np.zeros(total_dofs)
    d[free_dofs] = d_reduced

    # Extract only displacements u (not rotations)
    u = d[0::2]

    # Positions along the beam
    x = np.linspace(0, L, n_nodes)

    return x, u

# Example usage for testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Run solver
    x, u = fem_beam_solver_correct(EI=3.4456e9, L=450, F=4000, num_elements=100)

    # Plot
    plt.plot(x, u, label="Deflection", color='orange')
    plt.xlabel("Beam Length [mm]")
    plt.ylabel("Deflection [mm]")
    plt.title("Corrected Beam Deflection FEM (2 DOFs/node)")
    plt.grid(True)
    plt.legend()
    plt.show()