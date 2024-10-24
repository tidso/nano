# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:17:28 2024

@author: IDSO
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar_squared_over_2m = 3.8  # units of eV·Å^2
a = 4.05  # Lattice constant in Angstroms

# Reciprocal lattice basis vectors for the bcc lattice (reciprocal of fcc)
b1 = (2 * np.pi / a) * np.array([1, 1, -1])
b2 = (2 * np.pi / a) * np.array([1, -1, 1])
b3 = (2 * np.pi / a) * np.array([-1, 1, 1])

# Pseudopotential coefficients (in eV) mapped by |G|^2
V_pseudopotential = {
    0: -37.13,  # V_G(0,0,0)
    3: 0.24,    # For |G|^2 = 3 corresponding to G=(±1,±1,±1)
    4: 0.757,   # For |G|^2 = 4 corresponding to G=(±2,0,0) and permutations
    8: 0.687,   # For |G|^2 = 8 corresponding to G=(±2,±2,0) and permutations
    11: 0.205,  # For |G|^2 = 11 corresponding to G=(±3,±1,±1) and permutations
}

def find_Efermi(valence_count):
    """Deterimine the fermi energy based on the number of valence electrons and the lattice constant."""
    # Calculate the number density (number of electrons per unit volume)
    number_density = (4 * valence_count) / a**3  # 4 atoms per unit cell in fcc
    # Calculate the Fermi wavevector k_F
    k_fermi = (3 * np.pi**2 * number_density)**(1/3)
    # Calculate the Fermi energy E_F
    E_fermi = hbar_squared_over_2m * k_fermi**2
    return E_fermi

def generate_g_vectors(order):
    """Generate a list of reciprocal lattice vectors and their Miller indices up to a given order for the bcc lattice."""
    g_vectors = []
    for h in range(-order, order + 1):
        for k in range(-order, order + 1):
            for l in range(-order, order + 1):
                # In bcc lattice, allowed G vectors satisfy h + k + l = even
                if (h + k + l) % 2 == 0:
                    G = h * b1 + k * b2 + l * b3
                    g_vectors.append((G, (h, k, l)))
    return g_vectors

def get_V_G_squared(hkl_diff):
    """Return the pseudopotential coefficient V_G for the magnitude squared of G."""
    # Calculate |G|^2
    h, k, l = hkl_diff
    G_squared = h**2 + k**2 + l**2
    return V_pseudopotential.get(G_squared, 0.0)

def generate_kpoints(num_points, high_symmetry_points):
    """Generate k-points along the high-symmetry path in reciprocal space."""
    k_points = []
    for i in range(len(high_symmetry_points) - 1):
        start = high_symmetry_points[i]
        end = high_symmetry_points[i + 1]
        # Generate points between start and end
        segment = np.linspace(start, end, num_points, endpoint=False)
        k_points.append(segment)
    k_points.append(np.array([high_symmetry_points[-1]]))  # Add the last point
    k_points = np.concatenate(k_points)
    return k_points

def calculate_k_distances(k_points):
    """Calculate cumulative distances between consecutive k-points."""
    delta_k = np.diff(k_points, axis=0)
    delta_k_norm = np.linalg.norm(delta_k, axis=1)
    k_distances = np.concatenate(([0], np.cumsum(delta_k_norm)))
    return k_distances

def construct_hamiltonian(k_point, g_vectors):
    """Construct the Hamiltonian matrix for a given k-point."""
    N = len(g_vectors)
    H = np.zeros((N, N), dtype=np.float64)

    V_0 = V_pseudopotential.get(0, 0.0)  # V_G(0,0,0)

    for i in range(N):
        G_i, (h_i, k_i, l_i) = g_vectors[i]
        k_plus_G_i = k_point + G_i
        E_kinetic = hbar_squared_over_2m * np.dot(k_plus_G_i, k_plus_G_i)
        H[i, i] = E_kinetic + V_0  # Include V_G(0,0,0) in diagonal elements

        for j in range(i + 1, N):
            G_j, (h_j, k_j, l_j) = g_vectors[j]
            delta_h = h_i - h_j
            delta_k = k_i - k_j
            delta_l = l_i - l_j
            hkl_diff = (delta_h, delta_k, delta_l)
            V_G = get_V_G_squared(hkl_diff)
            if V_G != 0.0:
                H[i, j] = V_G
                H[j, i] = V_G  # Hermitian matrix

    return H

def plot_energy_bands(k_points, g_vectors, k_distances):
    """Plot the energy bands for the given k-points and G-vectors."""
    fig, ax = plt.subplots(figsize=(8, 6))

    N = len(g_vectors)
    energies_all_k = []

    for idx, k_point in enumerate(k_points):
        H = construct_hamiltonian(k_point, g_vectors)
        eigenvalues = np.linalg.eigvalsh(H)
        energies_all_k.append(eigenvalues)

    energies_all_k = np.array(energies_all_k)

    for n in range(N):
        ax.plot(k_distances, energies_all_k[:, n], color='black', alpha=0.5)

    return fig, ax

def mark_high_symmetry_points(ax, k_distances, num_points, high_symmetry_labels):
    """Mark high-symmetry points on the plot with vertical lines and labels."""
    num_segments = len(high_symmetry_labels) - 1
    high_symmetry_indices = [i * num_points for i in range(num_segments)]
    high_symmetry_indices.append(len(k_distances) - 1)
    high_symmetry_positions = k_distances[high_symmetry_indices]

    # Add vertical lines at high-symmetry points
    for position in high_symmetry_positions:
        ax.axvline(x=position, color='gray', linestyle='-', lw=1)

    # Set x-ticks and labels
    ax.set_xticks(high_symmetry_positions)
    ax.set_xticklabels(high_symmetry_labels)
    return ax

def finalize_plot(ax, k_distances):
    """Set plot limits, labels, and display the plot."""
    ax.set_xlim([k_distances[0], k_distances[-1]])
    ax.set_ylim(bottom=None)
    ax.set_xlabel('Wave Vector', fontsize=14)
    ax.set_ylabel('Energy (eV)', fontsize=14)
    ax.set_title('Energy Bands for FCC Aluminum\n(Including Pseudopotential)', fontsize=14)
    plt.tight_layout()
    plt.savefig('band_structure_Al_FCC.png', dpi=900)
    plt.show()

# Main function
def main():
    # User input for the order of G-vectors
    order = int(input('Enter the order of g-vectors that you would like to use: '))
    g_vectors = generate_g_vectors(order=order)

    # Calculate Fermi energy
    valence_count = int(input('Enter the integer number of valence electrons to consider: '))
    E_fermi = find_Efermi(valence_count)
    print(f"Fermi Energy (E_F) = {E_fermi:.2f} eV")

    # High-symmetry points in reciprocal space
    k_Gamma = np.array([0, 0, 0])
    k_X = (2 * np.pi / a) * np.array([1, 0, 0])      # X point
    k_L = (2 * np.pi / a) * np.array([0.5, 0.5, 0.5])  # L point

    # High-symmetry path
    high_symmetry_points = [k_Gamma, k_X, k_L, k_Gamma]

    # High-symmetry labels with symbolic expressions
    high_symmetry_labels = [
        r'$\Gamma$ (0, 0, 0)',            # Gamma
        r'$X$ $(2\pi/a, 0, 0)$',          # X
        r'$L$ $(\pi/a, \pi/a, \pi/a)$',   # L
        r'$\Gamma$ (0, 0, 0)'             # Gamma
    ]

    #Finalize plot aesthetics
    num_points = 50  # Increase for smoother curves
    k_points = generate_kpoints(num_points, high_symmetry_points)
    k_distances = calculate_k_distances(k_points)
    fig, ax = plot_energy_bands(k_points, g_vectors, k_distances)
    ax.axhline(y=E_fermi, color='red', linestyle='--', linewidth=1.5, label=f'$E_F = {E_fermi:.2f}$ eV')
    ax = mark_high_symmetry_points(ax, k_distances, num_points, high_symmetry_labels)
    ax.legend(fontsize=12, loc='upper right')
    finalize_plot(ax, k_distances)

if __name__ == "__main__":
    main()
