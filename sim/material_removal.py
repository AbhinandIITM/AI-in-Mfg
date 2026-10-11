"""
sim/material_removal.py

Material removal geometry calculator for CNC milling operations.
Computes actual workpiece geometry evolution pass-by-pass for visualization.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class PassGeometry:
    """Geometry state after one milling pass."""
    pass_number: int
    remaining_diameter: float  # mm
    material_removed: float    # mm³
    surface_profile: np.ndarray  # Radial profile array
    axial_position: float      # mm along workpiece length

class MaterialRemovalSimulator:
    """
    Simulates actual material removal geometry for turning operations.
    Tracks workpiece diameter reduction pass-by-pass.
    """

    def __init__(self,
                 initial_diameter: float,
                 final_diameter: float,
                 length: float,
                 resolution: int = 100):
        """
        Args:
            initial_diameter: Starting workpiece diameter (mm)
            final_diameter: Target diameter after all passes (mm)
            length: Workpiece cutting length (mm)
            resolution: Number of profile points along length
        """
        self.D_initial = initial_diameter
        self.D_final = final_diameter
        self.L = length
        self.resolution = resolution

        # Total material to remove
        self.kappa_total = (initial_diameter - final_diameter) / 2.0

        # Current geometry state
        self.current_diameter = initial_diameter
        self.axial_points = np.linspace(0, length, resolution)
        self.radial_profile = np.ones(resolution) * (initial_diameter / 2.0)

        self.pass_history: List[PassGeometry] = []

    def execute_pass(self, depth_of_cut: float, pass_number: int) -> PassGeometry:
        """
        Execute one turning pass and update geometry.

        Args:
            depth_of_cut: Radial depth of cut (mm)
            pass_number: Current pass index

        Returns:
            PassGeometry object with updated state
        """
        # Constrain depth to remaining material
        actual_depth = min(depth_of_cut, self.current_diameter / 2.0 - self.D_final / 2.0)

        if actual_depth <= 0:
            # No material left to remove
            return PassGeometry(
                pass_number=pass_number,
                remaining_diameter=self.current_diameter,
                material_removed=0.0,
                surface_profile=self.radial_profile.copy(),
                axial_position=0.0
            )

        # Update diameter
        new_diameter = self.current_diameter - 2 * actual_depth

        # Calculate volume removed (cylinder shell)
        volume_removed = np.pi * self.L * (
            (self.current_diameter / 2.0) ** 2 - (new_diameter / 2.0) ** 2
        )

        # Update radial profile
        self.radial_profile = np.ones(self.resolution) * (new_diameter / 2.0)
        self.current_diameter = new_diameter

        geometry = PassGeometry(
            pass_number=pass_number,
            remaining_diameter=new_diameter,
            material_removed=volume_removed,
            surface_profile=self.radial_profile.copy(),
            axial_position=0.0
        )

        self.pass_history.append(geometry)
        return geometry

    def get_cross_section_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 2D coordinates for cross-section visualization.

        Returns:
            (z_coords, r_coords): Axial and radial coordinates
        """
        return self.axial_points, self.radial_profile

    def get_3d_surface_mesh(self) -> Dict[str, np.ndarray]:
        """
        Generate 3D cylindrical surface mesh for plotly visualization.

        Returns:
            Dictionary with 'x', 'y', 'z' coordinate arrays
        """
        theta = np.linspace(0, 2 * np.pi, 50)
        z = self.axial_points

        Z, Theta = np.meshgrid(z, theta)
        R = np.tile(self.radial_profile, (len(theta), 1))

        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)

        return {'x': X, 'y': Y, 'z': Z}

    def reset(self):
        """Reset to initial geometry."""
        self.current_diameter = self.D_initial
        self.radial_profile = np.ones(self.resolution) * (self.D_initial / 2.0)
        self.pass_history.clear()
