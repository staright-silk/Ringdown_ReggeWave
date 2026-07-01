import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from dataclasses import dataclass, replace
from typing import Tuple, List, Dict


@dataclass
class SimulationConfig:
    Nx: int = 48
    Ny: int = 48
    Nz: int = 48
    Lx: float = 80.0
    Ly: float = 80.0
    Lz: float = 80.0
    schwarzschild_radius: float = 4.0
    potential_strength: float = 1.0
    potential_epsilon: float = 0.6
    absorber_width: float = 12.0
    absorber_strength: float = 40.0
    capture_buffer: float = 1.0
    packet_center: Tuple[float, float, float] = (-25.0, 0.0, 0.0)
    packet_momentum: Tuple[float, float, float] = (3.0, 0.0, 0.0)
    packet_width: float = 3.0
    dt: float = 0.01
    total_steps: int = 350
    snapshot_interval: int = 10
    output_dir: str = "bh_quantum_results"
    run_label: str = "baseline"


class SchwarzschildPotential:
    def __init__(self, config: SimulationConfig):
        self.rs = config.schwarzschild_radius
        self.strength = config.potential_strength
        self.epsilon = config.potential_epsilon

    def evaluate(self, R):
        denom = np.clip(R - self.rs, self.epsilon, None)
        return -self.strength / denom

    def capture_mask(self, R, buffer):
        return R <= (self.rs + buffer)


class AbsorbingBoundary:
    def __init__(self, config: SimulationConfig):
        self.width = config.absorber_width
        self.strength = config.absorber_strength

    def _edge_term(self, coord, half_extent):
        W = np.zeros_like(coord)
        left_edge = -half_extent + self.width
        right_edge = half_extent - self.width
        left_mask = coord < left_edge
        right_mask = coord > right_edge
        left_distance = (left_edge - coord[left_mask]) / self.width
        right_distance = (coord[right_mask] - right_edge) / self.width
        W[left_mask] += self.strength * left_distance ** 2
        W[right_mask] += self.strength * right_distance ** 2
        return W

    def build(self, X, Y, Z, config: SimulationConfig):
        Wx = self._edge_term(X, config.Lx / 2.0)
        Wy = self._edge_term(Y, config.Ly / 2.0)
        Wz = self._edge_term(Z, config.Lz / 2.0)
        return Wx + Wy + Wz


class WavePacket:
    def __init__(self, config: SimulationConfig):
        self.center = np.array(config.packet_center, dtype=np.float64)
        self.momentum = np.array(config.packet_momentum, dtype=np.float64)
        self.sigma = config.packet_width

    def generate(self, X, Y, Z):
        envelope = np.exp(
            -((X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2 + (Z - self.center[2]) ** 2)
            / (4.0 * self.sigma ** 2)
        )
        phase = np.exp(
            1j * (self.momentum[0] * X + self.momentum[1] * Y + self.momentum[2] * Z)
        )
        return envelope * phase


class SplitOperatorSolver:
    def __init__(self, potential_array, absorber_array, K2, dt):
        self.dt = dt
        effective_potential = potential_array - 1j * absorber_array
        self.U_half_potential = np.exp(-1j * effective_potential * dt / 2.0)
        self.U_kinetic = np.exp(-1j * 0.5 * K2 * dt)

    def step(self, psi):
        psi = psi * self.U_half_potential
        psi_k = np.fft.fftn(psi)
        psi_k = psi_k * self.U_kinetic
        psi = np.fft.ifftn(psi_k)
        psi = psi * self.U_half_potential
        return psi


class Diagnostics:
    def __init__(self, X, Y, Z, R, K2, dV, potential_array, capture_mask,
                 axis_index, axis_direction, axis_origin):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.R = R
        self.K2 = K2
        self.dV = dV
        self.N_total = X.size
        self.potential_array = potential_array
        self.capture_mask = capture_mask
        self.axis_coord = (X, Y, Z)[axis_index]
        self.axis_direction = axis_direction
        self.axis_origin = axis_origin
        self.records: List[Dict[str, float]] = []

    def evaluate(self, psi, t):
        density = np.abs(psi) ** 2
        norm = np.sum(density) * self.dV
        mean_x = np.sum(self.X * density) * self.dV
        mean_y = np.sum(self.Y * density) * self.dV
        mean_z = np.sum(self.Z * density) * self.dV
        mean_r = np.sum(self.R * density) * self.dV
        psi_k = np.fft.fftn(psi)
        kinetic_density = 0.5 * self.K2 * np.abs(psi_k) ** 2
        kinetic_energy = np.sum(kinetic_density) * self.dV / self.N_total
        potential_energy = np.sum(self.potential_array * density) * self.dV
        total_energy = kinetic_energy + potential_energy
        captured_probability = np.sum(density[self.capture_mask]) * self.dV
        relative = (self.axis_coord - self.axis_origin) * self.axis_direction
        transmitted_mask = (relative > 0) & (~self.capture_mask)
        reflected_mask = (relative <= 0) & (~self.capture_mask)
        transmitted_probability = np.sum(density[transmitted_mask]) * self.dV
        reflected_probability = np.sum(density[reflected_mask]) * self.dV
        record = {
            "time": t,
            "norm": norm,
            "mean_x": mean_x,
            "mean_y": mean_y,
            "mean_z": mean_z,
            "mean_r": mean_r,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "total_energy": total_energy,
            "captured_probability": captured_probability,
            "reflected_probability": reflected_probability,
            "transmitted_probability": transmitted_probability,
        }
        self.records.append(record)
        return record

    def export_csv(self, filepath):
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        with open(filepath, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(record)


class Visualization:
    def __init__(self, config: SimulationConfig, X, Y, Z, R):
        self.config = config
        self.X = X
        self.Y = Y
        self.Z = Z
        self.R = R
        self.figures_dir = os.path.join(config.output_dir, "figures")
        self.animations_dir = os.path.join(config.output_dir, "animations")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.animations_dir, exist_ok=True)

    def plot_density_slice(self, psi, step, label):
        mid = self.Z.shape[2] // 2
        density_slice = np.abs(psi[:, :, mid]) ** 2
        x_axis = self.X[:, 0, 0]
        y_axis = self.Y[0, :, 0]
        figure, axis = plt.subplots(figsize=(7, 6))
        mesh = axis.pcolormesh(x_axis, y_axis, density_slice.T, shading="auto", cmap="inferno")
        circle = plt.Circle((0, 0), self.config.schwarzschild_radius, color="cyan", fill=False, linewidth=2)
        axis.add_patch(circle)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(f"Probability Density Slice z=0, step {step}")
        figure.colorbar(mesh, ax=axis, label="|psi|^2")
        figure.tight_layout()
        filepath = os.path.join(self.figures_dir, f"{label}_density_step_{step:05d}.png")
        figure.savefig(filepath, dpi=160)
        plt.close(figure)
        return filepath

    def plot_radial_profile(self, psi, step, label):
        mid_y = self.Y.shape[1] // 2
        mid_z = self.Z.shape[2] // 2
        x_axis = self.X[:, 0, 0]
        density_line = np.abs(psi[:, mid_y, mid_z]) ** 2
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(x_axis, density_line, color="navy", linewidth=2)
        axis.axvline(self.config.schwarzschild_radius, color="black", linestyle=":", label="Event Horizon")
        axis.axvline(-self.config.schwarzschild_radius, color="black", linestyle=":")
        axis.set_xlabel("x")
        axis.set_ylabel("|psi(x,0,0)|^2")
        axis.set_title(f"Axial Density Profile, step {step}")
        axis.legend()
        figure.tight_layout()
        filepath = os.path.join(self.figures_dir, f"{label}_profile_step_{step:05d}.png")
        figure.savefig(filepath, dpi=160)
        plt.close(figure)
        return filepath

    def plot_diagnostics(self, records, label):
        times = [r["time"] for r in records]
        norm = [r["norm"] for r in records]
        captured = [r["captured_probability"] for r in records]
        reflected = [r["reflected_probability"] for r in records]
        transmitted = [r["transmitted_probability"] for r in records]
        total_energy = [r["total_energy"] for r in records]
        figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(times, norm, label="Total Norm", color="black")
        axes[0].plot(times, captured, label="Captured", color="red")
        axes[0].plot(times, reflected, label="Reflected", color="blue")
        axes[0].plot(times, transmitted, label="Transmitted", color="green")
        axes[0].set_ylabel("Probability")
        axes[0].legend()
        axes[0].set_title("Probability Diagnostics")
        axes[1].plot(times, total_energy, color="purple")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Total Energy")
        figure.tight_layout()
        filepath = os.path.join(self.figures_dir, f"{label}_diagnostics.png")
        figure.savefig(filepath, dpi=160)
        plt.close(figure)
        return filepath

    def render_isosurface(self, psi, step, label, threshold_fraction=0.12):
        density = np.abs(psi) ** 2
        max_density = np.max(density)
        if max_density <= 0:
            return None
        threshold = threshold_fraction * max_density
        try:
            verts, faces, normals, values = marching_cubes(density, level=threshold)
        except (ValueError, RuntimeError):
            return None
        spacing = np.array([
            self.X[1, 0, 0] - self.X[0, 0, 0],
            self.Y[0, 1, 0] - self.Y[0, 0, 0],
            self.Z[0, 0, 1] - self.Z[0, 0, 0],
        ])
        origin = np.array([self.X[0, 0, 0], self.Y[0, 0, 0], self.Z[0, 0, 0]])
        world_verts = origin + verts * spacing
        figure = plt.figure(figsize=(9, 7))
        axis = figure.add_subplot(111, projection="3d")
        axis.plot_trisurf(
            world_verts[:, 0], world_verts[:, 1], world_verts[:, 2],
            triangles=faces, cmap="inferno", linewidth=0, antialiased=True, alpha=0.85
        )
        angular_samples = np.linspace(0, 2 * np.pi, 30)
        polar_samples = np.linspace(0, np.pi, 30)
        horizon_x = self.config.schwarzschild_radius * np.outer(np.cos(angular_samples), np.sin(polar_samples))
        horizon_y = self.config.schwarzschild_radius * np.outer(np.sin(angular_samples), np.sin(polar_samples))
        horizon_z = self.config.schwarzschild_radius * np.outer(np.ones_like(angular_samples), np.cos(polar_samples))
        axis.plot_surface(horizon_x, horizon_y, horizon_z, color="cyan", alpha=0.3, linewidth=0)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(f"3D Probability Density Isosurface, step {step}")
        figure.tight_layout()
        filepath = os.path.join(self.figures_dir, f"{label}_isosurface_step_{step:05d}.png")
        figure.savefig(filepath, dpi=160)
        plt.close(figure)
        return filepath

    def render_animation(self, frames, label):
        if not frames:
            return None
        mid = self.Z.shape[2] // 2
        x_axis = self.X[:, 0, 0]
        y_axis = self.Y[0, :, 0]
        figure, axis = plt.subplots(figsize=(6, 6))
        initial_density = np.abs(frames[0][:, :, mid]) ** 2
        mesh = axis.pcolormesh(
            x_axis, y_axis, initial_density.T, shading="auto", cmap="inferno",
            vmin=0, vmax=np.max(initial_density) + 1e-9
        )
        circle = plt.Circle((0, 0), self.config.schwarzschild_radius, color="cyan", fill=False, linewidth=2)
        axis.add_patch(circle)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        title = axis.set_title("Wavefunction Evolution")

        def update(frame_index):
            density_slice = np.abs(frames[frame_index][:, :, mid]) ** 2
            mesh.set_array(density_slice.T.ravel())
            title.set_text(f"Wavefunction Evolution, frame {frame_index}")
            return mesh, title

        animation = FuncAnimation(figure, update, frames=len(frames), blit=False)
        filepath = os.path.join(self.animations_dir, f"{label}_evolution.gif")
        writer = PillowWriter(fps=12)
        animation.save(filepath, writer=writer)
        plt.close(figure)
        return filepath


class BlackHoleQuantumSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.potential = SchwarzschildPotential(config)
        self.absorber = AbsorbingBoundary(config)
        self.wave_packet = WavePacket(config)
        self._validate_config()
        self._build_grid()
        self._build_operators()
        self.visualization = Visualization(config, self.X, self.Y, self.Z, self.R)

    def _validate_config(self):
        config = self.config
        if config.Nx <= 0 or config.Ny <= 0 or config.Nz <= 0:
            raise ValueError("Grid resolution must be positive in all dimensions")
        if config.dt <= 0:
            raise ValueError("Time step must be positive")
        if config.total_steps <= 0:
            raise ValueError("Total steps must be positive")
        if config.absorber_width <= 0:
            raise ValueError("Absorber width must be positive")
        if config.packet_width <= 0:
            raise ValueError("Packet width must be positive")
        if config.absorber_width * 2 >= min(config.Lx, config.Ly, config.Lz):
            raise ValueError("Absorber width too large for domain size")

    def _build_grid(self):
        config = self.config
        x = np.linspace(-config.Lx / 2.0, config.Lx / 2.0, config.Nx, endpoint=False)
        y = np.linspace(-config.Ly / 2.0, config.Ly / 2.0, config.Ny, endpoint=False)
        z = np.linspace(-config.Lz / 2.0, config.Lz / 2.0, config.Nz, endpoint=False)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]
        self.dV = self.dx * self.dy * self.dz
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2 + self.Z ** 2)
        kx = 2.0 * np.pi * np.fft.fftfreq(config.Nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(config.Ny, d=self.dy)
        kz = 2.0 * np.pi * np.fft.fftfreq(config.Nz, d=self.dz)
        self.KX, self.KY, self.KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2 + self.KZ ** 2

    def _build_operators(self):
        config = self.config
        self.potential_array = self.potential.evaluate(self.R)
        self.absorber_array = self.absorber.build(self.X, self.Y, self.Z, config)
        self.solver = SplitOperatorSolver(self.potential_array, self.absorber_array, self.K2, config.dt)
        self.capture_mask = self.potential.capture_mask(self.R, config.capture_buffer)
        axis_index = int(np.argmax(np.abs(np.array(config.packet_momentum))))
        momentum_component = config.packet_momentum[axis_index]
        axis_direction = 1.0 if momentum_component >= 0 else -1.0
        axis_origin = config.packet_center[axis_index]
        self.diagnostics = Diagnostics(
            self.X, self.Y, self.Z, self.R, self.K2, self.dV,
            self.potential_array, self.capture_mask,
            axis_index, axis_direction, axis_origin
        )

    def run(self, save_outputs=True, collect_frames=True):
        config = self.config
        psi = self.wave_packet.generate(self.X, self.Y, self.Z)
        initial_norm = np.sum(np.abs(psi) ** 2) * self.dV
        if initial_norm <= 0:
            raise RuntimeError("Initial wave packet has zero norm")
        psi = psi / np.sqrt(initial_norm)
        frames = []
        if collect_frames:
            frames.append(psi.copy())
        self.diagnostics.evaluate(psi, 0.0)
        for step in range(1, config.total_steps + 1):
            psi = self.solver.step(psi)
            current_norm = np.sum(np.abs(psi) ** 2) * self.dV
            if np.isnan(current_norm) or np.isinf(current_norm):
                raise RuntimeError(f"Simulation became unstable at step {step}")
            self.diagnostics.evaluate(psi, step * config.dt)
            if collect_frames and step % config.snapshot_interval == 0:
                frames.append(psi.copy())
        self.final_psi = psi
        self.frames = frames
        self.result_figures = {}
        if save_outputs:
            self.result_figures = self.save_results()
        return self.diagnostics.records[-1]

    def save_results(self):
        config = self.config
        os.makedirs(config.output_dir, exist_ok=True)
        csv_dir = os.path.join(config.output_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"{config.run_label}_diagnostics.csv")
        self.diagnostics.export_csv(csv_path)
        density_path = self.visualization.plot_density_slice(self.final_psi, config.total_steps, config.run_label)
        profile_path = self.visualization.plot_radial_profile(self.final_psi, config.total_steps, config.run_label)
        diagnostics_path = self.visualization.plot_diagnostics(self.diagnostics.records, config.run_label)
        isosurface_path = self.visualization.render_isosurface(self.final_psi, config.total_steps, config.run_label)
        animation_path = None
        if self.frames:
            animation_path = self.visualization.render_animation(self.frames, config.run_label)
        return {
            "density_slice": density_path,
            "radial_profile": profile_path,
            "diagnostics": diagnostics_path,
            "isosurface": isosurface_path,
            "animation": animation_path,
            "csv": csv_path,
        }

    def summary(self):
        final_record = self.diagnostics.records[-1]
        print(f"Run: {self.config.run_label}")
        print(f"Final norm: {final_record['norm']:.6f}")
        print(f"Captured probability: {final_record['captured_probability']:.6f}")
        print(f"Reflected probability: {final_record['reflected_probability']:.6f}")
        print(f"Transmitted probability: {final_record['transmitted_probability']:.6f}")
        print(f"Total energy: {final_record['total_energy']:.6f}")


class ParameterSweep:
    def __init__(self, base_config: SimulationConfig):
        self.base_config = base_config
        self.results_dir = os.path.join(base_config.output_dir, "sweeps")
        os.makedirs(self.results_dir, exist_ok=True)

    def _run_single(self, config: SimulationConfig):
        simulation = BlackHoleQuantumSimulation(config)
        final_record = simulation.run(save_outputs=False, collect_frames=False)
        return final_record

    def _execute(self, parameter_name, values, config_updater, sweep_label):
        rows = []
        for value in values:
            trial_config = config_updater(self.base_config, value)
            trial_config.run_label = f"{sweep_label}_{value}"
            final_record = self._run_single(trial_config)
            row = {parameter_name: value}
            row.update(final_record)
            rows.append(row)
        self._export_csv(rows, sweep_label)
        self._plot_sweep(rows, parameter_name, sweep_label)
        return rows

    def _export_csv(self, rows, sweep_label):
        if not rows:
            return
        filepath = os.path.join(self.results_dir, f"{sweep_label}_results.csv")
        fieldnames = list(rows[0].keys())
        with open(filepath, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _plot_sweep(self, rows, parameter_name, sweep_label):
        values = [row[parameter_name] for row in rows]
        captured = [row["captured_probability"] for row in rows]
        reflected = [row["reflected_probability"] for row in rows]
        transmitted = [row["transmitted_probability"] for row in rows]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(values, captured, marker="o", label="Captured", color="red")
        axis.plot(values, reflected, marker="o", label="Reflected", color="blue")
        axis.plot(values, transmitted, marker="o", label="Transmitted", color="green")
        axis.set_xlabel(parameter_name)
        axis.set_ylabel("Final Probability")
        axis.set_title(f"Parameter Sweep: {sweep_label}")
        axis.legend()
        figure.tight_layout()
        filepath = os.path.join(self.results_dir, f"{sweep_label}_plot.png")
        figure.savefig(filepath, dpi=160)
        plt.close(figure)

    def sweep_momentum(self, values):
        def updater(base, value):
            momentum = (value, base.packet_momentum[1], base.packet_momentum[2])
            return replace(base, packet_momentum=momentum, total_steps=base.total_steps // 2,
                            Nx=32, Ny=32, Nz=32)
        return self._execute("momentum", values, updater, "momentum_sweep")

    def sweep_packet_width(self, values):
        def updater(base, value):
            return replace(base, packet_width=value, total_steps=base.total_steps // 2,
                            Nx=32, Ny=32, Nz=32)
        return self._execute("packet_width", values, updater, "width_sweep")

    def sweep_horizon_radius(self, values):
        def updater(base, value):
            return replace(base, schwarzschild_radius=value, total_steps=base.total_steps // 2,
                            Nx=32, Ny=32, Nz=32)
        return self._execute("schwarzschild_radius", values, updater, "horizon_sweep")

    def sweep_absorber_strength(self, values):
        def updater(base, value):
            return replace(base, absorber_strength=value, total_steps=base.total_steps // 2,
                            Nx=32, Ny=32, Nz=32)
        return self._execute("absorber_strength", values, updater, "absorber_sweep")

    def sweep_time_step(self, values):
        def updater(base, value):
            steps = max(50, int((base.dt * base.total_steps) / value))
            return replace(base, dt=value, total_steps=min(steps, 200), Nx=32, Ny=32, Nz=32)
        return self._execute("dt", values, updater, "timestep_sweep")


class ConvergenceTest:
    def __init__(self, base_config: SimulationConfig):
        self.base_config = base_config
        self.results_dir = os.path.join(base_config.output_dir, "convergence")
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self, resolutions):
        rows = []
        for resolution in resolutions:
            trial_config = replace(
                self.base_config,
                Nx=resolution, Ny=resolution, Nz=resolution,
                total_steps=max(60, self.base_config.total_steps // 4),
                run_label=f"convergence_{resolution}"
            )
            simulation = BlackHoleQuantumSimulation(trial_config)
            final_record = simulation.run(save_outputs=False, collect_frames=False)
            row = {"resolution": resolution}
            row.update(final_record)
            rows.append(row)
        self._export_csv(rows)
        self._plot(rows)
        return rows

    def _export_csv(self, rows):
        filepath = os.path.join(self.results_dir, "convergence_results.csv")
        fieldnames = list(rows[0].keys())
        with open(filepath, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _plot(self, rows):
        resolutions = [row["resolution"] for row in rows]
        norms = [row["norm"] for row in rows]
        captured = [row["captured_probability"] for row in rows]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(resolutions, norms, marker="o", label="Final Norm", color="black")
        axis.plot(resolutions, captured, marker="o", label="Captured Probability", color="red")
        axis.set_xlabel("Grid Resolution (points per axis)")
        axis.set_ylabel("Value")
        axis.set_title("Numerical Convergence Test")
        axis.legend()
        figure.tight_layout()
        filepath = os.path.join(self.results_dir, "convergence_plot.png")
        figure.savefig(filepath, dpi=160)
        plt.close(figure)


class ReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name="SectionHeading", parent=self.styles["Heading1"], spaceBefore=14, spaceAfter=8))
        self.styles.add(ParagraphStyle(name="SubHeading", parent=self.styles["Heading2"], spaceBefore=10, spaceAfter=6))

    def _table_style(self):
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ])

    def _image_flowable(self, path, max_width=6.4 * inch):
        if not path or not os.path.exists(path):
            return None
        with PILImage.open(path) as pil_image:
            original_width, original_height = pil_image.size
        aspect = original_height / original_width
        width = max_width
        height = width * aspect
        return RLImage(path, width=width, height=height)

    def _records_table(self, record, decimals=6):
        data = [["Quantity", "Value"]]
        for key, value in record.items():
            data.append([key, f"{value:.{decimals}f}"])
        table = Table(data, colWidths=[2.8 * inch, 2.8 * inch])
        table.setStyle(self._table_style())
        return table

    DISPLAY_NAMES = {
        "norm": "Norm",
        "captured_probability": "Captured",
        "reflected_probability": "Reflected",
        "transmitted_probability": "Transmitted",
        "total_energy": "Total Energy",
        "momentum": "Momentum",
        "packet_width": "Width",
        "schwarzschild_radius": "Horizon Rs",
        "absorber_strength": "Absorber",
        "dt": "Time Step",
        "resolution": "Resolution",
    }

    def _sweep_table(self, rows, decimals=4):
        if not rows:
            return None
        all_fieldnames = list(rows[0].keys())
        parameter_field = all_fieldnames[0]
        metric_fields = [
            field for field in
            ["norm", "captured_probability", "reflected_probability", "transmitted_probability", "total_energy"]
            if field in all_fieldnames
        ]
        fieldnames = [parameter_field] + metric_fields
        header_row = [self.DISPLAY_NAMES.get(field, field) for field in fieldnames]
        data = [header_row]
        for row in rows:
            formatted_row = []
            for field in fieldnames:
                value = row[field]
                if isinstance(value, float):
                    formatted_row.append(f"{value:.{decimals}f}")
                else:
                    formatted_row.append(str(value))
            data.append(formatted_row)
        table = Table(data, repeatRows=1, colWidths=[5.6 * inch / len(fieldnames)] * len(fieldnames))
        table.setStyle(self._table_style())
        return table

    def build(self, config, baseline_record, baseline_figures, sweep_data, convergence_rows, convergence_figure_path):
        report_path = os.path.join(self.output_dir, "research_report.pdf")
        doc = SimpleDocTemplate(
            report_path, pagesize=letter,
            topMargin=0.7 * inch, bottomMargin=0.7 * inch,
            leftMargin=0.7 * inch, rightMargin=0.7 * inch
        )
        story = []

        story.append(Paragraph("Quantum Wave Packet Scattering off a Schwarzschild Black Hole", self.styles["Title"]))
        story.append(Paragraph("A 3D Split-Operator Fourier Simulation Study", self.styles["Heading3"]))
        story.append(Spacer(1, 16))
        intro_text = (
            "This report documents a fully three-dimensional numerical simulation of a non-relativistic "
            "quantum wave packet propagating in the presence of a Schwarzschild-inspired central potential. "
            "The wavefunction is evolved using a second-order split-operator Fourier method on a Cartesian "
            "grid, with complex absorbing boundary layers applied at every face of the simulation domain. "
            "Capture, reflection, and transmission probabilities are computed automatically by partitioning "
            "the computational domain relative to the event horizon and the initial direction of motion."
        )
        story.append(Paragraph(intro_text, self.styles["Normal"]))
        story.append(Spacer(1, 10))

        story.append(Paragraph("1. Simulation Configuration", self.styles["SectionHeading"]))
        config_rows = [
            ["Grid resolution", f"{config.Nx} x {config.Ny} x {config.Nz}"],
            ["Domain size", f"{config.Lx} x {config.Ly} x {config.Lz}"],
            ["Schwarzschild radius", f"{config.schwarzschild_radius}"],
            ["Potential strength", f"{config.potential_strength}"],
            ["Absorber width / strength", f"{config.absorber_width} / {config.absorber_strength}"],
            ["Initial packet center", f"{config.packet_center}"],
            ["Initial momentum", f"{config.packet_momentum}"],
            ["Packet width (sigma)", f"{config.packet_width}"],
            ["Time step", f"{config.dt}"],
            ["Total steps", f"{config.total_steps}"],
        ]
        config_table = Table([["Parameter", "Value"]] + config_rows, colWidths=[2.8 * inch, 3.6 * inch])
        config_table.setStyle(self._table_style())
        story.append(config_table)
        story.append(PageBreak())

        story.append(Paragraph("2. Baseline Run Results", self.styles["SectionHeading"]))
        story.append(self._records_table(baseline_record))
        story.append(Spacer(1, 14))
        for label, key in [
            ("Probability Density Slice (z = 0)", "density_slice"),
            ("Axial Density Profile", "radial_profile"),
            ("Probability and Energy Diagnostics", "diagnostics"),
            ("3D Probability Density Isosurface", "isosurface"),
        ]:
            path = baseline_figures.get(key)
            image = self._image_flowable(path)
            if image is not None:
                story.append(KeepTogether([
                    Paragraph(label, self.styles["SubHeading"]),
                    image,
                    Spacer(1, 10),
                ]))
        story.append(PageBreak())

        story.append(Paragraph("3. Parameter Sweeps", self.styles["SectionHeading"]))
        for sweep_label, sweep_payload in sweep_data.items():
            rows, plot_path = sweep_payload
            block = [Paragraph(sweep_label.replace("_", " ").title(), self.styles["SubHeading"])]
            table = self._sweep_table(rows)
            if table is not None:
                block.append(table)
                block.append(Spacer(1, 8))
            image = self._image_flowable(plot_path, max_width=5.6 * inch)
            if image is not None:
                block.append(image)
            block.append(Spacer(1, 14))
            story.append(KeepTogether(block))
        story.append(PageBreak())

        story.append(Paragraph("4. Numerical Convergence Test", self.styles["SectionHeading"]))
        convergence_table = self._sweep_table(convergence_rows)
        if convergence_table is not None:
            story.append(convergence_table)
            story.append(Spacer(1, 10))
        convergence_image = self._image_flowable(convergence_figure_path, max_width=5.6 * inch)
        if convergence_image is not None:
            story.append(convergence_image)
        story.append(Spacer(1, 14))

        story.append(Paragraph("5. Summary", self.styles["SectionHeading"]))
        summary_text = (
            f"Under the baseline configuration, the simulation evolved a Gaussian wave packet with initial "
            f"momentum {config.packet_momentum} toward a Schwarzschild-inspired potential of horizon radius "
            f"{config.schwarzschild_radius}. The final captured probability was "
            f"{baseline_record['captured_probability']:.6f}, the reflected probability was "
            f"{baseline_record['reflected_probability']:.6f}, and the transmitted probability was "
            f"{baseline_record['transmitted_probability']:.6f}. The parameter sweeps and resolution "
            f"convergence test above characterize the sensitivity of these outcomes to the physical and "
            f"numerical parameters of the model."
        )
        story.append(Paragraph(summary_text, self.styles["Normal"]))

        doc.build(story)
        return report_path


def main():
    base_config = SimulationConfig()
    simulation = BlackHoleQuantumSimulation(base_config)
    simulation.run()
    simulation.summary()
    baseline_record = simulation.diagnostics.records[-1]
    baseline_figures = simulation.result_figures

    sweep = ParameterSweep(base_config)
    sweep_data = {}
    sweep_data["momentum_sweep"] = (
        sweep.sweep_momentum([1.5, 2.5, 3.5, 4.5, 5.5]),
        os.path.join(sweep.results_dir, "momentum_sweep_plot.png"),
    )
    sweep_data["width_sweep"] = (
        sweep.sweep_packet_width([1.5, 2.5, 3.5, 4.5, 5.5]),
        os.path.join(sweep.results_dir, "width_sweep_plot.png"),
    )
    sweep_data["horizon_sweep"] = (
        sweep.sweep_horizon_radius([2.0, 3.0, 4.0, 5.0, 6.0]),
        os.path.join(sweep.results_dir, "horizon_sweep_plot.png"),
    )
    sweep_data["absorber_sweep"] = (
        sweep.sweep_absorber_strength([10.0, 25.0, 40.0, 55.0, 70.0]),
        os.path.join(sweep.results_dir, "absorber_sweep_plot.png"),
    )
    sweep_data["timestep_sweep"] = (
        sweep.sweep_time_step([0.02, 0.015, 0.01, 0.0075, 0.005]),
        os.path.join(sweep.results_dir, "timestep_sweep_plot.png"),
    )

    convergence = ConvergenceTest(base_config)
    convergence_rows = convergence.run([16, 24, 32, 40])
    convergence_figure_path = os.path.join(convergence.results_dir, "convergence_plot.png")

    report_generator = ReportGenerator(base_config.output_dir)
    report_path = report_generator.build(
        base_config, baseline_record, baseline_figures,
        sweep_data, convergence_rows, convergence_figure_path
    )
    print(f"Research report saved to {report_path}")

    print("All simulations, sweeps, and convergence tests complete.")


if __name__ == "__main__":
    main()
