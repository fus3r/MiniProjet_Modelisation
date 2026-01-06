#!/usr/bin/env python3
"""
MODULE DE SIMULATION À ÉVÉNEMENTS DISCRETS (DES)
------------------------------------------------
Correspondance avec le cours "Systèmes à Événements Discrets" (TD2):
Ce simulateur implémente un Réseau de Petri Temporisé Stochastique (RdP-TS).

1. PLACES (P) : Les compartiments de population
   P = {S, I, D, T, H, E}
   Le marquage M(P) correspond au nombre d'individus (entiers) dans chaque état.

2. TRANSITIONS (Tr) : Les événements épidémiques
   - Tr_Infection : S + I -> 2I (Tir conditionné par la présence de jetons dans S et I)
   - Tr_Detection : I -> D
   - Tr_Hospitalisation : D -> T
   - Tr_Guerison : I->H, D->H, T->H
   - Tr_Deces : T -> E

3. TEMPORISATION :
   Les tirs de transitions suivent une loi exponentielle (Processus de Poisson).
   L'algorithme utilisé est celui de Gillespie (SSA).
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple
import heapq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sidthe.params import x0, theta_nom, T_MAX, DT
from sidthe.integrators import simulate_days


class Mode(Enum):
    """Epidemic control modes."""
    NORMAL = auto()
    ALERT = auto()
    LOCKDOWN = auto()
    RECOVERY = auto()


@dataclass
class Event:
    """Discrete event with time and type."""
    time: float
    event_type: str
    data: dict = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.time < other.time


class EpidemicDES:
    """
    Discrete Event Simulator for epidemic mode management.
    
    Combines continuous SIDTHE dynamics with discrete mode transitions.
    """
    
    # Thresholds (relative to T_MAX)
    ALERT_THRESHOLD = 0.6  # Enter ALERT if T > 0.6 * T_MAX
    CRITICAL_THRESHOLD = 0.95  # Enter LOCKDOWN if T > 0.95 * T_MAX
    RECOVERY_THRESHOLD = 0.2  # Enter RECOVERY if T < 0.2 * T_MAX
    
    # Control intensities per mode
    MODE_CONTROL = {
        Mode.NORMAL: 0.0,
        Mode.ALERT: 0.3,
        Mode.LOCKDOWN: 0.7,
        Mode.RECOVERY: 0.1,
    }
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.event_queue: List[Event] = []
        self.mode = Mode.NORMAL
        self.mode_history: List[Tuple[float, Mode]] = []
        self.x = x0.copy()
        self.theta = theta_nom
        self.time = 0.0
        
    def schedule_event(self, event: Event):
        """Add event to priority queue."""
        heapq.heappush(self.event_queue, event)
    
    def schedule_random_events(self, horizon: float):
        """Schedule stochastic events (variant, vaccine)."""
        # Variant arrival: Poisson process, rate ~ 1/120 days
        t = 0.0
        while t < horizon:
            dt = self.rng.exponential(120)
            t += dt
            if t < horizon and t > 60:  # No variant in first 60 days
                self.schedule_event(Event(t, "VARIANT", {"factor": 1.1}))
        
        # Vaccine: deterministic rollout starting day 180
        if horizon > 180:
            self.schedule_event(Event(180, "VACCINE", {"reduction": 0.02}))
            self.schedule_event(Event(240, "VACCINE", {"reduction": 0.03}))
            self.schedule_event(Event(300, "VACCINE", {"reduction": 0.05}))
    
    def check_icu_transition(self) -> Mode | None:
        """Check if ICU level triggers mode change."""
        T = self.x[3]  # ICU fraction
        
        if T > self.CRITICAL_THRESHOLD * T_MAX:
            if self.mode != Mode.LOCKDOWN:
                return Mode.LOCKDOWN
        elif T > self.ALERT_THRESHOLD * T_MAX:
            if self.mode == Mode.NORMAL:
                return Mode.ALERT
        elif T < self.RECOVERY_THRESHOLD * T_MAX:
            if self.mode in (Mode.ALERT, Mode.LOCKDOWN):
                return Mode.RECOVERY
            elif self.mode == Mode.RECOVERY and T < 0.1 * T_MAX:
                return Mode.NORMAL
        
        return None
    
    def apply_event(self, event: Event):
        """Process a discrete event."""
        if event.event_type == "VARIANT":
            # Increase transmission rate
            factor = event.data.get("factor", 1.1)
            self.theta = type(self.theta)(
                alpha=self.theta.alpha * factor,
                gamma=self.theta.gamma,
                lam=self.theta.lam,
                delta=self.theta.delta,
                sigma=self.theta.sigma,
                tau=self.theta.tau,
            )
            print(f"  Day {event.time:.0f}: VARIANT (α × {factor:.2f})")
            
        elif event.event_type == "VACCINE":
            # Reduce susceptible pool
            reduction = event.data.get("reduction", 0.02)
            self.x[0] = max(0.0, self.x[0] - reduction)  # S -= reduction
            self.x[4] += reduction  # H += reduction (healed/immune)
            print(f"  Day {event.time:.0f}: VACCINE (S -= {reduction:.0%})")
    
    def transition_mode(self, new_mode: Mode):
        """Execute mode transition."""
        old_mode = self.mode
        self.mode = new_mode
        self.mode_history.append((self.time, new_mode))
        print(f"  Day {self.time:.0f}: {old_mode.name} → {new_mode.name}")
    
    def simulate(self, horizon: float = 350) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Run the hybrid simulation.
        
        Returns
        -------
        ts : np.ndarray
            Time points
        xs : np.ndarray
            State trajectory
        mode_history : List[Tuple[float, Mode]]
            Mode transitions
        """
        print("=" * 60)
        print("DES Epidemic Simulator")
        print("=" * 60)
        
        # Initialize
        self.mode_history = [(0.0, Mode.NORMAL)]
        self.schedule_random_events(horizon)
        
        ts_all = [0.0]
        xs_all = [self.x.copy()]
        
        day = 0
        while day < horizon:
            # Process any events at current time
            while self.event_queue and self.event_queue[0].time <= day:
                event = heapq.heappop(self.event_queue)
                self.apply_event(event)
            
            # Get control for current mode
            u = self.MODE_CONTROL[self.mode]
            
            # Simulate one day
            _, xs_step = simulate_days(self.x, self.theta, np.array([u]), dt=DT)
            self.x = xs_step[-1]
            self.time = day + 1
            
            ts_all.append(self.time)
            xs_all.append(self.x.copy())
            
            # Check for mode transitions
            new_mode = self.check_icu_transition()
            if new_mode is not None:
                self.transition_mode(new_mode)
            
            day += 1
        
        print("-" * 60)
        print(f"Final mode: {self.mode.name}")
        print(f"Mode transitions: {len(self.mode_history)}")
        
        return np.array(ts_all), np.array(xs_all), self.mode_history


def plot_des_trace(ts: np.ndarray, xs: np.ndarray, mode_history: List, save_path: Path):
    """Create figure showing DES trace with mode regions."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Color map for modes
    mode_colors = {
        Mode.NORMAL: "#2ecc71",     # Green
        Mode.ALERT: "#f39c12",      # Orange
        Mode.LOCKDOWN: "#e74c3c",   # Red
        Mode.RECOVERY: "#3498db",   # Blue
    }
    
    # Add mode regions to all subplots
    for ax in axes:
        for i, (t_start, mode) in enumerate(mode_history):
            t_end = mode_history[i + 1][0] if i + 1 < len(mode_history) else ts[-1]
            ax.axvspan(t_start, t_end, alpha=0.15, color=mode_colors[mode])
    
    # Top: ICU (T)
    ax1 = axes[0]
    ax1.plot(ts, xs[:, 3] * 100, "k-", linewidth=1.5)
    ax1.axhline(y=100 * T_MAX, color="r", linestyle="--", linewidth=1, label="ICU threshold")
    ax1.axhline(y=100 * T_MAX * 0.6, color="orange", linestyle=":", linewidth=1, label="Alert threshold")
    ax1.set_ylabel("% ICU", fontsize=11)
    ax1.set_title("DES Epidemic Simulator: Mode Transitions", fontsize=13)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Middle: Infected (I) and Diagnosed (D)
    ax2 = axes[1]
    ax2.plot(ts, xs[:, 1] * 100, "b-", linewidth=1.5, label="Infected (I)")
    ax2.plot(ts, xs[:, 2] * 100, "m-", linewidth=1.5, label="Diagnosed (D)")
    ax2.set_ylabel("% Population", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Bottom: Mode timeline
    ax3 = axes[2]
    mode_y = {Mode.NORMAL: 0, Mode.ALERT: 1, Mode.LOCKDOWN: 2, Mode.RECOVERY: 3}
    mode_names = [m.name for m in Mode]
    
    # Plot mode as step function
    t_plot = []
    m_plot = []
    for i, (t, mode) in enumerate(mode_history):
        t_plot.append(t)
        m_plot.append(mode_y[mode])
        if i + 1 < len(mode_history):
            t_plot.append(mode_history[i + 1][0])
            m_plot.append(mode_y[mode])
    t_plot.append(ts[-1])
    m_plot.append(m_plot[-1])
    
    ax3.step(t_plot, m_plot, where="post", linewidth=2, color="black")
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(mode_names, fontsize=10)
    ax3.set_xlabel("Time [days]", fontsize=11)
    ax3.set_ylabel("Mode", fontsize=11)
    ax3.set_ylim(-0.5, 3.5)
    ax3.grid(True, alpha=0.3, axis="x")
    
    # Legend for modes
    patches = [mpatches.Patch(color=mode_colors[m], alpha=0.4, label=m.name) for m in Mode]
    ax3.legend(handles=patches, loc="upper right", fontsize=9, ncol=4)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"\nFigure saved: {save_path}")
    print(f"Figure saved: {save_path.with_suffix('.pdf')}")


def main() -> int:
    # Run simulation
    des = EpidemicDES(seed=42)
    ts, xs, mode_history = des.simulate(horizon=350)
    
    # Create output directory and plot
    output_dir = REPO_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_des_trace(ts, xs, mode_history, output_dir / "des_trace.png")
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
