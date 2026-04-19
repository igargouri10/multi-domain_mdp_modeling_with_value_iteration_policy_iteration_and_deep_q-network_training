r"""End-to-end experiment driver for the Mini-2 report.

Runs every solver on every problem, stores numerical results in
``data/results.json``, and writes the Matplotlib figures that
``docs/report/report.tex`` consumes into ``docs/report/figures/``.

This script is intentionally self-contained so the TA can reproduce every
figure in the report with one command.

References
----------
R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*,
2nd ed.  MIT Press, 2018.
V. Mnih *et al.*, "Human-level control through deep reinforcement
learning," *Nature*, vol. 518, pp. 529–533, 2015.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..problems.windy_chasm import Action, WindyChasmSpec, WindyChasmMDP
from ..problems.windy_chasm_variants import (
    ChallengeAFineGrid, FineGridSpec,
    ChallengeBContinuousX, ContinuousXSpec,
    ChallengeCVelocity, VelocitySpec,
    ChallengeDIntegralWind, IntegralWindSpec,
)
from ..problems.lidar_cone import (
    LidarConeSpec,
    LidarConeMDP,
    policy_iteration_discussion,
)
from ..problems.circuit_design import (
    CircuitDesignSpec,
    CircuitDesignMDP,
    CircuitSimulator,
    Cell,
    state_space_size,
)
from ..mdp.dp import policy_iteration, value_iteration
from ..agents.dqn_numpy import DQNConfig, train_dqn_on_mdp


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
FIG_DIR = REPO_ROOT / "docs" / "report" / "figures"


# ---------------------------------------------------------------------------
def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def run_problem1() -> Dict[str, Any]:
    """Problem 1: 20x7 Windy Chasm, VI/PI, sweep over B and r_crash."""
    spec = WindyChasmSpec()
    mdp = WindyChasmMDP(spec)
    start_idx = mdp._state_index_public[mdp.start_state()]

    V_vi, pi_vi, sw_vi = value_iteration(mdp, tol=1e-10)
    V_pi, pi_pi, sw_pi = policy_iteration(mdp)
    agreement = float(np.max(np.abs(V_vi - V_pi)))
    V_start = float(V_vi[start_idx])

    # Sweep over wind strength B
    B_sweep = [0.1, 0.2, 0.4, 0.6, 0.8]
    sweep_B = []
    for B in B_sweep:
        s2 = WindyChasmSpec(B=B)
        m2 = WindyChasmMDP(s2)
        V, _, _ = value_iteration(m2, tol=1e-9)
        si = m2._state_index_public[m2.start_state()]
        sweep_B.append({"B": B, "V_start": float(V[si])})

    # Sweep over crash penalty
    r_sweep = [1.0, 5.0, 20.0, 50.0, 100.0]
    sweep_r = []
    for r in r_sweep:
        s2 = WindyChasmSpec(r_crash=r)
        m2 = WindyChasmMDP(s2)
        V, pi, _ = value_iteration(m2, tol=1e-9)
        si = m2._state_index_public[m2.start_state()]
        # Count of cells whose greedy action is FORWARD
        policy_grid = m2.policy_grid(pi)
        forward_frac = float((policy_grid == Action.FORWARD.value).sum()) / max(
            1, (policy_grid >= 0).sum()
        )
        sweep_r.append({
            "r_crash": r, "V_start": float(V[si]),
            "forward_frac": forward_frac,
        })

    # Figure: value heatmap + policy arrows
    fig, (axv, axp) = plt.subplots(1, 2, figsize=(10, 5))
    Vgrid = mdp.value_grid(V_vi)
    im = axv.imshow(Vgrid, origin="upper", cmap="viridis")
    axv.set_title("Optimal $V^*(i, j)$ on Windy Chasm")
    axv.set_xlabel("column $j$"); axv.set_ylabel("row $i$")
    fig.colorbar(im, ax=axv, fraction=0.046, pad=0.04)
    PG = mdp.policy_grid(pi_vi)
    axp.imshow(Vgrid, origin="upper", cmap="viridis", alpha=0.35)
    # Draw arrows: FORWARD: down; LEFT: left; RIGHT: right
    arrow_map = {Action.FORWARD.value: (1, 0), Action.LEFT.value: (0, -1), Action.RIGHT.value: (0, 1)}
    for i in range(PG.shape[0]):
        for j in range(PG.shape[1]):
            a = PG[i, j]
            if a not in arrow_map:
                continue
            di, dj = arrow_map[a]
            axp.arrow(j, i, 0.3 * dj, 0.3 * di, head_width=0.2,
                      head_length=0.1, fc="white", ec="black", linewidth=0.5)
    axp.set_title(r"Greedy policy $\pi^*$")
    axp.set_xlabel("column $j$"); axp.set_ylabel("row $i$")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p1_value_policy.pdf")
    plt.close(fig)

    # Figure: value vs B and value vs r_crash
    fig, (axb, axr) = plt.subplots(1, 2, figsize=(10, 4))
    axb.plot([d["B"] for d in sweep_B], [d["V_start"] for d in sweep_B], "o-")
    axb.set_xlabel("baseline wind $B$"); axb.set_ylabel("$V^*$(start)")
    axb.set_title("Windier → safer route preferred")
    axb.grid(alpha=0.3)
    axr.plot([d["r_crash"] for d in sweep_r], [d["V_start"] for d in sweep_r], "s-",
             label="$V^*$(start)")
    axr.set_xlabel("crash penalty $r$"); axr.set_ylabel("$V^*$(start)")
    axr2 = axr.twinx()
    axr2.plot([d["r_crash"] for d in sweep_r],
              [d["forward_frac"] for d in sweep_r], "d--", color="tab:orange",
              label="fraction of cells choosing FORWARD")
    axr2.set_ylabel("fraction FORWARD")
    axr.set_title("Harsher crash penalty → cautious drone")
    axr.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p1_sweeps.pdf")
    plt.close(fig)

    return {
        "num_states": mdp.num_states,
        "vi_sweeps": int(sw_vi), "pi_sweeps": int(sw_pi),
        "V_start": V_start, "agreement_vi_pi": agreement,
        "sweep_B": sweep_B, "sweep_r": sweep_r,
    }


def run_problem1_challenge() -> Dict[str, Any]:
    """Problem 1 CHALLENGE parts (a)-(d)."""
    results: Dict[str, Any] = {}

    m_a = ChallengeAFineGrid(FineGridSpec(subdiv=3))
    V_a, _, sw_a = value_iteration(m_a, tol=1e-8)
    results["a_fine_grid"] = {
        "num_states": m_a.num_states, "sweeps": int(sw_a),
        "V_start_fine": float(V_a[m_a._state_index_public[m_a.start_state()]]),
    }

    m_b = ChallengeBContinuousX(ContinuousXSpec(dx=0.05))
    V_b, _, sw_b = value_iteration(m_b, tol=1e-7)
    results["b_continuous_x"] = {
        "num_states": m_b.num_states, "sweeps": int(sw_b),
        "V_start": float(V_b[m_b._state_index_public[m_b.start_state()]]),
    }

    m_c = ChallengeCVelocity(VelocitySpec(v=1.5))
    V_c, _, sw_c = value_iteration(m_c, tol=1e-8)
    results["c_velocity"] = {
        "num_states": m_c.num_states, "sweeps": int(sw_c),
        "V_start": float(V_c[m_c._state_index_public[m_c.start_state()]]),
    }

    m_d = ChallengeDIntegralWind(IntegralWindSpec(v=1.5, t=0.7))
    V_d, _, sw_d = value_iteration(m_d, tol=1e-8)
    results["d_integral_wind"] = {
        "num_states": m_d.num_states, "sweeps": int(sw_d),
        "V_start": float(V_d[m_d._state_index_public[m_d.start_state()]]),
    }

    # Figure: bar chart of V* at start for the four variants
    labels = ["(a) fine grid", "(b) continuous x", "(c) velocity", "(d) integral wind"]
    V_vals = [results["a_fine_grid"]["V_start_fine"],
              results["b_continuous_x"]["V_start"],
              results["c_velocity"]["V_start"],
              results["d_integral_wind"]["V_start"]]
    n_states_list = [results[k]["num_states"] for k in
                     ("a_fine_grid", "b_continuous_x", "c_velocity", "d_integral_wind")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(range(4), V_vals, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    ax1.set_xticks(range(4)); ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("$V^*$ at start"); ax1.set_title("Challenge variants: starting value")
    ax1.grid(alpha=0.3)
    ax2.bar(range(4), n_states_list, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("$|\\mathcal{S}|$"); ax2.set_title("State-space growth with discretisation")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p1_challenge.pdf")
    plt.close(fig)

    return results


def run_problem2() -> Dict[str, Any]:
    spec = LidarConeSpec()
    mdp = LidarConeMDP(spec)
    start_idx = mdp._state_index_public[spec.start]
    V_vi, pi_vi, sw_vi = value_iteration(mdp, tol=1e-10)
    V_pi, pi_pi, sw_pi = policy_iteration(mdp)
    # Sweep over shaping coefficient lambda
    lam_sweep = [0.0, 0.1, 0.2, 0.5, 1.0]
    rows = []
    for lam in lam_sweep:
        s2 = LidarConeSpec(lambda_shape=lam)
        m2 = LidarConeMDP(s2)
        V, _, _ = value_iteration(m2, tol=1e-9)
        si = m2._state_index_public[s2.start]
        rows.append({"lambda": lam, "V_start": float(V[si])})
    Vgrid = mdp.value_grid(V_vi)
    PG = mdp.policy_grid(pi_vi)

    # Figure: value heatmap (diamond corridor) + policy arrows
    fig, (axv, axp) = plt.subplots(1, 2, figsize=(10, 4.5))
    im = axv.imshow(Vgrid, origin="upper", cmap="magma")
    axv.set_title("Lidar Cone $V^*$")
    axv.set_xlabel("col"); axv.set_ylabel("row")
    fig.colorbar(im, ax=axv, fraction=0.046, pad=0.04)
    arrow_map = {0: (0, 1), 1: (-1, 0), 2: (1, 0)}  # FORWARD, CORRECT_UP, CORRECT_DOWN
    axp.imshow(Vgrid, origin="upper", cmap="magma", alpha=0.35)
    for i in range(PG.shape[0]):
        for j in range(PG.shape[1]):
            a = PG[i, j]
            if a not in arrow_map:
                continue
            di, dj = arrow_map[a]
            axp.arrow(j, i, 0.3 * dj, 0.3 * di, head_width=0.2,
                      head_length=0.1, fc="white", ec="black", linewidth=0.5)
    axp.set_title("Greedy $\\pi^*$ (lidar cone)")
    axp.set_xlabel("col"); axp.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p2_value_policy.pdf")
    plt.close(fig)

    # Figure: shaping sweep
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["lambda"] for r in rows], [r["V_start"] for r in rows], "o-")
    ax.set_xlabel(r"shaping coefficient $\lambda$")
    ax.set_ylabel("$V^*$(start)")
    ax.set_title("Reward shaping ($\\lambda |\\Delta|$) is policy-invariant\n"
                 "(Ng, Harada, Russell 1999) — curve flattens once shaping floor is paid")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p2_shaping_sweep.pdf")
    plt.close(fig)

    return {
        "num_states": mdp.num_states,
        "vi_sweeps": int(sw_vi), "pi_sweeps": int(sw_pi),
        "V_start": float(V_vi[start_idx]),
        "agreement": float(np.max(np.abs(V_vi - V_pi))),
        "shaping_sweep": rows,
        "policy_iteration_note": policy_iteration_discussion(),
    }


def run_problem3() -> Dict[str, Any]:
    """Problem 3: tabular MDP + complexity scaling."""
    # State-space scaling for report Q(d)
    scaling = {
        f"{h}x{w}": state_space_size(h, w)
        for (h, w) in [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]
    }

    # Small MDP: 3x3 with 2-component budget, max_steps=6
    spec = CircuitDesignSpec(max_steps=6)
    mdp = CircuitDesignMDP(spec, max_components=2)
    V_vi, pi_vi, sw_vi = value_iteration(mdp, tol=1e-8)
    V_pi, pi_pi, sw_pi = policy_iteration(mdp, max_sweeps=500)
    start_idx = 0  # by construction, state 0 is the BFS root
    # Test: design a hand-crafted correct XOR-from-NAND circuit and score it
    # Standard 4-NAND XOR — but our 3x3 grid is too small for 4 NANDs.
    # We at least verify the simulator by checking NAND truth table.
    sim = CircuitSimulator(spec)
    test_grid = np.zeros((3, 3), dtype=np.int8)
    test_grid[0, 1] = Cell.WIRE.value
    test_grid[1, 1] = Cell.NAND.value
    sim_rows = [{"A": A, "B": B, "C": sim.evaluate(test_grid, A, B)}
                for A, B in [(0, 0), (0, 1), (1, 0), (1, 1)]]

    # Figure: |S| vs grid size (log scale)
    fig, ax = plt.subplots(figsize=(7, 4))
    sizes = list(scaling.keys())
    S_vals = [scaling[s]["state_space_upper_bound"] for s in sizes]
    ax.bar(sizes, S_vals)
    ax.set_yscale("log")
    ax.set_ylabel("$|\\mathcal{S}|$ upper bound (log scale)")
    ax.set_xlabel("grid size $H \\times W$")
    ax.set_title("State space explosion: tabular VI cost is $O(|\\mathcal{S}|^2|\\mathcal{A}|)$")
    ax.grid(alpha=0.3, which="both", axis="y")
    for i, v in enumerate(S_vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p3_complexity.pdf")
    plt.close(fig)

    return {
        "scaling": scaling,
        "small_mdp_num_states": mdp.num_states,
        "small_mdp_vi_sweeps": int(sw_vi),
        "small_mdp_pi_sweeps": int(sw_pi),
        "small_mdp_V_start": float(V_vi[start_idx]),
        "small_mdp_agreement": float(np.max(np.abs(V_vi - V_pi))),
        "simulator_test_grid": sim_rows,
    }


def run_dqn() -> Dict[str, Any]:
    """V2: DQN on the Windy Chasm."""
    spec = WindyChasmSpec()
    mdp = WindyChasmMDP(spec)
    start_idx = mdp._state_index_public[mdp.start_state()]
    V_opt, _, _ = value_iteration(mdp, tol=1e-10)
    optimal = float(V_opt[start_idx])

    cfg = DQNConfig(
        hidden=(64, 64), lr=5e-4,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=20_000,
        buffer_capacity=50_000, min_buffer=1_000, batch_size=128,
        target_sync=500, seed=1,
    )
    out = train_dqn_on_mdp(mdp, cfg, episodes=3000, max_steps=60,
                           start_state_index=start_idx, verbose=False)
    returns = out["returns"]

    # Smoothed curve
    window = 100
    smooth = np.array([
        returns[max(0, i - window):i + 1].mean() for i in range(len(returns))
    ])

    fig, (axr, axl) = plt.subplots(1, 2, figsize=(10, 4))
    axr.plot(returns, alpha=0.35, label="episode return")
    axr.plot(smooth, linewidth=2, label=f"{window}-episode MA")
    axr.axhline(optimal, color="tab:red", linestyle="--", label=f"$V^*_{{\\rm VI}}$={optimal:.2f}")
    axr.set_xlabel("episode"); axr.set_ylabel("return from start")
    axr.set_title("NumPy DQN on Windy Chasm (one-hot features)")
    axr.legend(loc="lower right"); axr.grid(alpha=0.3)

    loss = out["losses"]
    if len(loss):
        axl.plot(loss, alpha=0.4, label="TD^2")
        w = 200
        loss_smooth = np.array([loss[max(0, i - w):i + 1].mean() for i in range(len(loss))])
        axl.plot(loss_smooth, linewidth=2, label=f"{w}-step MA")
    axl.set_xlabel("gradient step"); axl.set_ylabel("TD loss")
    axl.set_title("TD loss over training")
    axl.set_yscale("log"); axl.legend(); axl.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dqn_learning_curve.pdf")
    plt.close(fig)

    return {
        "num_states": mdp.num_states, "num_actions": mdp.num_actions,
        "optimal_V_start": optimal,
        "dqn_mean_last_50": float(np.mean(returns[-50:])),
        "dqn_mean_last_200": float(np.mean(returns[-200:])),
        "dqn_eval_return_start": out["eval_return_start"],
        "num_gradient_steps": int(len(loss)),
        "config": cfg.__dict__,
    }


# ---------------------------------------------------------------------------
def main() -> None:
    _ensure_dirs()
    results: Dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("[1/5] Problem 1 (base) ...")
    results["problem1"] = run_problem1()
    print("  done:", results["problem1"]["V_start"])
    print("[2/5] Problem 1 challenge ...")
    results["problem1_challenge"] = run_problem1_challenge()
    print("[3/5] Problem 2 (lidar cone) ...")
    results["problem2"] = run_problem2()
    print("  done:", results["problem2"]["V_start"])
    print("[4/5] Problem 3 (circuit design) ...")
    results["problem3"] = run_problem3()
    print("[5/5] DQN on Windy Chasm ...")
    results["dqn"] = run_dqn()
    print("  done:", results["dqn"]["dqn_mean_last_200"])

    with open(DATA_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults dumped to", DATA_DIR / "results.json")
    print("Figures written to", FIG_DIR)


if __name__ == "__main__":  # pragma: no cover
    main()
