"""
Generate Nonconvex splitting -- 3x2: avg/min metrics, openloop vs
adaptive schedules
"""

import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(THIS_DIR / ".mplconfig"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)
from frank_wolfe import Frames
from matplotlib.lines import Line2D
from tqdm import tqdm

legendfontsize = 12
titlefontsize = 20
labelfontsize = 18
suptitlefontsize = 24


def estimate_lipschitz(obj, lmo, dim, n_samples=500, seed=99):
    rng = np.random.default_rng(seed)
    pool = np.array([lmo(rng.standard_normal(dim)) for _ in range(20)])
    L_est = 0.0
    for _ in range(n_samples):
        w1, w2 = rng.dirichlet(np.ones(len(pool))), rng.dirichlet(np.ones(len(pool)))
        z1, z2 = w1 @ pool, w2 @ pool
        diff = np.linalg.norm(z1 - z2)
        if diff < 1e-12:
            continue
        L_est = max(L_est, np.linalg.norm(obj.gradient(z1) - obj.gradient(z2)) / diff)
    print("L_est is " + str(L_est))
    return L_est


def figure3():
    from paper.experiments.l1_splitting_nonconvex import (
        IndefiniteQuadraticSplitting,
        create_product_l1_lmo,
        generate_indefinite_quadratic,
        nonsmooth_gap,
        run_frames_splitting_ol,
        run_frames_splitting_ada,
        zero_prox,
    )

    nn = 10
    n_steps = 10000
    seed = 42
    c1 = np.zeros(nn)
    c1[0] = 1.0
    c2 = np.zeros(nn)
    c2[0] = -1.0
    radius = 2.0
    Q, b, eigvals = generate_indefinite_quadratic(nn, frac_neg=0.3, seed=seed)
    Q_op = np.linalg.norm(Q, ord=2)
    L_product = 8.0 * 0.5 * Q_op
    inv_L = 1.0 / L_product
    n_pos = np.sum(eigvals > 0)
    n_neg = np.sum(eigvals < 0)
    print(f"Splitting: n={nn}, {n_pos} pos / {n_neg} neg eigs")
    print(f"||Q||_op={Q_op:.2f}, L_product=||Q||/2={L_product:.2f}, 1/L={inv_L:.4f}")
    obj = IndefiniteQuadraticSplitting(Q, b, nn)
    lmo = create_product_l1_lmo(c1, c2, radius, nn)
    beta0_values = [inv_L]

    # Power schedule
    results = {}
    p_ol = 2/3
    q_ol = 1/3
    for beta0 in beta0_values:
        (_, fvals, sgaps, nsgaps, feas, l1_dist, lam_ol) = run_frames_splitting_ol(
            obj, lmo, n_steps, beta0=beta0, p=p_ol, q=q_ol
        )
        results[beta0] = dict(
            func_vals=fvals, 
            smoothed_gaps=sgaps,
            ns_gaps=nsgaps,
            feasibility=feas,
            l1_ball_distance=l1_dist,
            lam=lam_ol
        )
        lab_str = f"1/L={inv_L:.4f}" if beta0 == inv_L else f"{beta0}"
        print(
            f"  [openloop] beta0={lab_str:>12s}  f={fvals[-1]:.4f}  ||x1-x2||={np.sqrt(feas[-1]):.4e}  ns_gap={nsgaps[-1]:.4e}"
        )

#old:
#    p_ada1 = 0.5
#    q_ada1 = 0.5
#    def ada_func1(beta, beta0, it, step_size, feas):
#        xi = (it + 2) ** (-q_ada1)
#        lam = 1/beta
#        if feas <= xi:
#            #If approximately feasible, leave the smoothing parameter
#            #unchanged
#            return beta
#        else:
#            return 1/(lam + (beta0*step_size*step_size/feas))
#

    p_ada1 = 2/3
    q_ada1 = 1
    theta_ada1 = 9/27
    def ada_func1(beta, beta0, it, step_size, feas, sgap):
        xi = (it + 2) ** (-q_ada1)
        lam = 1/beta
        if feas <= sgap :
            #If approximately feasible, leave the smoothing parameter
            #unchanged
            return beta
        else:
#            return 1/((1/beta0)*((1+(0.5*it)) ** theta_ada1))
            return 1/((1/beta0)*((1+(0.5*it)) ** theta_ada1))


        
    p_ada2 = 2/3
    #q \geq 2/3
    q_ada2 = 1.
    #zeta_ada2 = 1.0
    theta_ada2 = 9/27
    def ada_func2(beta, beta0, it, step_size, feas, sgap):
        q = q_ada2
        #q \geq 0.5
        xi = (it + 1) ** (-q_ada2)
        lam = 1/beta
        #zeta = 1
        if feas <= sgap :
            #If approximately feasible, leave the smoothing parameter
            #unchanged
            return beta
        else:
            zeta = min(1.0, 1/feas)
            incr = (1/beta0)*(((1+(0.5*it)) ** theta_ada2) - ((1+(0.5*(it-1))) ** theta_ada2))
            return 1/(lam + zeta*(incr))
    
    # Adaptive schedule
    results_ada = {}
    for f_label in [1, 2, 3]:
        if f_label == 1:
            ada_func = ada_func1
            p_val = p_ada1
        elif f_label == 2:
            ada_func = ada_func2
            p_val = p_ada2
        elif f_label == 3:
            ada_func = ada_func2
            p_val = "short"


        _, fvals_ada, sgaps_ada, nsgaps_ada, feas_ada, l1_dist_ada, lam_ada = run_frames_splitting_ada( obj, lmo, n_steps, ada_func=ada_func, beta0=beta0_values[0], p = p_val)
        results_ada[f_label] = dict(
            func_vals=fvals_ada,
            smoothed_gaps=sgaps_ada,
            ns_gaps=nsgaps_ada,
            feasibility=feas_ada,
            l1_ball_distance=l1_dist_ada,
            lam=lam_ada
        )
        print(
            f"  [ada{f_label}]   beta0={beta0_values[0]}  f={fvals_ada[-1]:.4f}  ||x1-x2||={np.sqrt(feas_ada[-1]):.4e}  ns_gap={nsgaps_ada[-1]:.4e}"
        )

    iters = np.arange(1, n_steps + 1)
    fig, axs = plt.subplots(4, 2, figsize=(12, 20))
    method_specs = [
            (f"Normal p={p_ol:.2f}, q={q_ol:.2f}", results[beta0_values[0]], "-", "C0"),
            (f"Adaptive 1 p={p_ada1:.2f}, q={q_ada1:.2f}", results_ada[1], "--", "C1"),
            (f"Adaptive 2 p={p_ada2:.2f}, q={q_ada2:.2f}, "+r"$\theta$="+f"{theta_ada2:.2f}", results_ada[2], ":", "C2"),
            (f"Adaptive 2 (short), q={q_ada2:.2f}, "+r"$\theta$="+f"{theta_ada2:.2f}",
                results_ada[3],
                "-.",
                "C3",
            ),
    ]
    plot_stride = 10
    plot_idx = np.arange(plot_stride - 1, n_steps, plot_stride)
    plot_iters = iters[plot_idx]

    metric_specs = [
        (
            "Smoothed gaps",
            r"$\frac{1}{k}\sum\limits_{j=0}^{k-1}\mathrm{gap}^{\beta_j}(\mathbf{x}_j)$",
            r"$\min\limits_{0\leq j\leq k-1}\mathrm{gap}^{\beta_j}(\mathbf{x}_j)$",
            lambda R: R["smoothed_gaps"],
        ),
        (
            "Feasibility",
            r"$\frac{1}{k}\sum\limits_{j=0}^{k-1}\mathrm{dist}_{\mathcal{D}}(T\mathbf{x}_j)$",
            r"$\mathrm{dist}_{\mathcal{D}}(T\mathbf{x}_j)$",
            lambda R: np.sqrt(np.maximum(R["feasibility"], 1e-16)),
        ),
        (
            r"Distance to the unit $\ell_1$-ball",
            r"$\lambda_k$",
            r"$\mathrm{dist}_2(\bar{\mathbf{x}}_j,B_1)$",
            lambda R: np.maximum(R["l1_ball_distance"], 1e-16),
        ),
        (
            "Nonsmooth gaps",
            r"$\frac{1}{k}\sum\limits_{j=0}^{k-1}|\widetilde{\mathrm{gap}}(\mathbf{x}_j)|$",
            r"$\min\limits_{0\leq j\leq k-1}|\widetilde{\mathrm{gap}}(\mathbf{x}_j)|$",
            lambda R: np.abs(R["ns_gaps"]),
        ),
    ]

    for method_label, R, linestyle, color in method_specs:
        for row, (title, _, _, metric_fn) in enumerate(metric_specs):
            y = metric_fn(R)

            if title == r"Distance to the unit $\ell_1$-ball":
                left_y = np.asarray(R["lam"])
            else:
                method_iters = np.arange(1, len(y) + 1)
                left_y = np.cumsum(y) / method_iters
            
            if title in (r"Distance to the unit $\ell_1$-ball", "Feasibility"):
                right_y = y
            else:
                right_y = np.minimum.accumulate(y)            
    
            # Restrict the shared plotting indices to this array's length.
            method_plot_idx = plot_idx[plot_idx < len(y)]
            method_plot_iters = method_iters[method_plot_idx]
    
            axs[row, 0].semilogy(
                method_plot_iters,
                left_y[method_plot_idx],
                color=color,
                linestyle=linestyle,
                label=method_label,
                linewidth=2.0,
            )
    
            axs[row, 1].semilogy(
                method_plot_iters,
                right_y[method_plot_idx],
                color=color,
                linestyle=linestyle,
                label=method_label,
                linewidth=2.0,
            )
        
   
    for row, (title, avg_ylabel, min_ylabel, _) in enumerate(metric_specs):
        if title == r"Distance to the unit $\ell_1$-ball":
            axs[row, 0].set_title(
                r"Smoothing parameter $\lambda_k$",
                fontsize=titlefontsize,
            )
        else:
            axs[row, 0].set_title(
                f"Average {title.lower()}",
                fontsize=titlefontsize,
            )
        if ((title == r"Distance to the unit $\ell_1$-ball" ) |(
            title == "Feasibility")):
            axs[row, 1].set_title(f"{title.lower()}", fontsize=titlefontsize)
        else:
            axs[row, 1].set_title(f"Minimum {title.lower()}", fontsize=titlefontsize)
        axs[row, 0].set_ylabel(avg_ylabel, fontsize=labelfontsize)
        axs[row, 1].set_ylabel(min_ylabel, fontsize=labelfontsize)

        for col in range(2):
            axs[row, col].set_xlabel("Iteration $k$", fontsize=labelfontsize)
            axs[row, col].grid(True, alpha=0.3)
            handles, _ = axs[row, col].get_legend_handles_labels()
            axs[row, col].legend(
                handles=handles,
                fontsize=8,
                loc="lower left",
                ncol=1,
            )
#
#    axs[0, 0].set_ylim(6e-2, 2e0)
#    axs[0, 1].set_ylim(8e-3, 6e-1)
#    axs[1, 0].set_ylim(6e-2, 9e-1)
#    axs[1, 1].set_ylim(1e-2, 9e-1)
#    axs[2, 0].set_ylim(8e-2, 1.5e0)
    axs[2, 1].set_ylim(5e-10, 5e-1)
#
    fig.suptitle(
        f"Nonconvex splitting,  $n={nn},\\; {n_neg}$ neg.\\ eigs",
        fontsize=suptitlefontsize,
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "paper_fig3_splitting.pdf"), dpi=150
    )
    plt.close(fig)
    print("Saved paper_fig3_splitting.pdf")


def figure2():
    figure3()


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 3: Nonconvex splitting")
    print("=" * 60)
    figure3()
