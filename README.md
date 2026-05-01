Developed an 3-variable ODE-based model to study receptor-mediated signaling and its effect on cell proliferation. Simulated cancer vs normal behavior by varying receptor activity, integrated with growth models, and analyzed effects of signaling inhibition. Implemented in Python.


import numpy as np
from scipy.integrate import odeint
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Try to import SALib for sensitivity analysis
try:
    from SALib.sample import saltelli, sobol
    from SALib.analyze import sobol as analyze_sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    print("SALib not installed. Skipping Sobol sensitivity analysis.")
    print("Install with: pip install SALib\n")

# ============================================================
# 1. MODEL EQUATIONS
# ============================================================
def model(y, t, params):
    R, S, C = y
    k1, k2, k3, k4, k_inh = params
    dR_dt = 0                          # receptor held constant
    dS_dt = k1 * R - k2 * S - k_inh * S
    dC_dt = k3 * S - k4 * C
    return [dR_dt, dS_dt, dC_dt]


# ============================================================
# 2. SIMULATION FUNCTION
# ============================================================
def run_simulation(R_init, k_inh=0.0, label="", k1=1.0, k2=0.5, k3=1.0, k4=0.3):
    y0 = [R_init, 0.0, 0.0]
    t  = np.linspace(0, 50, 500)
    params = [k1, k2, k3, k4, k_inh]
    sol = odeint(model, y0, t, args=(params,))
    R, S, C = sol[:, 0], sol[:, 1], sol[:, 2]
    return t, R, S, C, label


# ============================================================
# 3. CELL GROWTH MODEL  (logistic — biologically realistic)
# ============================================================
def cell_growth(C, t, carrying_capacity=1e6):
    """
    Logistic growth where division rate scales with cyclin level.
    dN/dt = r(C) * N * (1 - N/K)
    Cyclin acts as a driver: higher C → faster division up to a max rate.
    """
    N = [1.0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        growth_rate = 0.7 * (C[i] / (C[i] + 5))          # Michaelis-Menten-like scaling
        dN = growth_rate * N[-1] * (1 - N[-1] / carrying_capacity) * dt
        N.append(N[-1] + dN)
    return np.array(N)


# ============================================================
# 4. BASE SCENARIOS
# ============================================================
scenarios = [
    run_simulation(1.0, 0.0, "Normal"),
    run_simulation(3.0, 0.0, "Cancer"),
    run_simulation(3.0, 0.8, "Cancer + Inhibition"),
]


# ============================================================
# 5. DRUG DOSE-RESPONSE  (sweep k_inh from 0 → 2)
# ============================================================
def dose_response(R_init=3.0, doses=np.linspace(0, 2.0, 50)):
    """
    For each inhibitor dose (k_inh), run the simulation and record
    the final cell count after 50 days.
    """
    final_cells = []
    for dose in doses:
        t, R, S, C, _ = run_simulation(R_init, k_inh=dose)
        N = cell_growth(C, t)
        final_cells.append(N[-1])
    return doses, np.array(final_cells)


# ============================================================
# 6. BIFURCATION ANALYSIS  (sweep receptor activity R)
# ============================================================
def bifurcation(R_range=np.linspace(0.1, 5.0, 80)):
    """
    Find the steady-state cyclin level (C_ss) as receptor activity varies.
    Reveals the tipping point between normal and cancer-like behaviour.
    """
    C_ss_values = []
    for R_val in R_range:
        t, _, _, C, _ = run_simulation(R_val, k_inh=0.0)
        C_ss_values.append(C[-1])      # value at t=50 ≈ steady state
    return R_range, np.array(C_ss_values)


# ============================================================
# 7. PARAMETER SENSITIVITY ANALYSIS
# ============================================================
def sensitivity_analysis():
    """
    Sobol sensitivity: which parameter (k1–k4) most controls final cyclin?
    Uses SALib if available; falls back to a manual one-at-a-time sweep.
    """
    if SALIB_AVAILABLE:
        problem = {
            'num_vars': 4,
            'names': ['k1', 'k2', 'k3', 'k4'],
            'bounds': [[0.1, 2.0], [0.1, 1.5], [0.1, 2.0], [0.05, 1.0]],
        }
        param_values = sobol.sample(problem, 256, calc_second_order=False)
        C_final = []
        for p in param_values:
            k1, k2, k3, k4 = p
            t, _, _, C, _ = run_simulation(3.0, k_inh=0.0, k1=k1, k2=k2, k3=k3, k4=k4)
            C_final.append(C[-1])
        Si = analyze_sobol.analyze(problem, np.array(C_final), calc_second_order=False)
        return problem['names'], Si['S1'], Si['ST']
    else:
        # Manual one-at-a-time sensitivity
        baseline_t, _, _, baseline_C, _ = run_simulation(3.0, k_inh=0.0)
        baseline_val = baseline_C[-1]
        param_names  = ['k1', 'k2', 'k3', 'k4']
        defaults     = {'k1': 1.0, 'k2': 0.5, 'k3': 1.0, 'k4': 0.3}
        S1 = []
        for name in param_names:
            vals, outputs = [], []
            for scale in np.linspace(0.5, 2.0, 20):
                kw = {**defaults, name: defaults[name] * scale}
                t, _, _, C, _ = run_simulation(3.0, k_inh=0.0, **kw)
                vals.append(defaults[name] * scale)
                outputs.append(C[-1])
            # Normalised sensitivity: % change in output / % change in param
            sens = np.std(outputs) / (np.mean(outputs) + 1e-9)
            S1.append(sens)
        # Normalise to sum = 1 for interpretability
        S1 = np.array(S1) / (np.sum(S1) + 1e-9)
        return param_names, S1, S1   # ST same as S1 for OAT


# ============================================================
# 8. PLOTTING  — all panels in one figure
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Membrane Signaling & Cell Proliferation Model", fontsize=16, fontweight='bold', y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

colors = ['steelblue', 'tomato', 'seagreen']

# --- Row 1: Core dynamics ---
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

for (t, R, S, C, label), col in zip(scenarios, colors):
    ax1.plot(t, S, label=label, color=col, linewidth=2)
    ax2.plot(t, C, label=label, color=col, linewidth=2)
    N = cell_growth(C, t)
    ax3.plot(t, N, label=label, color=col, linewidth=2)

ax1.set_title("Signaling Activity (S)", fontweight='bold')
ax1.set_xlabel("Time"); ax1.set_ylabel("Signal Strength"); ax1.legend(fontsize=8)

ax2.set_title("Cyclin Levels (C)", fontweight='bold')
ax2.set_xlabel("Time"); ax2.set_ylabel("Cyclin"); ax2.legend(fontsize=8)

ax3.set_title("Cell Growth — Logistic (50 days)", fontweight='bold')
ax3.set_xlabel("Time (days)"); ax3.set_ylabel("Number of Cells"); ax3.legend(fontsize=8)

# --- Row 2: Phase portrait + Bifurcation + Dose-Response ---
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

# Phase portrait: S vs C
for (t, R, S, C, label), col in zip(scenarios, colors):
    ax4.plot(S, C, label=label, color=col, linewidth=2)
    ax4.plot(S[0], C[0], 'o', color=col, markersize=6)    # mark start
ax4.set_title("Phase Portrait (S vs C)", fontweight='bold')
ax4.set_xlabel("Signal (S)"); ax4.set_ylabel("Cyclin (C)")
ax4.legend(fontsize=8); ax4.annotate("start", xy=(0, 0), fontsize=7, color='gray')

# Bifurcation
R_range, C_ss = bifurcation()
ax5.plot(R_range, C_ss, color='darkorchid', linewidth=2.5)
ax5.axvline(x=1.0, color='steelblue', linestyle='--', alpha=0.7, label='Normal (R=1)')
ax5.axvline(x=3.0, color='tomato',    linestyle='--', alpha=0.7, label='Cancer (R=3)')
ax5.set_title("Bifurcation: Receptor Activity vs Cyclin SS", fontweight='bold')
ax5.set_xlabel("Receptor Activity (R)"); ax5.set_ylabel("Steady-State Cyclin (C_ss)")
ax5.legend(fontsize=8)

# Dose-Response
doses, final_cells = dose_response()
# Normalise to fraction of untreated (dose=0) for cleaner presentation
ax6.plot(doses, final_cells / final_cells[0] * 100, color='darkorange', linewidth=2.5)
ax6.axhline(50, color='gray', linestyle=':', alpha=0.8, label='50% cell viability')
ax6.set_title("Drug Dose-Response (Cancer Cells)", fontweight='bold')
ax6.set_xlabel("Inhibitor Dose (k_inh)"); ax6.set_ylabel("Cell Count (% of untreated)")
ax6.legend(fontsize=8)

# --- Row 3: Sensitivity analysis ---
ax7 = fig.add_subplot(gs[2, :])

param_names, S1, ST = sensitivity_analysis()
x = np.arange(len(param_names))
width = 0.35
bars1 = ax7.bar(x - width/2, S1, width, label='First-Order (S1)', color='cornflowerblue', edgecolor='black')
bars2 = ax7.bar(x + width/2, ST, width, label='Total-Order (ST)', color='salmon',         edgecolor='black')
ax7.set_title("Parameter Sensitivity Analysis — Effect on Steady-State Cyclin", fontweight='bold')
ax7.set_xlabel("Parameter"); ax7.set_ylabel("Sensitivity Index")
ax7.set_xticks(x); ax7.set_xticklabels(param_names, fontsize=12)
ax7.legend()

# Label bars
for bar in bars1:
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

plt.savefig("cell_signaling_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as cell_signaling_results.png")
