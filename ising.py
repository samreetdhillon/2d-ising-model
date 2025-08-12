# ising_metropolis.py
# 2D Ising model Monte Carlo (Metropolis)
# Save snapshots & produce plots for observables vs. T and autocorrelation.

import numpy as np
import matplotlib.pyplot as plt
import time
import os

np.random.seed(42)

# ---- Parameters (tweak these as needed) ----
L = 32                      # lattice size LxL
temps = np.linspace(1.5, 3.5, 16)  # temperature grid around Tc ~ 2.269
n_eq = 1000                 # equilibration sweeps per T (one sweep = L*L attempted flips)
n_meas = 2000               # measurement sweeps per T
measure_interval = 1        # record every 'measure_interval' sweeps
snapshot_temps = [1.5, 2.27, 3.0]  # temps to save spin-lattice snapshots
save_dir = "ising_outputs"
os.makedirs(save_dir, exist_ok=True)

# ---- Helper functions ----
def init_lattice(L, mode="random"):
    if mode == "up":
        return np.ones((L, L), dtype=int)
    elif mode == "down":
        return -np.ones((L, L), dtype=int)
    else:
        return np.random.choice([-1, 1], size=(L, L))

def energy_of_lattice(spins):
    # periodic BCs; count each nearest-neighbor pair once (right+down)
    E = 0
    E -= np.sum(spins * np.roll(spins, 1, axis=0))
    E -= np.sum(spins * np.roll(spins, 1, axis=1))
    return E

def magnetization(spins):
    return np.sum(spins)

def metropolis_sweep(spins, beta):
    L = spins.shape[0]
    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]
        nb = (spins[(i+1)%L, j] + spins[(i-1)%L, j] +
              spins[i, (j+1)%L] + spins[i, (j-1)%L])
        dE = 2 * s * nb  # ΔE with J=1
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] = -s
    return spins

def autocorrelation(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    var = x.var()
    if var == 0:
        return np.zeros(n)
    corr = np.correlate(x, x, mode='full')[n-1:]  # from lag 0..n-1
    # normalize for decreasing sample count at large lags
    norm = var * np.arange(n, 0, -1)
    corr = corr / norm
    return corr / corr[0]

# ---- Run temperature scan ----
results = {
    "T": [],
    "E_mean_per_spin": [],
    "E_var_per_spin": [],
    "M_mean_abs_per_spin": [],
    "M_var_per_spin": [],
    "mag_time_series_example": None,
}
start_time = time.time()

for T in temps:
    beta = 1.0 / T
    spins = init_lattice(L, mode="random")
    # Equilibrate
    for _ in range(n_eq):
        metropolis_sweep(spins, beta)

    # Measurement
    E_samples = []
    M_samples = []
    for sweep in range(n_meas):
        metropolis_sweep(spins, beta)
        if sweep % measure_interval == 0:
            E_samples.append(energy_of_lattice(spins))
            M_samples.append(magnetization(spins))

    E_samples = np.array(E_samples)
    M_samples = np.array(M_samples)
    E_mean = E_samples.mean() / (L*L)
    E_var = E_samples.var() / (L*L)
    M_mean_abs = np.mean(np.abs(M_samples)) / (L*L)
    M_var = np.var(M_samples) / (L*L)

    results["T"].append(T)
    results["E_mean_per_spin"].append(E_mean)
    results["E_var_per_spin"].append(E_var)
    results["M_mean_abs_per_spin"].append(M_mean_abs)
    results["M_var_per_spin"].append(M_var)

    # keep an M(t) series for a T near Tc for autocorr
    if abs(T - 2.27) < 0.1 and results["mag_time_series_example"] is None:
        results["mag_time_series_example"] = {"T": T, "M_samples": M_samples.copy()}

    # save snapshot if requested
    for st in snapshot_temps:
        if abs(T - st) < 1e-8:
            plt.figure(figsize=(4,4))
            plt.imshow(spins, interpolation='nearest')
            plt.title(f"L={L}, T={T:.2f}")
            plt.axis('off')
            fname = os.path.join(save_dir, f"snapshot_L{L}_T{T:.2f}.png")
            plt.savefig(fname, bbox_inches='tight', dpi=150)
            plt.close()

    elapsed = time.time() - start_time
    print(f"T={T:.3f}  <e>={E_mean:.4f}  <|m|>={M_mean_abs:.4f}  elapsed {elapsed:.1f}s")

print("Scan complete.")

# ---- Derived quantities ----
T_arr = np.array(results["T"])
E_mean_arr = np.array(results["E_mean_per_spin"]) * (L*L)  # total energy
E_var_total = np.array(results["E_var_per_spin"]) * (L*L) # total variance
M_mean_abs_total = np.array(results["M_mean_abs_per_spin"]) * (L*L)
M_var_total = np.array(results["M_var_per_spin"]) * (L*L)

C_per_spin = E_var_total / (T_arr**2) / (L*L) # heat capacity per spin
chi_per_spin = M_var_total / T_arr / (L*L)    # susceptibility per spin

# ---- Plots (one plot per figure) ----
plt.figure()
plt.plot(T_arr, M_mean_abs_total/(L*L), marker='o')
plt.xlabel("T")
plt.ylabel("⟨|m|⟩ per spin")
plt.title(f"Magnetization vs T (L={L})")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "magnetization_vs_T.png"), dpi=150)
plt.show()

plt.figure()
plt.plot(T_arr, E_mean_arr/(L*L), marker='o')
plt.xlabel("T")
plt.ylabel("⟨e⟩ (energy per spin)")
plt.title(f"Energy per spin vs T (L={L})")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "energy_vs_T.png"), dpi=150)
plt.show()

plt.figure()
plt.plot(T_arr, C_per_spin, marker='o')
plt.xlabel("T")
plt.ylabel("Heat capacity per spin C")
plt.title(f"Heat capacity vs T (L={L})")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "heat_capacity_vs_T.png"), dpi=150)
plt.show()

plt.figure()
plt.plot(T_arr, chi_per_spin, marker='o')
plt.xlabel("T")
plt.ylabel("Susceptibility per spin χ")
plt.title(f"Susceptibility vs T (L={L})")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "susceptibility_vs_T.png"), dpi=150)
plt.show()

# ---- Autocorrelation for example T near Tc ----
if results["mag_time_series_example"] is not None:
    M_ts = results["mag_time_series_example"]["M_samples"]
    ac = autocorrelation(M_ts)
    lags = np.arange(len(ac))
    plt.figure()
    plt.plot(lags, ac, marker='o')
    plt.xlabel("Lag (sweeps)")
    plt.ylabel("Autocorrelation of M(t)")
    plt.title(f"Autocorr of M at T={results['mag_time_series_example']['T']:.2f}")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "autocorr_M_TnearTc.png"), dpi=150)
    plt.show()

print("Saved plots and snapshots to", save_dir)
