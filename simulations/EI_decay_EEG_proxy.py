# ============================================================
# PyNEST – E/I imbalance model (Conductance-based LIF) - SMOOTH
# EEG/MEG proxy from synaptic currents
# Maximum smoothing for publication-quality spectra
# ============================================================
import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter

def run_simulation(condition, g_ratio, N_total=1000, frac_exc=0.8, p_conn=0.2,
                   nu_ext=3.0, sim_time=6000.0, warmup=2000.0, seed=42,
                   record_fraction=0.15, smooth_spectrum=True):
    """
    Run a simple E/I network with conductance-based neurons and synaptic current LFP proxy.
    
    Parameters
    ----------
    record_fraction : float
        Fraction of excitatory neurons to record from (default: 0.15 = 15%)
    smooth_spectrum : bool
        Apply additional Savitzky-Golay smoothing to power spectrum (default: True)
    """
    # --------------------
    # NEST kernel setup
    # --------------------
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.set_verbosity("M_WARNING")
    
    # Use the provided seed (different for each condition)
    np.random.seed(seed)
    nest.SetKernelStatus({"rng_seed": seed})
    
    # --------------------
    # Populations
    # --------------------
    N_E = int(frac_exc * N_total)
    N_I = N_total - N_E
    
    # Conductance-based LIF neurons
    E = nest.Create("iaf_cond_exp", N_E)
    I = nest.Create("iaf_cond_exp", N_I)
    
    # Conductance-based neuron parameters
    neuron_params = {
        "C_m": 250.0,          # pF - membrane capacitance
        "g_L": 16.67,          # nS - leak conductance
        "E_L": -70.0,          # mV - leak reversal potential
        "V_th": -50.0,         # mV - spike threshold
        "V_reset": -70.0,      # mV - reset potential
        "t_ref": 2.0,          # ms - refractory period
        "E_ex": 0.0,           # mV - excitatory reversal potential
        "E_in": -80.0,         # mV - inhibitory reversal potential
        "tau_syn_ex": 2.0,     # ms - excitatory synaptic time constant
        "tau_syn_in": 8.0,     # ms - inhibitory synaptic time constant
    }
    
    # Set parameters for both populations
    for pop in (E, I):
        pop.set(neuron_params)
        # Initialize membrane potentials with variability
        pop.V_m = -70.0 + 5.0 * np.random.randn(len(pop))
    
    # --------------------
    # Synaptic strengths (conductances in nS)
    # --------------------
    g_E = 2.0  # nS - excitatory synaptic conductance
    g_I = g_ratio * g_E  # nS - inhibitory synaptic conductance
    delay = 1.5
    
    print(f"\n[{condition}] N_total={N_total}, g_I/g_E = {g_ratio:.2f}, seed = {seed}")
    print(f"  g_E = {g_E:.2f} nS, g_I = {g_I:.2f} nS")
    
    # --------------------
    # External drive
    # --------------------
    ext = nest.Create("poisson_generator")
    ext.rate = nu_ext * 1000.0  # Hz
    
    nest.Connect(ext, E, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(ext, I, syn_spec={"weight": g_E, "delay": delay})
    
    # --------------------
    # Recurrent connections
    # --------------------
    conn = {"rule": "pairwise_bernoulli", "p": p_conn}
    
    nest.Connect(E, E, conn, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(E, I, conn, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(I, E, conn, syn_spec={"weight": g_I, "delay": delay})
    nest.Connect(I, I, conn, syn_spec={"weight": g_I, "delay": delay})
    
    # --------------------
    # Multimeter - increased recording for better averaging
    # --------------------
    n_rec = max(50, int(record_fraction * N_E))  # Minimum 50, or 15% of E neurons
    n_rec = min(n_rec, N_E)
    
    mm = nest.Create("multimeter")
    mm.set({
        "interval": 1.0,
        "record_from": ["g_ex", "g_in", "V_m"]
    })
    
    nest.Connect(mm, E[:n_rec])
    print(f"  Recording from {n_rec}/{N_E} excitatory neurons ({100*n_rec/N_E:.1f}%)")
    
    # --------------------
    # Spike recorder
    # --------------------
    spike_rec = nest.Create("spike_recorder")
    nest.Connect(E, spike_rec)
    
    # --------------------
    # Simulate
    # --------------------
    nest.Simulate(warmup)
    nest.Simulate(sim_time)
    
    # --------------------
    # EEG / LFP proxy
    # --------------------
    ev = mm.get("events")
    times = np.array(ev["times"])
    senders = np.array(ev["senders"])
    g_ex = np.array(ev["g_ex"])
    g_in = np.array(ev["g_in"])
    V_m = np.array(ev["V_m"])
    
    # Calculate synaptic currents from conductances
    E_ex = 0.0
    E_in = -80.0
    
    I_ex = g_ex * (V_m - E_ex)
    I_in = g_in * (V_m - E_in)
    
    # Filter out warmup period
    mask = times > warmup
    times_filtered = times[mask]
    senders_filtered = senders[mask]
    I_ex_filtered = I_ex[mask]
    I_in_filtered = I_in[mask]
    V_m_filtered = V_m[mask]
    
    # Get unique time points and neurons
    unique_times = np.sort(np.unique(times_filtered))
    unique_neurons = np.sort(np.unique(senders_filtered))
    n_times = len(unique_times)
    n_neurons = len(unique_neurons)
    
    print(f"  Recording from {n_neurons} neurons over {n_times} time points")
    
    # Handle data reshaping
    expected_length = n_times * n_neurons
    if len(times_filtered) != expected_length:
        print(f"  WARNING: Expected {expected_length} points, got {len(times_filtered)}")
        I_ex_matrix = np.zeros((n_times, n_neurons))
        I_in_matrix = np.zeros((n_times, n_neurons))
        V_m_matrix = np.zeros((n_times, n_neurons))
        
        neuron_to_idx = {nid: idx for idx, nid in enumerate(unique_neurons)}
        time_to_idx = {t: idx for idx, t in enumerate(unique_times)}
        
        for i in range(len(times_filtered)):
            t_idx = time_to_idx[times_filtered[i]]
            n_idx = neuron_to_idx[senders_filtered[i]]
            I_ex_matrix[t_idx, n_idx] = I_ex_filtered[i]
            I_in_matrix[t_idx, n_idx] = I_in_filtered[i]
            V_m_matrix[t_idx, n_idx] = V_m_filtered[i]
    else:
        I_ex_matrix = I_ex_filtered.reshape(n_times, n_neurons)
        I_in_matrix = I_in_filtered.reshape(n_times, n_neurons)
        V_m_matrix = V_m_filtered.reshape(n_times, n_neurons)
    
    # LFP proxy: average synaptic currents across neurons
    lfp = I_ex_matrix.mean(axis=1) - I_in_matrix.mean(axis=1)
    lfp -= lfp.mean()
    
    print(f"  LFP: {len(lfp)} samples, range [{lfp.min():.2f}, {lfp.max():.2f}] pA")
    print(f"  Mean V_m: {V_m_matrix.mean():.2f} mV, std: {V_m_matrix.std():.2f} mV")
    
    # Get spike statistics
    spike_events = spike_rec.get("events")
    spike_times = spike_events["times"]
    spike_times = spike_times[spike_times > warmup]
    firing_rate = len(spike_times) / (sim_time / 1000.0) / N_E
    print(f"  Mean firing rate (E): {firing_rate:.2f} Hz")
    
    # --------------------
    # Power spectrum with MAXIMUM smoothing
    # --------------------
    fs = 1000.0
    
    # Use very large window and maximum overlap for smoothest spectra
    # For N=1000, we have ~6000ms of data = 6000 samples
    # Use half of the data as window size for maximum smoothing
    nperseg = min(len(lfp) // 2, 16384)  # Very large window
    nperseg = max(nperseg, 4096)          # But at least 4096
    
    # Maximum overlap (93.75% = 15/16)
    noverlap = int(15 * nperseg // 16)
    
    f, Pxx = welch(lfp, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                   window='hann', detrend='constant')
    
    # Filter to frequency band of interest
    band = (f >= 1) & (f <= 40)
    f = f[band]
    Pxx = Pxx[band]
    
    # Apply additional Savitzky-Golay smoothing if requested
    if smooth_spectrum:
        # Window must be odd and less than data length
        window_length = min(21, len(Pxx) // 2 * 2 - 1)  # Make it odd
        if window_length >= 5:  # Need at least 5 points
            Pxx = savgol_filter(Pxx, window_length=window_length, polyorder=3)
            print(f"  Applied Savitzky-Golay filter (window={window_length})")
    
    # Normalize to relative power
    Pxx = np.maximum(Pxx, 0)  # Ensure non-negative after filtering
    Pxx /= Pxx.sum()
    
    print(f"  Welch: nperseg={nperseg}, noverlap={noverlap}, freq_res={fs/nperseg:.3f} Hz")
    
    return {
        "condition": condition,
        "g_ratio": g_ratio,
        "f": f,
        "Pxx": Pxx,
        "lfp": lfp,
        "firing_rate": firing_rate,
        "n_rec": n_rec,
        "N_E": N_E
    }


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    # Network size to test
    N_TOTAL = 400
    
    conditions = [
        ("AD", 2.5, 42),
        ("MCI", 3.5, 123),
        ("HC", 6.5, 456),
    ]
    
    print("=" * 60)
    print(f"Running E/I Balance Simulations (Maximum Smoothing)")
    print(f"Network size: {N_TOTAL} neurons")
    print("=" * 60)
    
    results = []
    for name, g, seed in conditions:
        res = run_simulation(name, g, N_total=N_TOTAL, seed=seed, 
                           record_fraction=0.15, smooth_spectrum=True)
        if res is not None:
            results.append(res)
    
    # --------------------
    # Verification
    # --------------------
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for r in results:
        print(f"{r['condition']:3s}: {r['n_rec']}/{r['N_E']} neurons, "
              f"FR = {r['firing_rate']:.2f} Hz")
    
    # --------------------
    # Plot
    # --------------------
    if results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = {
            "AD": "#90EE90",
            "MCI": "#FFD700",
            "HC": "#A9A9A9",
        }
        
        for r in results:
            ax.plot(r["f"], r["Pxx"], 
                    label=f"{r['condition']} ", 
                    linewidth=2.5, 
                    color=colors.get(r["condition"], 'gray'))
        
        # Frequency band boundaries
        band_boundaries = [4, 8, 13, 30]
        for boundary in band_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', 
                      linewidth=1.5, alpha=0.6)
        
        # Band labels
        y_max = ax.get_ylim()[1]
        band_centers = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
        band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
        for center, name in zip(band_centers, band_names):
            if center <= 40:
                ax.text(center, y_max * 0.95, name,
                       horizontalalignment='center',
                       fontsize=10, style='italic',
                       color='gray', alpha=0.7)
        
        ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Relative power", fontsize=14, fontweight='bold')
        ax.set_xlim(1, 40)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11,loc='upper right')

#        ax.set_title(f"E/I Balance Effects on EEG Power Spectrum (N={N_TOTAL}, Smoothed)",                    fontsize=13)
        
        plt.tight_layout()
        plt.savefig(f"EI_EEG_proxy_conductance_N{N_TOTAL}_smooth.png", dpi=300)
        print(f"\n✓ Plot saved as EI_EEG_proxy_conductance_N{N_TOTAL}_smooth.png")
        plt.show()
    else:
        print("\n✗ No successful simulations to plot!")
