# ============================================================
# PyNEST – E/I imbalance model with Cohen's d Analysis
# EEG/MEG proxy from synaptic currents
# Multiple runs with different random network connectivity
# ============================================================
import nest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter
from fooof import FOOOF


# ============================================================
# Cohen's d Functions
# ============================================================

def compute_cohens_d(group1_values, group2_values):
    """Compute Cohen's d with pooled standard deviation."""
    g1 = np.array(group1_values)
    g2 = np.array(group2_values)
    
    n1, n2 = len(g1), len(g2)
    mean1, mean2 = np.mean(g1), np.mean(g2)
    std1 = np.std(g1, ddof=1)
    std2 = np.std(g2, ddof=1)
    
    s_pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / s_pooled
    
    return {
        'cohens_d': cohens_d,
        'mean_diff': mean1 - mean2,
        's_pooled': s_pooled,
        'mean_AD': mean1,
        'mean_HC': mean2,
        'std_AD': std1,
        'std_HC': std2,
        'n_AD': n1,
        'n_HC': n2
    }


# ============================================================
# NEST Simulation with Full Randomization Control
# ============================================================

def run_simulation(condition, g_ratio, N_total=1000, frac_exc=0.8, p_conn=0.2,
                   nu_ext=3.0, sim_time=6000.0, warmup=2000.0, seed=42,
                   record_fraction=0.15, smooth_spectrum=True):
    """
    Run a simple E/I network with conductance-based neurons and synaptic current LFP proxy.
    
    Parameters
    ----------
    condition : str
        Condition name (e.g., 'AD', 'MCI', 'HC')
    g_ratio : float
        Ratio of inhibitory to excitatory conductance
    seed : int
        Random seed for reproducibility
    record_fraction : float
        Fraction of excitatory neurons to record from (default: 0.15 = 15%)
    smooth_spectrum : bool
        Apply additional Savitzky-Golay smoothing to power spectrum (default: True)
    """
    # --------------------
    # NEST kernel setup with full randomization control
    # --------------------
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.set_verbosity("M_WARNING")
    
    # Set NEST RNG seeds (controls stochastic connections!)
    nest.rng_seed = seed
    nest.total_num_virtual_procs = 1  # Use single thread for reproducibility
    
    # Set NumPy RNG seed (controls parameter generation)
    np.random.seed(seed)
    
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
    
    # Set parameters for both populations with heterogeneity
    for pop in (E, I):
        pop.set(neuron_params)
        # Initialize membrane potentials with variability (varies with seed)
        pop.V_m = -70.0 + 5.0 * np.random.randn(len(pop))
        # Add threshold variability (varies with seed)
        pop.V_th = -50.0 + 2.0 * np.random.randn(len(pop))
    
    # --------------------
    # Synaptic strengths (conductances in nS)
    # --------------------
    g_E = 2.0  # nS - excitatory synaptic conductance
    g_I = g_ratio * g_E  # nS - inhibitory synaptic conductance
    delay = 1.5
    
    # --------------------
    # External drive with heterogeneity (varies with seed)
    # --------------------
    ext_drives = []
    for _ in range(N_total):
        pg = nest.Create("poisson_generator")
        rate = nu_ext * 1000.0 * np.random.lognormal(mean=0.0, sigma=0.2)
        pg.rate = rate
        ext_drives.append(pg)
    
    for i, pg in enumerate(ext_drives):
        if i < N_E:
            nest.Connect(pg, E[i:i+1], syn_spec={"weight": g_E, "delay": delay})
        else:
            nest.Connect(pg, I[i-N_E:i-N_E+1], syn_spec={"weight": g_E, "delay": delay})
    
    # Background noise
    noise = nest.Create("poisson_generator")
    noise.rate = 200.0
    nest.Connect(noise, E, syn_spec={"weight": 0.3 * g_E, "delay": delay})
    nest.Connect(noise, I, syn_spec={"weight": 0.3 * g_E, "delay": delay})
    
    # --------------------
    # Recurrent connections (NEST RNG - varies with seed!)
    # --------------------
    conn = {"rule": "pairwise_bernoulli", "p": p_conn}
    
    nest.Connect(E, E, conn, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(E, I, conn, syn_spec={"weight": 1.2 * g_E, "delay": delay})
    nest.Connect(I, E, conn, syn_spec={"weight": -g_I, "delay": delay})
    nest.Connect(I, I, conn, syn_spec={"weight": -0.8 * g_I, "delay": delay})
    
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
    
    # Handle data reshaping
    expected_length = n_times * n_neurons
    if len(times_filtered) != expected_length:
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
    
    # Get spike statistics
    spike_events = spike_rec.get("events")
    spike_times = spike_events["times"]
    spike_times = spike_times[spike_times > warmup]
    firing_rate = len(spike_times) / (sim_time / 1000.0) / N_E
    
    # --------------------
    # Power spectrum with MAXIMUM smoothing
    # --------------------
    fs = 1000.0
    
    # Use very large window and maximum overlap for smoothest spectra
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
        window_length = min(21, len(Pxx) // 2 * 2 - 1)  # Make it odd
        if window_length >= 5:
            Pxx = savgol_filter(Pxx, window_length=window_length, polyorder=3)
    
    # Ensure non-negative after filtering
    Pxx = np.maximum(Pxx, 0)
    
    # --------------------
    # Band powers (before normalization for absolute values)
    # --------------------
    def band_power(f, Pxx, fmin, fmax):
        idx = (f >= fmin) & (f < fmax)
        return np.mean(Pxx[idx]) if np.any(idx) else 0.0
    
    theta_power = band_power(f, Pxx, 4, 8)
    alpha_power = band_power(f, Pxx, 8, 13)
    beta_power = band_power(f, Pxx, 13, 30)
    gamma_power = band_power(f, Pxx, 30, 40)
    
    # --------------------
    # Aperiodic fitting with FOOOF
    # --------------------
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=5, min_peak_height=0.1)
    try:
        fm.fit(f, Pxx, freq_range=[1, 40])
        aperiodic_params = fm.get_params('aperiodic_params')
        offset = aperiodic_params[0]
        exponent = aperiodic_params[1]
    except:
        offset = np.nan
        exponent = np.nan
    
    # Normalize to relative power for plotting
    Pxx_norm = Pxx / Pxx.sum()
    
    return {
        "condition": condition,
        "g_ratio": g_ratio,
        "seed": seed,
        "f": f,
        "Pxx": Pxx_norm,
        "lfp": lfp,
        "firing_rate": firing_rate,
        "n_rec": n_rec,
        "N_E": N_E,
        # Metrics for Cohen's d analysis
        "aperiodic_exponent": exponent,
        "aperiodic_offset": offset,
        "theta_power": theta_power,
        "alpha_power": alpha_power,
        "beta_power": beta_power,
        "gamma_power": gamma_power,
        "spike_rate": firing_rate
    }


# ============================================================
# Multiple Simulations with Different Seeds
# ============================================================

def run_multiple_simulations(n_runs=10, N_total=1000):
    """
    Run multiple simulations with different random seeds.
    
    Each run will have:
    - Different random connectivity patterns
    - Different initial conditions
    - Different external input patterns
    """
    
    print("="*70)
    print("RUNNING MULTIPLE SIMULATIONS WITH DIFFERENT RANDOM NETWORKS")
    print("="*70)
    print(f"Network size: {N_total} neurons")
    print(f"Number of runs per condition: {n_runs}")
    print(f"Each run uses a different random seed to generate:")
    print("  • Random network connectivity (Bernoulli connections)")
    print("  • Random initial conditions (V_m, V_th)")
    print("  • Random external drive rates")
    print("="*70)
    
    ad_results = []
    hc_results = []
    
    for run in range(n_runs):
        # Use widely spaced seeds to ensure independence
        seed = 1000 + run * 1000
        
        print(f"\n{'='*70}")
        print(f"Run {run+1}/{n_runs} (seed={seed})")
        print(f"{'='*70}")
        
        # AD simulation
        print(f"  [AD] Running with g_ratio=2.5, seed={seed}...")
        res_ad = run_simulation("AD", 2.5, N_total=N_total, seed=seed)
        if res_ad:
            ad_results.append(res_ad)
            print(f"       ✓ Exponent: {res_ad['aperiodic_exponent']:.3f}, "
                  f"Spike rate: {res_ad['spike_rate']:.2f} Hz")
        else:
            print(f"       ✗ Failed")
        
        # HC simulation (different seed for truly independent network)
        seed_hc = seed + 100  # Offset to ensure different connectivity
        print(f"  [HC] Running with g_ratio=6.5, seed={seed_hc}...")
        res_hc = run_simulation("HC", 6.5, N_total=N_total, seed=seed_hc)
        if res_hc:
            hc_results.append(res_hc)
            print(f"       ✓ Exponent: {res_hc['aperiodic_exponent']:.3f}, "
                  f"Spike rate: {res_hc['spike_rate']:.2f} Hz")
        else:
            print(f"       ✗ Failed")
    
    print(f"\n{'='*70}")
    print(f"✓ Completed {len(ad_results)} AD and {len(hc_results)} HC simulations")
    print(f"{'='*70}")
    
    return ad_results, hc_results


def extract_measures(results_list):
    """Extract all measures into dictionary format."""
    measures = {}
    
    for key in ['aperiodic_exponent', 'aperiodic_offset', 'theta_power', 
                'alpha_power', 'beta_power', 'gamma_power', 'spike_rate']:
        measures[key] = [r[key] for r in results_list if not np.isnan(r[key])]
    
    return measures


def compute_all_cohens_d(ad_measures, hc_measures):
    """Compute Cohen's d for all measures."""
    results = []
    
    for measure in ad_measures.keys():
        if measure in hc_measures:
            result = compute_cohens_d(ad_measures[measure], hc_measures[measure])
            result['measure'] = measure
            results.append(result)
    
    df = pd.DataFrame(results)
    return df


def plot_cohens_d_comparison(sim_effects, emp_effects, output_prefix='cohens_d'):
    """Create comprehensive comparison plots."""
    
    comparison = pd.merge(
        sim_effects[['measure', 'cohens_d']], 
        emp_effects[['measure', 'cohens_d']], 
        on='measure', 
        suffixes=('_sim', '_emp')
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Bar comparison
    x_pos = np.arange(len(comparison))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, comparison['cohens_d_sim'], width, 
                label='Simulation', color='#4A90E2', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].bar(x_pos + width/2, comparison['cohens_d_emp'], width, 
                label='Empirical', color='#E27D60', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel("Cohen's d (AD - HC)", fontsize=13, fontweight='bold')
    axes[0].set_title('Effect Sizes: Simulation vs Empirical', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(comparison['measure'], rotation=45, ha='right')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Plot 2: Correlation
    axes[1].scatter(comparison['cohens_d_emp'], comparison['cohens_d_sim'], 
                   s=150, color='#9B59B6', alpha=0.7, edgecolor='black', linewidth=2)
    
    all_d = list(comparison['cohens_d_emp']) + list(comparison['cohens_d_sim'])
    lims = [min(all_d) - 0.3, max(all_d) + 0.3]
    axes[1].plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect match')
    
    if len(comparison) > 1:
        corr = np.corrcoef(comparison['cohens_d_emp'], comparison['cohens_d_sim'])[0, 1]
        axes[1].text(0.05, 0.95, f'r = {corr:.3f}', 
                    transform=axes[1].transAxes, fontsize=13, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    for idx, row in comparison.iterrows():
        axes[1].annotate(row['measure'], 
                        (row['cohens_d_emp'], row['cohens_d_sim']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
    
    axes[1].set_xlabel("Empirical Cohen's d", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Simulation Cohen's d", fontsize=13, fontweight='bold')
    axes[1].set_title('Correlation Analysis', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Plot 3: Differences
    comparison['difference'] = comparison['cohens_d_sim'] - comparison['cohens_d_emp']
    colors = ['#2ECC71' if d > 0 else '#E74C3C' for d in comparison['difference']]
    
    axes[2].barh(x_pos, comparison['difference'], color=colors, 
                alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[2].axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    axes[2].set_yticks(x_pos)
    axes[2].set_yticklabels(comparison['measure'])
    axes[2].set_xlabel("Δ Cohen's d (Sim - Emp)", fontsize=13, fontweight='bold')
    axes[2].set_title('Model Bias', fontsize=13, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved {output_prefix}_comparison.png/pdf")


def plot_variability_across_runs(ad_results, hc_results):
    """Visualize variability across different random network realizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    measures = ['aperiodic_exponent', 'theta_power', 'alpha_power', 
                'beta_power', 'gamma_power', 'spike_rate']
    
    for idx, measure in enumerate(measures):
        ad_vals = [r[measure] for r in ad_results if not np.isnan(r[measure])]
        hc_vals = [r[measure] for r in hc_results if not np.isnan(r[measure])]
        
        # Box plot
        bp = axes[idx].boxplot([ad_vals, hc_vals], labels=['AD', 'HC'],
                               patch_artist=True, widths=0.6)
        
        # Color boxes
        bp['boxes'][0].set_facecolor('#90EE90')
        bp['boxes'][1].set_facecolor('#A9A9A9')
        
        # Add individual points
        for i, vals in enumerate([ad_vals, hc_vals], 1):
            x = np.random.normal(i, 0.04, len(vals))
            axes[idx].scatter(x, vals, alpha=0.5, s=30, c='black', zorder=3)
        
        axes[idx].set_ylabel(measure.replace('_', ' ').title(), fontsize=11)
        axes[idx].set_title(f'{measure.replace("_", " ").title()}\n'
                           f'(n_runs={len(ad_vals)})', 
                           fontsize=11, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
    
    plt.suptitle('Variability Across Different Random Network Realizations', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('network_variability_eeg.png', dpi=300, bbox_inches='tight')
    plt.savefig('network_variability_eeg.pdf', bbox_inches='tight')
    print("✓ Saved network_variability_eeg.png/pdf")


def plot_power_spectra(ad_results, hc_results):
    """Plot average power spectra across all runs."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Average spectra across all runs
    if ad_results and hc_results:
        # Get frequency array (should be same for all runs)
        f = ad_results[0]['f']
        
        # Average power spectra
        ad_pxx = np.mean([r['Pxx'] for r in ad_results], axis=0)
        hc_pxx = np.mean([r['Pxx'] for r in hc_results], axis=0)
        
        # Standard errors
        ad_sem = np.std([r['Pxx'] for r in ad_results], axis=0) / np.sqrt(len(ad_results))
        hc_sem = np.std([r['Pxx'] for r in hc_results], axis=0) / np.sqrt(len(hc_results))
        
        # Plot with error bands
        ax.plot(f, ad_pxx, label=f'AD (n={len(ad_results)})', 
                linewidth=2.5, color='#90EE90')
        ax.fill_between(f, ad_pxx - ad_sem, ad_pxx + ad_sem, 
                        alpha=0.3, color='#90EE90')
        
        ax.plot(f, hc_pxx, label=f'HC (n={len(hc_results)})', 
                linewidth=2.5, color='#A9A9A9')
        ax.fill_between(f, hc_pxx - hc_sem, hc_pxx + hc_sem, 
                        alpha=0.3, color='#A9A9A9')
        
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
        
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Relative power", fontsize=12)
        ax.set_xlim(1, 40)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_title(f"E/I Balance Effects on EEG Power Spectrum (Average across runs)", 
                    fontsize=13)
        
        plt.tight_layout()
        plt.savefig('eeg_power_spectra_averaged.png', dpi=300)
        plt.savefig('eeg_power_spectra_averaged.pdf')
        print("✓ Saved eeg_power_spectra_averaged.png/pdf")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    
    # Network size
    N_TOTAL = 1000
    N_RUNS = 10
    
    print("="*70)
    print(f"E/I Balance Simulation with Cohen's d Analysis")
    print(f"Network size: {N_TOTAL} neurons")
    print(f"Number of runs: {N_RUNS}")
    print("="*70)
    
    # Run simulations with different random networks
    ad_results, hc_results = run_multiple_simulations(n_runs=N_RUNS, N_total=N_TOTAL)
    
    # Extract measures
    ad_sim = extract_measures(ad_results)
    hc_sim = extract_measures(hc_results)
    
    # Compute Cohen's d
    sim_effects = compute_all_cohens_d(ad_sim, hc_sim)
    
    print("\n" + "="*70)
    print("SIMULATION EFFECT SIZES (AD vs HC)")
    print("="*70)
    print(sim_effects[['measure', 'cohens_d', 's_pooled', 'mean_AD', 'mean_HC']].to_string(index=False))
    
    # Plot variability
    plot_variability_across_runs(ad_results, hc_results)
    
    # Plot average power spectra
    plot_power_spectra(ad_results, hc_results)
    
    # Load empirical data (REPLACE WITH YOUR DATA)
    print("\n" + "="*70)
    print("EMPIRICAL DATA (PLACEHOLDER - REPLACE WITH YOUR DATA)")
    print("="*70)
    
    ad_empirical = {
        'aperiodic_exponent': np.random.normal(1.25, 0.18, 15),
        'aperiodic_offset': np.random.normal(-2.3, 0.35, 15),
        'theta_power': np.random.normal(0.24, 0.05, 15),
        'alpha_power': np.random.normal(0.21, 0.04, 15),
        'beta_power': np.random.normal(0.16, 0.03, 15),
        'gamma_power': np.random.normal(0.12, 0.02, 15),
        'spike_rate': np.random.normal(8.5, 1.2, 15)
    }
    
    hc_empirical = {
        'aperiodic_exponent': np.random.normal(1.75, 0.22, 20),
        'aperiodic_offset': np.random.normal(-2.9, 0.40, 20),
        'theta_power': np.random.normal(0.18, 0.04, 20),
        'alpha_power': np.random.normal(0.23, 0.05, 20),
        'beta_power': np.random.normal(0.19, 0.04, 20),
        'gamma_power': np.random.normal(0.14, 0.03, 20),
        'spike_rate': np.random.normal(7.2, 1.0, 20)
    }
    
    # Compute empirical Cohen's d
    emp_effects = compute_all_cohens_d(ad_empirical, hc_empirical)
    
    print("\n" + "="*70)
    print("EMPIRICAL EFFECT SIZES (AD vs HC)")
    print("="*70)
    print(emp_effects[['measure', 'cohens_d', 's_pooled', 'mean_AD', 'mean_HC']].to_string(index=False))
    
    # Compare
    plot_cohens_d_comparison(sim_effects, emp_effects)
    
    # Detailed comparison
    comparison = pd.merge(
        sim_effects[['measure', 'cohens_d', 's_pooled']], 
        emp_effects[['measure', 'cohens_d', 's_pooled']], 
        on='measure', 
        suffixes=('_sim', '_emp')
    )
    
    comparison['d_difference'] = comparison['cohens_d_sim'] - comparison['cohens_d_emp']
    comparison['d_ratio'] = comparison['cohens_d_sim'] / comparison['cohens_d_emp']
    
    print("\n" + "="*70)
    print("DETAILED COMPARISON")
    print("="*70)
    print(comparison.to_string(index=False))
    
    # Goodness of fit
    if len(comparison) > 1:
        corr = np.corrcoef(comparison['cohens_d_sim'], comparison['cohens_d_emp'])[0, 1]
        mae = np.mean(np.abs(comparison['d_difference']))
        rmse = np.sqrt(np.mean(comparison['d_difference']**2))
        
        print("\n" + "="*70)
        print("MODEL FIT METRICS")
        print("="*70)
        print(f"Pearson correlation (r): {corr:7.3f}")
        print(f"Mean Absolute Error:     {mae:7.3f}")
        print(f"Root Mean Square Error:  {rmse:7.3f}")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nFiles generated:")
    print("  • cohens_d_comparison.png/pdf - Effect size comparison")
    print("  • network_variability_eeg.png/pdf - Variability across random networks")
    print("  • eeg_power_spectra_averaged.png/pdf - Average power spectra with SEM")
