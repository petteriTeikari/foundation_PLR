#!/usr/bin/env python3
"""
PLR Waveform Decomposition: Multi-Method Comparison

Decomposes PLR waveforms into 3 canonical physiological components + residual:
1. TRANSIENT/PHASIC - Fast M-pathway response (saturates)
2. SUSTAINED/TONIC - P-pathway + rod-driven response (linear)
3. PIPR - Melanopsin/ipRGC contribution (slow decay)
4. RESIDUAL - Heterogeneous noise (should vary across subjects)

The goal: components sum to reconstruct waveform; residual captures subject variability.

Methods compared:
- Young 1993 Covariance-Based (extended to 3 components)
- Standard PCA (3 components)
- Rotated PCA (Promax, 3 factors)
- Template Fitting (3 physiological templates)

Usage:
    uv run python scripts/plr_decomposition_analysis.py

Reference:
    Young RS, Han BC, Wu PY (1993) Transient and sustained components of the
    pupillary responses evoked by luminance and color. Vision Res 33:437-446
"""

import sys
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA, FastICA

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from project config (NO HARDCODING!)
from src.viz.plot_config import COLORS, save_figure, setup_style  # noqa: E402

# Suppress factor_analyzer convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="factor_analyzer")

try:
    from factor_analyzer import FactorAnalyzer

    HAS_FACTOR_ANALYZER = True
except ImportError:
    HAS_FACTOR_ANALYZER = False
    print("Warning: factor_analyzer not available, skipping Promax rotation")

# Paths
DB_PATH = PROJECT_ROOT.parent / "SERI_PLR_GLAUCOMA.db"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated" / "ggplot2" / "supplementary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Component colors (using semantic COLORS from plot_config)
COMPONENT_COLORS = {
    "transient": COLORS.get("secondary", "#D64045"),  # Coral red
    "sustained": COLORS.get("tertiary", "#45B29D"),  # Teal
    "pipr": COLORS.get("quinary", "#7B68EE"),  # Purple
    "residual": COLORS.get("neutral", "#4A4A4A"),  # Gray
    "observed": COLORS.get("text_primary", "#333333"),
    "reconstructed": COLORS.get("accent", "#F5A623"),
    "blue_stim": COLORS.get("blue_stimulus", "#1f77b4"),
    "red_stim": COLORS.get("red_stimulus", "#d62728"),
    "ci": COLORS.get("grid_lines", "#CCCCCC"),
}


def load_all_subjects():
    """Load pupil_gt from ALL 507 subjects for decomposition analysis."""
    print("Loading data from database...")
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    query = """
    SELECT subject_code, time, pupil_gt
    FROM (
        SELECT subject_code, time, pupil_gt FROM train
        UNION ALL
        SELECT subject_code, time, pupil_gt FROM test
    )
    WHERE pupil_gt IS NOT NULL
    ORDER BY subject_code, time
    """
    df = conn.execute(query).fetchdf()
    conn.close()

    subjects = df["subject_code"].unique()
    print(f"  Found {len(subjects)} unique subjects")

    # Find common timepoints
    time_counts = df.groupby("time").size()
    common_times = time_counts[time_counts > len(subjects) * 0.9].index.values
    print(f"  Using {len(common_times)} common timepoints")

    # Build data matrix
    X = np.zeros((len(subjects), len(common_times)))
    time_vector = np.sort(common_times)

    for i, subj in enumerate(subjects):
        subj_data = df[df["subject_code"] == subj].set_index("time")["pupil_gt"]
        for j, t in enumerate(time_vector):
            if t in subj_data.index:
                X[i, j] = subj_data.loc[t]
            else:
                X[i, j] = np.nan

    # Interpolate missing values
    for i in range(X.shape[0]):
        mask = np.isnan(X[i, :])
        if mask.any() and not mask.all():
            X[i, mask] = np.interp(np.where(mask)[0], np.where(~mask)[0], X[i, ~mask])

    print(f"  Matrix shape: {X.shape} (subjects × timepoints)")
    return X, time_vector, subjects


# =============================================================================
# METHOD 1: Young 1993 Covariance-Based (EXTENDED to 3 components)
# =============================================================================


def young_1993_extended(X, time_vector):
    """
    Extended Young 1993 covariance-based decomposition with 3 components.

    Original method separates transient (saturating) from sustained (linear).
    We extend by adding PIPR extraction from post-stimulus recovery.

    Components:
    1. SUSTAINED: Extracted from late covariance (where transient saturates)
    2. TRANSIENT: Residual during stimulus period
    3. PIPR: Slow decay component in post-stimulus period
    4. RESIDUAL: What remains after removing all 3 components
    """
    print("\n--- Young 1993 Extended (3 Components + Residual) ---")

    n_subjects, n_timepoints = X.shape
    X_centered = X - X.mean(axis=1, keepdims=True)

    # Stimulus timing
    blue_onset, blue_offset = 15.5, 24.5
    red_onset, red_offset = 46.5, 55.5

    # Time masks
    stim1_mask = (time_vector >= blue_onset) & (time_vector <= blue_offset)
    stim2_mask = (time_vector >= red_onset) & (time_vector <= red_offset)
    post_stim1 = (time_vector > blue_offset) & (time_vector < red_onset)
    post_stim2 = time_vector > red_offset
    late_mask = time_vector > 58  # Very late for sustained estimation

    # --- 1. Extract SUSTAINED component from late covariance structure ---
    cov_matrix = np.cov(X_centered.T)
    late_times = time_vector[late_mask]
    sustained_shape = np.zeros(n_timepoints)

    for t_idx in range(n_timepoints):
        cov_with_late = cov_matrix[t_idx, late_mask]
        if len(cov_with_late) > 2:
            coeffs = np.polyfit(late_times - late_times.mean(), cov_with_late, 1)
            sustained_shape[t_idx] = coeffs[0]

    if np.abs(sustained_shape).max() > 0:
        sustained_shape = sustained_shape / np.abs(sustained_shape).max()

    # Project and remove sustained
    sustained_amp = (
        X_centered @ sustained_shape / (sustained_shape @ sustained_shape + 1e-10)
    )
    X_no_sustained = X_centered - np.outer(sustained_amp, sustained_shape)

    # --- 2. Extract PIPR from post-stimulus decay ---
    # PIPR should show slow exponential decay after stimulus offset
    pipr_shape = np.zeros(n_timepoints)

    # Create PIPR template: exponential decay starting at stimulus offsets
    tau_pipr = 15.0
    t_from_blue_off = time_vector - blue_offset
    t_from_red_off = time_vector - red_offset
    pipr_shape[post_stim1] = np.exp(-t_from_blue_off[post_stim1] / tau_pipr)
    pipr_shape[post_stim2] = np.exp(-t_from_red_off[post_stim2] / tau_pipr)

    if np.abs(pipr_shape).max() > 0:
        pipr_shape = -pipr_shape / np.abs(pipr_shape).max()  # Constriction is negative

    # Project and remove PIPR
    pipr_amp = X_no_sustained @ pipr_shape / (pipr_shape @ pipr_shape + 1e-10)
    X_no_sustained_pipr = X_no_sustained - np.outer(pipr_amp, pipr_shape)

    # --- 3. Extract TRANSIENT as residual during stimulus periods ---
    # Transient should be prominent during stimulus onset
    # Note: stim_mask could be used for more sophisticated transient extraction
    _ = stim1_mask | stim2_mask  # Available for future use
    transient_shape = X_no_sustained_pipr.mean(axis=0)

    if np.abs(transient_shape).max() > 0:
        transient_shape = transient_shape / np.abs(transient_shape).max()

    transient_amp = (
        X_no_sustained_pipr
        @ transient_shape
        / (transient_shape @ transient_shape + 1e-10)
    )
    X_residual = X_no_sustained_pipr - np.outer(transient_amp, transient_shape)

    # --- Compute variance explained ---
    total_var = np.var(X_centered)
    sustained_var = np.var(np.outer(sustained_amp, sustained_shape))
    pipr_var = np.var(np.outer(pipr_amp, pipr_shape))
    transient_var = np.var(np.outer(transient_amp, transient_shape))
    residual_var = np.var(X_residual)

    print(f"  Sustained: {100 * sustained_var / total_var:.1f}% variance")
    print(f"  PIPR: {100 * pipr_var / total_var:.1f}% variance")
    print(f"  Transient: {100 * transient_var / total_var:.1f}% variance")
    print(f"  Residual: {100 * residual_var / total_var:.1f}% variance")

    # Check residual heterogeneity (should be high variance across subjects)
    residual_per_subject = np.std(X_residual, axis=1)
    print(
        f"  Residual std across subjects: mean={residual_per_subject.mean():.2f}, "
        f"range=[{residual_per_subject.min():.2f}, {residual_per_subject.max():.2f}]"
    )

    return {
        "transient_shape": transient_shape,
        "sustained_shape": sustained_shape,
        "pipr_shape": pipr_shape,
        "transient_amp": transient_amp,
        "sustained_amp": sustained_amp,
        "pipr_amp": pipr_amp,
        "residual": X_residual,
        "variance_explained": {
            "transient": transient_var / total_var,
            "sustained": sustained_var / total_var,
            "pipr": pipr_var / total_var,
            "residual": residual_var / total_var,
        },
        "residual_std_per_subject": residual_per_subject,
    }


# =============================================================================
# METHOD 2: Standard PCA (3 components)
# =============================================================================


def pca_decomposition(X, time_vector, n_components=3):
    """Standard PCA with 3 components + residual."""
    print(f"\n--- Standard PCA (n_components={n_components}) ---")

    X_centered = X - X.mean(axis=0)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_centered)
    loadings = pca.components_

    # Reconstruction with 3 components
    X_reconstructed = scores @ loadings + X.mean(axis=0)
    X_residual = X - X_reconstructed

    print(f"  Explained variance: {pca.explained_variance_ratio_[:3]}")
    print(f"  Cumulative: {pca.explained_variance_ratio_.cumsum()[:3]}")

    residual_std = np.std(X_residual, axis=1)
    print(f"  Residual std across subjects: mean={residual_std.mean():.2f}")

    return {
        "loadings": loadings,
        "scores": scores,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "residual": X_residual,
        "residual_std_per_subject": residual_std,
    }


# =============================================================================
# METHOD 3: Rotated PCA (Promax, 3 factors)
# =============================================================================


def rotated_pca_decomposition(X, time_vector, n_factors=3):
    """Rotated PCA with Promax oblique rotation (Bustos 2024 approach)."""
    print(f"\n--- Rotated PCA (Promax, n_factors={n_factors}) ---")

    if not HAS_FACTOR_ANALYZER:
        print("  Skipping: factor_analyzer not available")
        return None

    X_centered = X - X.mean(axis=0)

    try:
        fa = FactorAnalyzer(n_factors=n_factors, rotation="promax", method="principal")
        fa.fit(X_centered)

        loadings = fa.loadings_.T  # n_factors × n_timepoints
        phi = fa.phi_
        scores = fa.transform(X_centered)

        # Reconstruct and compute residual
        X_reconstructed = scores @ loadings + X.mean(axis=0)
        X_residual = X - X_reconstructed

        print(f"  Factor correlations:\n{phi}")
        residual_std = np.std(X_residual, axis=1)
        print(f"  Residual std across subjects: mean={residual_std.mean():.2f}")

        return {
            "loadings": loadings,
            "scores": scores,
            "phi": phi,
            "residual": X_residual,
            "residual_std_per_subject": residual_std,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


# =============================================================================
# METHOD 4: Template Fitting (3 physiological templates)
# =============================================================================


def template_fitting(X, time_vector):
    """Fit 3 physiological templates + residual."""
    print("\n--- Template Fitting (3 Physiological Components) ---")

    mean_waveform = X.mean(axis=0)
    n_t = len(time_vector)

    # Stimulus timing
    blue_onset, blue_offset = 15.5, 24.5
    red_onset, red_offset = 46.5, 55.5

    def make_template(onset, offset, tau_rise, tau_decay):
        template = np.zeros(n_t)
        t_from_onset = time_vector - onset
        t_from_offset = time_vector - offset

        stim_mask = (t_from_onset >= 0) & (t_from_offset <= 0)
        template[stim_mask] = (
            1 - np.exp(-t_from_onset[stim_mask] / tau_rise)
        ) * np.exp(-t_from_onset[stim_mask] / tau_decay)

        post_mask = t_from_offset > 0
        if stim_mask.any():
            peak_val = (
                template[stim_mask].max() if template[stim_mask].max() > 0 else 1.0
            )
            template[post_mask] = peak_val * np.exp(
                -t_from_offset[post_mask] / tau_decay
            )

        return -template  # Constriction is negative

    def make_pipr_template(offset, tau):
        template = np.zeros(n_t)
        t_from_off = time_vector - offset
        mask = t_from_off > 0
        template[mask] = np.exp(-t_from_off[mask] / tau)
        return -template

    # Create templates for both stimuli
    transient = make_template(blue_onset, blue_offset, 0.3, 1.0) + make_template(
        red_onset, red_offset, 0.3, 1.0
    )
    sustained = make_template(blue_onset, blue_offset, 2.0, 3.0) + make_template(
        red_onset, red_offset, 2.0, 3.0
    )
    pipr = make_pipr_template(blue_offset, 15.0) + make_pipr_template(red_offset, 15.0)

    # Normalize
    transient = transient / (np.abs(transient).max() + 1e-10)
    sustained = sustained / (np.abs(sustained).max() + 1e-10)
    pipr = pipr / (np.abs(pipr).max() + 1e-10)

    # Fit to mean waveform
    design = np.column_stack([transient, sustained, pipr, np.ones(n_t)])
    coeffs, _, _, _ = np.linalg.lstsq(design, mean_waveform, rcond=None)

    reconstructed = design @ coeffs
    rmse = np.sqrt(np.mean((mean_waveform - reconstructed) ** 2))

    # Compute per-subject residuals
    X_residual = np.zeros_like(X)
    for i in range(X.shape[0]):
        coeffs_i, _, _, _ = np.linalg.lstsq(design, X[i], rcond=None)
        X_residual[i] = X[i] - design @ coeffs_i

    residual_std = np.std(X_residual, axis=1)

    print(f"  Transient: {coeffs[0]:.1f}%")
    print(f"  Sustained: {coeffs[1]:.1f}%")
    print(f"  PIPR: {coeffs[2]:.1f}%")
    print(f"  Baseline: {coeffs[3]:.1f}%")
    print(f"  RMSE: {rmse:.2f}%")
    print(f"  Residual std across subjects: mean={residual_std.mean():.2f}")

    return {
        "templates": {
            "transient": transient,
            "sustained": sustained,
            "pipr": pipr,
        },
        "amplitudes": {
            "transient": coeffs[0],
            "sustained": coeffs[1],
            "pipr": coeffs[2],
            "baseline": coeffs[3],
        },
        "components": {
            "transient": transient * coeffs[0],
            "sustained": sustained * coeffs[1],
            "pipr": pipr * coeffs[2],
        },
        "reconstructed": reconstructed,
        "rmse": rmse,
        "residual": X_residual,
        "residual_std_per_subject": residual_std,
    }


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def create_comparison_figure(time_vector, X, results, n_subjects):
    """Create publication figure comparing decomposition methods."""
    print("\nCreating comparison figure...")

    # Apply style from config (NO HARDCODING!)
    setup_style()

    mean_waveform = X.mean(axis=0)
    std_waveform = X.std(axis=0)
    se = std_waveform / np.sqrt(n_subjects)

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Panel A: Mean Waveform
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_mean.set_title(
        f"A  Mean PLR Waveform (N={n_subjects})", fontweight="bold", loc="left"
    )

    ax_mean.fill_between(
        time_vector,
        mean_waveform - 1.96 * se,
        mean_waveform + 1.96 * se,
        color=COMPONENT_COLORS["ci"],
        alpha=0.5,
        label="95% CI",
    )
    ax_mean.plot(
        time_vector,
        mean_waveform,
        color=COMPONENT_COLORS["observed"],
        linewidth=2,
        label="Mean waveform",
    )

    ax_mean.axvspan(15.5, 24.5, alpha=0.15, color=COMPONENT_COLORS["blue_stim"])
    ax_mean.axvspan(46.5, 55.5, alpha=0.15, color=COMPONENT_COLORS["red_stim"])
    ax_mean.set_xlabel("Time (s)")
    ax_mean.set_ylabel("Pupil Size (% change)")
    ax_mean.set_xlim(0, time_vector[-1])
    ax_mean.legend(loc="lower right", fontsize=8)
    ax_mean.axhline(0, color="gray", linestyle="--", alpha=0.3)

    # Panel B: Young 1993 Extended (3 components)
    ax_young = fig.add_subplot(gs[0, 1])
    ax_young.set_title(
        "B  Young 1993 Extended (3 Components)", fontweight="bold", loc="left"
    )

    young = results["young_1993"]
    ax_young.plot(
        time_vector,
        young["transient_shape"],
        color=COMPONENT_COLORS["transient"],
        linewidth=2,
        label=f"Transient ({100 * young['variance_explained']['transient']:.1f}%)",
    )
    ax_young.plot(
        time_vector,
        young["sustained_shape"],
        color=COMPONENT_COLORS["sustained"],
        linewidth=2,
        label=f"Sustained ({100 * young['variance_explained']['sustained']:.1f}%)",
    )
    ax_young.plot(
        time_vector,
        young["pipr_shape"],
        color=COMPONENT_COLORS["pipr"],
        linewidth=2,
        label=f"PIPR ({100 * young['variance_explained']['pipr']:.1f}%)",
    )

    ax_young.axvspan(15.5, 24.5, alpha=0.1, color=COMPONENT_COLORS["blue_stim"])
    ax_young.axvspan(46.5, 55.5, alpha=0.1, color=COMPONENT_COLORS["red_stim"])
    ax_young.set_xlabel("Time (s)")
    ax_young.set_ylabel("Component Shape (normalized)")
    ax_young.set_xlim(0, time_vector[-1])
    ax_young.legend(loc="lower right", fontsize=8)
    ax_young.axhline(0, color="gray", linestyle="--", alpha=0.3)

    # Panel C: Standard PCA (3 components)
    ax_pca = fig.add_subplot(gs[1, 0])
    ax_pca.set_title("C  Standard PCA (3 Components)", fontweight="bold", loc="left")

    pca = results["pca"]
    pca_colors = [
        COMPONENT_COLORS["transient"],
        COMPONENT_COLORS["sustained"],
        COMPONENT_COLORS["pipr"],
    ]
    for i in range(3):
        var_pct = 100 * pca["explained_variance_ratio"][i]
        ax_pca.plot(
            time_vector,
            pca["loadings"][i],
            color=pca_colors[i],
            linewidth=2,
            label=f"PC{i + 1} ({var_pct:.1f}%)",
        )

    ax_pca.axvspan(15.5, 24.5, alpha=0.1, color=COMPONENT_COLORS["blue_stim"])
    ax_pca.axvspan(46.5, 55.5, alpha=0.1, color=COMPONENT_COLORS["red_stim"])
    ax_pca.set_xlabel("Time (s)")
    ax_pca.set_ylabel("Loading (normalized)")
    ax_pca.set_xlim(0, time_vector[-1])
    ax_pca.legend(loc="lower right", fontsize=8)
    ax_pca.axhline(0, color="gray", linestyle="--", alpha=0.3)

    # Panel D: Rotated PCA or ICA
    ax_rot = fig.add_subplot(gs[1, 1])
    if results["rotated_pca"] is not None:
        ax_rot.set_title(
            "D  Rotated PCA (Promax, 3 Factors)", fontweight="bold", loc="left"
        )
        rot = results["rotated_pca"]
        for i in range(3):
            loading = rot["loadings"][i] / (np.abs(rot["loadings"][i]).max() + 1e-10)
            ax_rot.plot(
                time_vector,
                loading,
                color=pca_colors[i],
                linewidth=2,
                label=f"RC{i + 1}",
            )

        phi = rot["phi"]
        corr_text = f"Factor correlations:\nRC1-RC2: {phi[0, 1]:.2f}\nRC1-RC3: {phi[0, 2]:.2f}\nRC2-RC3: {phi[1, 2]:.2f}"
        ax_rot.text(
            0.02,
            0.98,
            corr_text,
            transform=ax_rot.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    else:
        ax_rot.set_title("D  ICA Components (3)", fontweight="bold", loc="left")
        X_centered = X - X.mean(axis=0)
        ica = FastICA(n_components=3, random_state=42)
        ica.fit(X_centered)
        for i in range(3):
            comp = ica.components_[i] / (np.abs(ica.components_[i]).max() + 1e-10)
            ax_rot.plot(
                time_vector, comp, color=pca_colors[i], linewidth=2, label=f"IC{i + 1}"
            )

    ax_rot.axvspan(15.5, 24.5, alpha=0.1, color=COMPONENT_COLORS["blue_stim"])
    ax_rot.axvspan(46.5, 55.5, alpha=0.1, color=COMPONENT_COLORS["red_stim"])
    ax_rot.set_xlabel("Time (s)")
    ax_rot.set_ylabel("Loading (normalized)")
    ax_rot.set_xlim(0, time_vector[-1])
    ax_rot.legend(loc="lower right", fontsize=8)
    ax_rot.axhline(0, color="gray", linestyle="--", alpha=0.3)

    # Panel E: Template Fitting (3 components)
    ax_tmpl = fig.add_subplot(gs[2, 0])
    ax_tmpl.set_title(
        "E  Template Fitting (3 Components)", fontweight="bold", loc="left"
    )

    tmpl = results["template"]
    comps = tmpl["components"]
    amps = tmpl["amplitudes"]

    ax_tmpl.plot(
        time_vector,
        comps["transient"],
        color=COMPONENT_COLORS["transient"],
        linewidth=2,
        label=f"Transient ({amps['transient']:.0f}%)",
    )
    ax_tmpl.plot(
        time_vector,
        comps["sustained"],
        color=COMPONENT_COLORS["sustained"],
        linewidth=2,
        label=f"Sustained ({amps['sustained']:.0f}%)",
    )
    ax_tmpl.plot(
        time_vector,
        comps["pipr"],
        color=COMPONENT_COLORS["pipr"],
        linewidth=2,
        label=f"PIPR ({amps['pipr']:.0f}%)",
    )

    ax_tmpl.axvspan(15.5, 24.5, alpha=0.1, color=COMPONENT_COLORS["blue_stim"])
    ax_tmpl.axvspan(46.5, 55.5, alpha=0.1, color=COMPONENT_COLORS["red_stim"])
    ax_tmpl.set_xlabel("Time (s)")
    ax_tmpl.set_ylabel("Component Amplitude (%)")
    ax_tmpl.set_xlim(0, time_vector[-1])
    ax_tmpl.legend(loc="lower right", fontsize=8)
    ax_tmpl.axhline(0, color="gray", linestyle="--", alpha=0.3)

    ax_tmpl.text(
        0.02,
        0.98,
        f"RMSE: {tmpl['rmse']:.1f}%",
        transform=ax_tmpl.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Panel F: Linear Combination (stacked)
    ax_stack = fig.add_subplot(gs[2, 1])
    ax_stack.set_title(
        "F  Linear Combination (3 Components)", fontweight="bold", loc="left"
    )

    baseline = amps["baseline"]
    ax_stack.fill_between(
        time_vector,
        baseline,
        baseline + comps["pipr"],
        color=COMPONENT_COLORS["pipr"],
        alpha=0.6,
        label="PIPR",
    )
    ax_stack.fill_between(
        time_vector,
        baseline + comps["pipr"],
        baseline + comps["pipr"] + comps["sustained"],
        color=COMPONENT_COLORS["sustained"],
        alpha=0.6,
        label="Sustained",
    )
    ax_stack.fill_between(
        time_vector,
        baseline + comps["pipr"] + comps["sustained"],
        baseline + comps["pipr"] + comps["sustained"] + comps["transient"],
        color=COMPONENT_COLORS["transient"],
        alpha=0.6,
        label="Transient",
    )

    ax_stack.plot(
        time_vector,
        mean_waveform,
        color=COMPONENT_COLORS["observed"],
        linewidth=2,
        label="Observed",
    )

    ax_stack.axvspan(15.5, 24.5, alpha=0.1, color=COMPONENT_COLORS["blue_stim"])
    ax_stack.axvspan(46.5, 55.5, alpha=0.1, color=COMPONENT_COLORS["red_stim"])
    ax_stack.set_xlabel("Time (s)")
    ax_stack.set_ylabel("Pupil Size (%)")
    ax_stack.set_xlim(0, time_vector[-1])
    ax_stack.legend(loc="lower right", fontsize=8)
    ax_stack.axhline(0, color="gray", linestyle="--", alpha=0.3)

    # Panel G: Residual Distribution (KEY: shows heterogeneous noise)
    ax_resid = fig.add_subplot(gs[3, 0])
    ax_resid.set_title(
        "G  Residual Std per Subject (Noise Heterogeneity)",
        fontweight="bold",
        loc="left",
    )

    # Show residual std for each method
    methods = ["Young 1993", "PCA", "Template"]
    residual_data = [
        young["residual_std_per_subject"],
        pca["residual_std_per_subject"],
        tmpl["residual_std_per_subject"],
    ]
    positions = range(len(methods))
    bp = ax_resid.boxplot(
        residual_data, positions=positions, widths=0.6, patch_artist=True
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(pca_colors[i])
        patch.set_alpha(0.6)

    ax_resid.set_xticks(positions)
    ax_resid.set_xticklabels(methods)
    ax_resid.set_ylabel("Residual Std (% pupil)")
    ax_resid.set_xlabel("Decomposition Method")

    ax_resid.text(
        0.02,
        0.98,
        "Higher = more inter-subject variability\nin unexplained signal",
        transform=ax_resid.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Panel H: Summary Comparison
    ax_summ = fig.add_subplot(gs[3, 1])
    ax_summ.set_title(
        "H  Variance Decomposition Summary", fontweight="bold", loc="left"
    )

    # Create stacked bar chart
    methods = ["Young 1993\nExtended", "Template\nFitting"]
    comp_names = ["Transient", "Sustained", "PIPR", "Residual"]
    comp_colors = [
        COMPONENT_COLORS["transient"],
        COMPONENT_COLORS["sustained"],
        COMPONENT_COLORS["pipr"],
        COMPONENT_COLORS["residual"],
    ]

    young_var = young["variance_explained"]
    tmpl_var_total = np.var(X - X.mean(axis=1, keepdims=True))
    tmpl_comp_vars = {
        "transient": np.var(comps["transient"]) / tmpl_var_total
        if tmpl_var_total > 0
        else 0,
        "sustained": np.var(comps["sustained"]) / tmpl_var_total
        if tmpl_var_total > 0
        else 0,
        "pipr": np.var(comps["pipr"]) / tmpl_var_total if tmpl_var_total > 0 else 0,
    }
    tmpl_comp_vars["residual"] = 1 - sum(tmpl_comp_vars.values())

    data = (
        np.array(
            [
                [young_var["transient"], tmpl_comp_vars["transient"]],
                [young_var["sustained"], tmpl_comp_vars["sustained"]],
                [young_var["pipr"], tmpl_comp_vars["pipr"]],
                [young_var["residual"], tmpl_comp_vars["residual"]],
            ]
        )
        * 100
    )

    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for i, (comp, color) in enumerate(zip(comp_names, comp_colors)):
        ax_summ.bar(x, data[i], 0.6, bottom=bottom, label=comp, color=color, alpha=0.8)
        bottom += data[i]

    ax_summ.set_xticks(x)
    ax_summ.set_xticklabels(methods)
    ax_summ.set_ylabel("Variance Explained (%)")
    ax_summ.legend(loc="upper right", fontsize=8)
    ax_summ.set_ylim(0, 105)

    plt.tight_layout()
    return fig


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("PLR Waveform Decomposition: 3 Components + Residual")
    print("=" * 70)

    X, time_vector, subjects = load_all_subjects()
    n_subjects = X.shape[0]

    # Run all methods
    results = {
        "young_1993": young_1993_extended(X, time_vector),
        "pca": pca_decomposition(X, time_vector, n_components=3),
        "rotated_pca": rotated_pca_decomposition(X, time_vector, n_factors=3),
        "template": template_fitting(X, time_vector),
    }

    # Create figure
    fig = create_comparison_figure(time_vector, X, results, n_subjects)

    # Save using proper function (NO HARDCODING!)
    output_path = save_figure(
        fig,
        "fig_plr_decomposition",
        output_dir=OUTPUT_DIR,
        formats=["png"],
        data={
            "title": "PLR Waveform Decomposition: 3 Components + Residual",
            "n_subjects": int(n_subjects),
            "n_timepoints": len(time_vector),
            "young_1993": {
                "variance_explained": {
                    k: round(v, 3)
                    for k, v in results["young_1993"]["variance_explained"].items()
                },
            },
            "pca": {
                "explained_variance_ratio": [
                    round(v, 3) for v in results["pca"]["explained_variance_ratio"][:3]
                ],
            },
            "template": {
                "transient_pct": round(
                    results["template"]["amplitudes"]["transient"], 1
                ),
                "sustained_pct": round(
                    results["template"]["amplitudes"]["sustained"], 1
                ),
                "pipr_pct": round(results["template"]["amplitudes"]["pipr"], 1),
                "rmse_pct": round(results["template"]["rmse"], 2),
            },
            "key_insight": "Residual captures heterogeneous inter-subject noise",
        },
    )

    print(f"\nSaved figure: {output_path}")
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    plt.close("all")


if __name__ == "__main__":
    main()
