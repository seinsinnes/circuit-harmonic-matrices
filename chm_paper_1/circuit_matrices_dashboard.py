
#
# Visualise the Fourier-coefficient correlation matrices Corr_C and Corr_MC
# as heatmaps, for a sequence of training depths from a single compute run.
#
# plots three heatmap grids:
#   - Re(Corr)  : real part of the correlation matrix
#   - Im(Corr)  : imaginary part
#   - |Corr|    : absolute value (magnitude)
#   row 0: C-matrix prediction (Corr_C)
#   row 1: Direct Monte Carlo estimate (Corr_MC)
# A textbox below each column shows scalar diagnostics (Frobenius error,
# cosine similarity of off-diagonal entries, and mean off-diagonal correlation).



import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import multiprocessing as mp




from ansatze import CircuitSpec, build_qnode

from correlation_matrices_plot import relative_support_mask, apply_support_mask_matrix, mask_diagonal_for_display, \
                                        offdiag_vmax_diverging, offdiag_vmax_abs, omega_side_mask, omega_phys_mask, \
                                        restrict_by_mask, restrict_vector_list, combine_support_masks, \
                                        _fallback_metrics_complex_parts, set_omega_ticks, _dig, _get_se, fmt_pm

from variance_profile_plot import restrict_to_omega_band, transform_profile, textbox_lines

from circuit_matrices_sampling import compute_worker

def plot_grid_figure_hermitian(
    data,
    omega_grid,
    title_prefix,
    clip_q=0.995,
    omega_phys=None,
    omega_side="all",
    mask_diag="none",
    support_rel_var=None,
    support_mode="intersection",
):
    assert omega_side in ("all", "nonneg", "nonpos")
    assert mask_diag in ("none", "zero", "nan")

    depths = [data["train_depth"]]

    Corr_C0  = [data["Corr_C"]]
    Corr_MC0 = [data["Corr_MC"]]
    var_C0   = [data["var_C"]]
    var_MC0  = [data["var_MC"]]


    ims = []
    
    # Layout
    ncols = 3
    fig_h = 7.2
    fig_w = max(8.0, 3.0 * ncols)
    fig, axes = plt.subplots(
        3, ncols, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.48]},
        constrained_layout=True
    )
    if ncols == 1:
        axes = np.array(axes).reshape(3, 1)

    side_note = {"all": "all ω", "nonneg": "ω≥0", "nonpos": "ω≤0"}[omega_side]
    band_note = f", |ω|≤{int(omega_phys)}" if omega_phys is not None else ""
    supp_note = ""
    if support_rel_var is not None:
        supp_note = f", support>{support_rel_var:.0e} ({support_mode})"

    fig.suptitle(
        f"Hermitian Corr(a) | {title_prefix} | {side_note}{band_note}{supp_note}",
        fontsize=14
    )

    
    for p, part in enumerate(("real", "imag", "abs")):

        if part == "real":
            A_C_raw0  = [np.real(A) for A in Corr_C0]
            A_MC_raw0 = [np.real(A) for A in Corr_MC0]
            part_label = "Re"
            cmap = "coolwarm"
            diverging = True
        elif part == "imag":
            A_C_raw0  = [np.imag(A) for A in Corr_C0]
            A_MC_raw0 = [np.imag(A) for A in Corr_MC0]
            part_label = "Im"
            cmap = "coolwarm"
            diverging = True
        else:
            A_C_raw0  = [np.abs(A) for A in Corr_C0]
            A_MC_raw0 = [np.abs(A) for A in Corr_MC0]
            part_label = "|.|"
            cmap = "viridis"
            diverging = False

        # Restrict omega consistently
        keep = omega_side_mask(omega_grid, omega_side) & omega_phys_mask(omega_grid, omega_phys)

        A_C_raw, omega_grid_r  = restrict_by_mask(A_C_raw0,  omega_grid, keep)
        A_MC_raw, _            = restrict_by_mask(A_MC_raw0, omega_grid, keep)
        var_list_r, _          = restrict_vector_list([*var_C0, *var_MC0], omega_grid, keep)
        var_C_r = var_list_r[:len(var_C0)]
        var_MC_r = var_list_r[len(var_C0):]

        # Apply support mask based on relative variances
        support_masks = []
        if support_rel_var is not None:
            A_C_masked = []
            A_MC_masked = []
            for j in range(ncols):
                mC = relative_support_mask(var_C_r[j], support_rel_var)
                mM = relative_support_mask(var_MC_r[j], support_rel_var)
                ms = combine_support_masks(mC, mM, support_mode)
                support_masks.append(ms)
                A_C_masked.append(apply_support_mask_matrix(A_C_raw[j], ms))
                A_MC_masked.append(apply_support_mask_matrix(A_MC_raw[j], ms))
            A_C_raw = A_C_masked
            A_MC_raw = A_MC_masked
        else:
            support_masks = [None] * ncols

        # Display mask (display-only)
        if mask_diag == "none":
            A_C_disp  = [np.array(A, copy=True) for A in A_C_raw]
            A_MC_disp = [np.array(A, copy=True) for A in A_MC_raw]
        else:
            A_C_disp  = [mask_diagonal_for_display(A, mode=mask_diag) for A in A_C_raw]
            A_MC_disp = [mask_diagonal_for_display(A, mode=mask_diag) for A in A_MC_raw]

        # Color scaling
        if diverging:
            vmax = offdiag_vmax_diverging(A_C_disp + A_MC_disp, clip_q=clip_q)
            vmin = -vmax
        else:
            vmax = offdiag_vmax_abs(A_C_disp + A_MC_disp, clip_q=clip_q)
            vmin = 0.0

        # Metrics fallback computed from masked meaningful matrices
        fb_errs, fb_cos, fb_mean_offdiag_C, fb_mean_offdiag_M = _fallback_metrics_complex_parts(A_C_raw, A_MC_raw)
        
        for j, depth in enumerate(depths[:1]):
            ax0 = axes[0, p]
            im0 = ax0.imshow(
                np.ma.masked_invalid(A_C_disp[j]),
                origin="lower", aspect="auto",
                vmin=vmin, vmax=vmax, cmap=cmap
            )
            ax0.set_title(f"depth={depth} | {part_label} | C")
            set_omega_ticks(ax0, omega_grid_r)
            ims.append(im0)

            ax1 = axes[1, p]
            im1 = ax1.imshow(
                np.ma.masked_invalid(A_MC_disp[j]),
                origin="lower", aspect="auto",
                vmin=vmin, vmax=vmax, cmap=cmap
            )
            ax1.set_title(f"depth={depth} | {part_label} | MC")
            set_omega_ticks(ax1, omega_grid_r)
            ims.append(im1)

            diag = data.get("diag", {}) or {}

            # defaults from fallback on masked matrices
            err_mean = fb_errs[j]; err_se = np.nan
            cos_mean = fb_cos[j];  cos_se = np.nan
            mean_offdiag_C_mean = fb_mean_offdiag_C[j]; mean_offdiag_C_se = np.nan
            mean_offdiag_M_mean = fb_mean_offdiag_M[j]; mean_offdiag_M_se = np.nan

            # Prefer diag_json only when no support masking is applied.
            # Once support masking is applied, the fallback metrics are the relevant ones.
            if (support_rel_var is None) and isinstance(diag, dict) and diag:
                if part in ("real", "imag"):
                    parts = _dig(diag, "complex", "parts", default=None)
                    group = "complex_re" if part == "real" else "complex_im"
                    tag = "re" if part == "real" else "im"

                    if isinstance(parts, dict):
                        v = parts.get(f"corr_rel_frob_{tag}", None)
                        if v is not None: err_mean = float(v)
                        v = parts.get(f"corr_offdiag_cosine_{tag}", None)
                        if v is not None: cos_mean = float(v)
                        v = parts.get(f"corrC_mean_abs_offdiag_{tag}", None)
                        if v is not None: mean_offdiag_C_mean = float(v)
                        v = parts.get(f"corr_mean_abs_offdiag_{tag}", None)
                        if v is not None: mean_offdiag_M_mean = float(v)

                    err_se             = _get_se(diag, group, "corr_rel_frob_se", default=np.nan)
                    cos_se             = _get_se(diag, group, "corr_offdiag_cosine_se", default=np.nan)
                    mean_offdiag_C_se  = _get_se(diag, group, "corr_mean_abs_offdiag_c_se", default=np.nan)
                    mean_offdiag_M_se  = _get_se(diag, group, "corr_mean_abs_offdiag_mc_se", default=np.nan)

                elif part == "abs":
                    comp = _dig(diag, "complex", default=None)
                    if isinstance(comp, dict):
                        v = comp.get("corr_rel_frob", None)
                        if v is not None: err_mean = float(v)
                        v = comp.get("corr_offdiag_cosine", None)
                        if v is not None: cos_mean = float(v)
                        v = comp.get("corrC_mean_abs_offdiag", None)
                        if v is not None: mean_offdiag_C_mean = float(v)
                        v = comp.get("corr_mean_abs_offdiag", None)
                        if v is not None: mean_offdiag_M_mean = float(v)

                    err_se             = _get_se(diag, "complex", "corr_rel_frob_se", default=np.nan)
                    cos_se             = _get_se(diag, "complex", "corr_offdiag_cosine_se", default=np.nan)
                    mean_offdiag_C_se  = _get_se(diag, "complex", "corr_mean_abs_offdiag_c_se", default=np.nan)
                    mean_offdiag_M_se  = _get_se(diag, "complex", "corr_mean_abs_offdiag_mc_se", default=np.nan)

            ax2 = axes[2, p]
            ax2.axis("off")

            support_txt = ""
            if support_masks[j] is not None:
                nsupp = int(np.count_nonzero(support_masks[j]))
                support_txt = f"support size: {nsupp}/{len(support_masks[j])}\n"

            ax2.text(
                0.5, 0.5,
                support_txt +
                "Frobenius err (MC vs C):\n"
                f"{fmt_pm(err_mean, err_se, fmt_val='{:.3e}', fmt_err='{:.1e}')}\n"
                "Cos(offdiag) (MC vs C):\n"
                f"{fmt_pm(cos_mean, cos_se, fmt_val='{:.3f}', fmt_err='{:.3f}')}\n"
                "mean|offdiag(Corr)|:\n"
                f"C:  {fmt_pm(mean_offdiag_C_mean, mean_offdiag_C_se, fmt_val='{:.3e}', fmt_err='{:.1e}')}\n"
                f"MC: {fmt_pm(mean_offdiag_M_mean, mean_offdiag_M_se, fmt_val='{:.3e}', fmt_err='{:.1e}')}\n",
                ha="center", va="center", fontsize=10.0
            )

        cbar = fig.colorbar(ims[-1], ax=axes[0:2, :], fraction=0.025, pad=0.02)
        if part == "abs":
            cbar.set_label(r"$|{\rm Corr}(\omega,\omega')|$")
        else:
            cbar.set_label(f"{part_label} value" + ("" if mask_diag == "none" else " (diag masked)"))

    st.pyplot(fig)
    plt.close(fig)


def plot_grid(
    data,
    mode="relmax",
    omega_phys=None,
    omega_side="all",
    show_textbox=True,
    show_legend=True,
    ylim=None,
    title=None,
):
    #print(rows)
    depths = [data["train_depth"]]
    ncols = 1

    fig, axes = plt.subplots(
        2, ncols,
        figsize=(4.8 * ncols, 5.4),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.0, 0.35]}
    )

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.10, top=0.84, wspace=0.25, hspace=0.28)

    title_fs = 18
    axis_label_fs = 15
    tick_fs = 11
    legend_fs = 13
    textbox_fs = 15
    suptitle_fs = 18

    j = 0

    r = data
    ax = axes[0, j]

    omega, arrs = restrict_to_omega_band(
        r["omega_grid"],
        [r["var_C"], r["var_MC"], None, None],
        omega_phys=omega_phys,
        omega_side=omega_side,
    )
    var_C, var_MC, var_C_se, var_MC_se = arrs

    yC, yC_se, mode_label, ylabel = transform_profile(var_C, var_C_se, mode)
    yM, yM_se, _, _ = transform_profile(var_MC, var_MC_se, mode)

    if yC_se is not None:
        ax.errorbar(
            omega, yC, yerr=yC_se,
            fmt="s--", linewidth=1.0, markersize=5,
            capsize=2.5, label=r"Row Energy of $C$"
        )
    else:
        ax.plot(
            omega, yC,
            "s--", linewidth=1.0, markersize=5,
            label=r"Row Energy of $C$"
        )

    if yM_se is not None:
        ax.errorbar(
            omega, yM, yerr=yM_se,
            fmt="^-.", linewidth=1.0, markersize=5,
            capsize=2.5, label=r"Variance from MC"
        )
    else:
        ax.plot(
            omega, yM,
            "^-.", linewidth=1.0, markersize=5,
            label=r"Variance from MC"
        )

    ax.set_title(f"depth={r['train_depth']}", fontsize=title_fs)
    ax.set_xlabel(r"frequency $\omega$", fontsize=axis_label_fs)
    ax.set_ylabel(ylabel, fontsize=axis_label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax_txt = axes[1, j]
    ax_txt.axis("off")
    if show_textbox:
        txt = textbox_lines(r["diag"], var_C, var_MC, yC, yM, mode_label)
        ax_txt.text(
            0.5, 0.5, txt,
            ha="center", va="center", fontsize=textbox_fs
        )

    if show_legend:
        axes[0, 0].legend(fontsize=legend_fs, loc="best")

    if title is None:
        n = data["n_qubits"]
        L = data["n_layers"]
        side_note = {"all": "all ω", "nonneg": "ω≥0", "nonpos": "ω≤0"}[omega_side]
        band_note = f", |ω|≤{int(omega_phys)}" if omega_phys is not None else ""
        mode_note = {
            "raw": "raw",
            "norm": "normalised",
            "relmax": "relative-to-max",
            "logrel": "log10 relative-to-max",
        }[mode]
        title = f"Variance profiles ({mode_note}) | n={n}, L={L} | {side_note}{band_note}"

    fig.suptitle(title, fontsize=suptitle_fs)
    st.pyplot(fig)
    plt.close(fig)

def generate_circuit_figure(params):
    """Builds a static Matplotlib figure of the circuit."""
    spec = CircuitSpec(
        ansatz=str(params["ansatz"]),
        n_qubits=int(params["n_qubits"]),
        n_layers=int(params["n_layers"]),
        train_depth=int(params["train_depth"]),
        encoder_axis=str(params["encoder_axis"]),
        encoder_scale=float(params["encoder_scale"]),
        obs_kind=str(params["obs_kind"]),
        device_name="default.qubit",
        diff_method="parameter-shift",
        jit=False,
    )
    qnode, m = build_qnode(spec)
    
    # Dummy data to feed qml drawer
    dummy_x = 0.5
    dummy_params = np.linspace(0.0, 1.0, m, endpoint=False)
    
    qml.drawer.use_style('black_white')
    fig, ax = qml.draw_mpl(qnode)(dummy_x, dummy_params)
    
    return fig


def handle_change():
    if st.session_state.runnning:
        st.session_state.cmd_queue.put({"action": "STOP"})
        st.session_state.runnning = False



if __name__ == '__main__':
    st.set_page_config(page_title="Quantum Dashboard", layout="wide")

    # --- Initialize IPC ---
    if "cmd_queue" not in st.session_state:
        st.session_state.cmd_queue = mp.Queue(maxsize=5)
        st.session_state.data_queue = mp.Queue(maxsize=5) 
        
        # Spawn the daemon worker
        p = mp.Process(
            target=compute_worker, 
            args=(st.session_state.cmd_queue, st.session_state.data_queue),
            daemon=True 
        )
        p.start()
        st.session_state.worker_process = p

    # --- Sidebar Controls ---

    st.sidebar.title("**Circuit Parameters**")
    ansatz = st.sidebar.selectbox("Ansatz", ["YZY_ENTANGLING", "YZY", "CIRCUIT_19", "CIRCUIT_18",  "CIRCUIT_17" , "CIRCUIT_16", "CIRCUIT_15", "HEA"], on_change=handle_change)
    encoder_axis = st.sidebar.selectbox("Encoder Axis", ["RX", "RY", "RZ"], on_change=handle_change)
    obs_kind =  st.sidebar.selectbox("Observable", ["OX", "OY", "OZ"], on_change=handle_change)
    encoder_scale = st.sidebar.slider("Scaling factor on encoder angle", 0.0, 1.0, 1.0, 0.1, on_change=handle_change)
    n_qubits = st.sidebar.slider("Qubits (n)", 1, 8, 6, on_change=handle_change)
    train_depth = st.sidebar.slider("Train Depth (d)", 1, 10, 1, on_change=handle_change)
    n_layers = st.sidebar.slider("Layers (L)", 1, 10, 1, on_change=handle_change)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sampling**")
    n_theta_samples = st.sidebar.number_input("Total Theta Samples", value=100000, step=1000, on_change=handle_change)
    split_fraction_for_C = st.sidebar.slider("Fraction of samples assigned to C-split", 0.0, 1.0, 0.5, on_change=handle_change)
    batch_size = st.sidebar.number_input("Batch Size", value=256, on_change=handle_change)
    k_block = st.sidebar.number_input("Block size for K-chunking", value=5000, on_change=handle_change)
    max_omega = st.sidebar.slider("Max Data Freq (ω)", 1, 20, 6, on_change=handle_change)
    max_hw_for_K = st.sidebar.slider("Maximum Hamming weight of k-vectors in K", 1.0, 3.0, 1.0, on_change=handle_change)
    max_K_cap = st.sidebar.number_input("Hard K cap", value=30000, on_change=handle_change)
    n_x = st.sidebar.number_input("x-side Fourier n", value=256, on_change=handle_change)
    x_min = st.sidebar.slider("x-side Fourier min", 0.0, 2 * np.pi, 0.0, on_change=handle_change)
    x_max = st.sidebar.slider("x-side Fourier max", 0.0, 2 * np.pi, 2 * np.pi, on_change=handle_change)

    no_spectral = st.sidebar.checkbox("Disable spectral diagnostics (faster)", on_change=handle_change)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Plotting**")

    clip_q = st.sidebar.slider("Quantile for off-diagonal color scaling", 0.0, 1.0, 0.995,0.05) #
    omega_phys = None # Restrict to |omega|<=omega_phys.
    omega_side = st.sidebar.selectbox("Spectrum sides",["all", "nonneg", "nonpos"])# Keep only one side of spectrum (e.g. nonneg).
    mask_diag = st.sidebar.selectbox("Mask diagonal for display",["none", "zero", "nan"]) # Mask diagonal for display only.
    support_rel_var = None # Relative variance threshold for defining omega-support. (float)
    support_mode=st.sidebar.selectbox("How to combine C-side and MC-side support masks.", ["intersection", "union", "c", "mc"])
    mode=st.sidebar.selectbox("Variance profiles", ["relmax", "raw", "norm", "logrel"])
    ylim=None

    st.session_state.runnning = False
    st.session_state.latest_data = None



    # Push params to Worker
    if st.sidebar.button("🚀 Start / Restart"):
        params = {
                "ansatz": ansatz,
                "n_qubits": n_qubits,
                "n_layers" : n_layers,
                "train_depth" : train_depth,
                "encoder_axis" : encoder_axis,
                "obs_kind" : obs_kind,
                "encoder_scale" : encoder_scale,
                "batch_size": batch_size,
                "k_block" : k_block,
                "seed" : 42,
                'max_omega' : max_omega,
                'max_hw_for_K' : max_hw_for_K,
                'max_K_cap' : max_K_cap,
                'n_theta_samples' : n_theta_samples,
                'split_fraction_for_C' : split_fraction_for_C,
                'n_x' : n_x,
                'x_min' : x_min,
                'x_max' : x_max,
                "no_spectral" : no_spectral
            }
        st.session_state.cmd_queue.put({
            "action": "START",
            "params": params
            })
        
        with st.spinner("Drawing circuit diagram..."):
            st.session_state.circuit_fig = generate_circuit_figure(params)
        st.session_state.runnning = True
        
    if st.sidebar.button("🛑 Stop"):
        st.session_state.cmd_queue.put({"action": "STOP"})
        st.session_state.runnning = False
        st.session_state.data_queue.put(st.session_state.latest_data)

    st.sidebar.divider()
    st.sidebar.subheader("Simulation Status")
    status_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)

    # --- Dashboard Layout ---
    st.title("Circuit Harmonic Matrices Correlations")
    dashboard_placeholder = st.empty()

    if "circuit_fig" in st.session_state:
        with st.expander("Circuit Architecture", expanded=True):
            st.pyplot(st.session_state.circuit_fig, use_container_width=False)
            
    st.divider()

    

    # --- Read Data Loop ---
    while True:

        st.session_state.latest_data = st.session_state.data_queue.get()

        title_prefix = f"n={n_qubits}, L={n_layers}"
        
        with dashboard_placeholder.container():
            col1, col2 = st.columns(2)
            if st.session_state.latest_data:
                progress_frac = min(st.session_state.latest_data["current_batch_count"] / st.session_state.latest_data["total_batches"] , 1.0)
                progress_bar.progress(progress_frac)
                if st.session_state.runnning:
                    status_text.success(f"Running...")
                
                with col1:
                    plot_grid_figure_hermitian(
                        st.session_state.latest_data, st.session_state.latest_data["omega_grid"], title_prefix,
                        clip_q=clip_q, omega_phys=omega_phys,
                        omega_side=omega_side, mask_diag=mask_diag,
                        support_rel_var=support_rel_var, support_mode=support_mode
                    )
                    
                with col2:
                    plot_grid(
                        st.session_state.latest_data,
                        mode=mode,
                        omega_phys=omega_phys,
                        omega_side=omega_side,
                        show_textbox=True,
                        show_legend=True,
                        ylim=ylim
                    )
        plt.close('all')







