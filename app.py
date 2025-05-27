import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import seaborn as sns
import decoupler as dc

from pipeline_scripts.preprocessing import preprocess_adata
from pipeline_scripts.dimensionality_reduction import run_pca_umap_leiden
from pipeline_scripts.integration import run_scanorama_integration
from pipeline_scripts.deg_analysis import run_deg_analysis
from pipeline_scripts.annotation import create_marker_dict_from_df, run_decoupler_ora, run_decoupler_wmean

# --- Volcano Plot Function (Προσθήκη εδώ ή σε plotting.py αργότερα) ---
def plot_volcano(deg_results_df, lfc_threshold=1, padj_threshold=0.05, gene_name_col='names', lfc_col='logfoldchanges', padj_col='pvals_adj'):
    fig, ax = plt.subplots(figsize=(8, 6))
    if deg_results_df is None or deg_results_df.empty:
        st.warning("Δεν υπάρχουν δεδομένα DEG για το Volcano Plot.")
        ax.text(0.5, 0.5, "No DEG data available", horizontalalignment='center', verticalalignment='center')
        return fig
        
    deg_df_plot = deg_results_df.copy()
    
    # Έλεγχος και χειρισμός NaN/inf στο lfc_col
    if lfc_col in deg_df_plot.columns:
        deg_df_plot[lfc_col] = pd.to_numeric(deg_df_plot[lfc_col], errors='coerce')
        deg_df_plot[lfc_col] = np.nan_to_num(deg_df_plot[lfc_col], nan=0.0, 
                                           posinf=np.finfo(np.float32).max, 
                                           neginf=np.finfo(np.float32).min)
    else:
        st.error(f"Η στήλη '{lfc_col}' δεν βρέθηκε στα αποτελέσματα DEG για το Volcano Plot.")
        return fig

    # Έλεγχος και χειρισμός NaN/inf/zero στο padj_col
    if padj_col in deg_df_plot.columns:
        deg_df_plot[padj_col] = pd.to_numeric(deg_df_plot[padj_col], errors='coerce')
        deg_df_plot[padj_col] = deg_df_plot[padj_col].fillna(1.0)
        min_p_val = 1e-300 # Μικρότερη δυνατή τιμή για να αποφύγουμε log(0)
        # Εξασφάλιση ότι όλες οι τιμές είναι θετικές πριν το log10
        deg_df_plot[padj_col] = deg_df_plot[padj_col].apply(lambda x: x if x > min_p_val else min_p_val)
        deg_df_plot['-log10padj'] = -np.log10(deg_df_plot[padj_col])
    else:
        st.error(f"Η στήλη '{padj_col}' δεν βρέθηκε στα αποτελέσματα DEG για το Volcano Plot.")
        return fig

    deg_df_plot['significant'] = 'NS' 
    up_condition = (deg_df_plot[lfc_col] > lfc_threshold) & (deg_df_plot[padj_col] < padj_threshold)
    down_condition = (deg_df_plot[lfc_col] < -lfc_threshold) & (deg_df_plot[padj_col] < padj_threshold)
    deg_df_plot.loc[up_condition, 'significant'] = 'Up-regulated'
    deg_df_plot.loc[down_condition, 'significant'] = 'Down-regulated'
    
    colors = {"Up-regulated": "red", "Down-regulated": "blue", "NS": "grey"}
    
    sns.scatterplot(data=deg_df_plot, x=lfc_col, y='-log10padj', 
                    hue='significant', palette=colors, ax=ax, s=20, edgecolor=None, hue_order=['Up-regulated', 'Down-regulated', 'NS'])
    
    ax.axhline(-np.log10(padj_threshold), ls='--', color='gray', lw=0.8)
    ax.axvline(lfc_threshold, ls='--', color='gray', lw=0.8)
    ax.axvline(-lfc_threshold, ls='--', color='gray', lw=0.8)
    
    ax.set_xlabel(f"Log2 Fold Change ({lfc_col})")
    ax.set_ylabel("-log10 Adjusted P-value")
    ax.set_title("Volcano Plot των DEGs")
    # ax.legend(title='Significance', loc='upper right') # Το legend μπορεί να καλύπτει σημεία
    # Προσαρμογή του legend για να μην είναι πολύ μεγάλο
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels: # Εμφάνιση legend μόνο αν υπάρχουν δεδομένα
        ax.legend(handles=handles, labels=labels, title='Significance', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

st.set_page_config(layout="wide", page_title="Εφαρμογή Ανάλυσης scRNA-seq")

def init_session_keys():
    keys_defaults = {
        'adata_initial_list': [],
        'adata_merged_for_integration_or_qc': None,
        'adata_integrated': None,
        'adata_for_qc': None,
        'adata_processed': None,
        'adata_clustered': None,
        'deg_results': None,
        'adata_annotated': None,
        'marker_dict': None,
        'raw_marker_df': None,
        'marker_df_columns': [],
        'current_stage': "load_data",
        'debug_counter': 0,
        'uploader_key_suffix': 0,
        'trigger_file_processing': False,
        'qc_adata_before_processing': None,
        'param_deg_n_genes_display_cache': 50,
        'param_lfc_thresh_volcano_cache': 1.0,
        'param_padj_thresh_volcano_cache': 0.05
    }
    for key, default_value in keys_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_keys()
st.session_state.debug_counter += 1
st.sidebar.write(f"Rerun: {st.session_state.debug_counter}, Stage: {st.session_state.current_stage}")

st.sidebar.title("Επιλογές Pipeline")

# Create tabs
tab1, tab2 = st.tabs(["🔬 Ανάλυση Δεδομένων", "👥 Πληροφορίες Ομάδας"])

with tab1:
    st.title("Εφαρμογή Ανάλυσης Δεδομένων scRNA-seq")

    st.header("1. Φόρτωση Δεδομένων")
    current_uploader_key = f"uploader_widget_state_{st.session_state.uploader_key_suffix}"
    uploaded_files_val = st.file_uploader(
        "Ανεβάστε ένα ή περισσότερα αρχεία AnnData (.h5ad)",
        type="h5ad",
        accept_multiple_files=True,
        key=current_uploader_key
    )

    # --- Integration Section in Sidebar ---
    with st.sidebar.expander("1.5 Ενοποίηση (Scanorama)", expanded=True):
        can_display_scanorama_options = st.session_state.get('adata_merged_for_integration_or_qc') is not None
        num_batches = 0
        if can_display_scanorama_options and 'batch_upload' in st.session_state.adata_merged_for_integration_or_qc.obs:
            num_batches = st.session_state.adata_merged_for_integration_or_qc.obs['batch_upload'].nunique()
        
        scanorama_possible = can_display_scanorama_options and (num_batches > 1)
        
        run_scanorama_checkbox = st.checkbox(
            "Ενεργοποίηση Scanorama", 
            value=False, 
            disabled=not scanorama_possible, 
            key="run_scanorama_cb"
        )
        param_scanorama_dimred = st.number_input(
            "Scanorama Dimensions", 
            10, 100, 50, 10, 
            disabled=not (scanorama_possible and run_scanorama_checkbox), 
            key="scanorama_dimred"
        )
        # Οποιοδήποτε κουμπί εκτέλεσης Scanorama θα πρέπει επίσης να είναι μέσα στο expander
        if run_scanorama_checkbox and scanorama_possible:
            if st.button("Εφαρμογή Scanorama (πριν το QC)", key="apply_scanorama_btn_inside_expander"):
                if st.session_state.get('adata_merged_for_integration_or_qc') is not None:
                    with st.spinner("Εκτελείται Scanorama..."):
                        try:
                            adata_for_scanorama = st.session_state.adata_merged_for_integration_or_qc.copy()
                            st.session_state.adata_integrated = run_scanorama_integration(
                                adata_for_scanorama, 
                                batch_key='batch_upload', 
                                dimred=param_scanorama_dimred
                            )
                            st.session_state.adata_for_qc = st.session_state.adata_integrated
                            st.success(f"Scanorama ολοκληρώθηκε. Σχήμα: {st.session_state.adata_for_qc.shape}")
                            st.session_state.current_stage = "preprocessing_params"
                        except Exception as e:
                            st.error(f"Σφάλμα Scanorama: {e}")
                            st.session_state.adata_for_qc = st.session_state.adata_merged_for_integration_or_qc 
                    st.rerun()

    if uploaded_files_val:
        if st.button("Επεξεργασία & Φόρτωση Επιλεγμένων Αρχείων", key="process_button"):
            st.session_state.trigger_file_processing = True
            st.session_state.uploader_key_suffix += 1 # Invalidate previous uploader state

    if st.session_state.get('trigger_file_processing', False):
        # Reset relevant parts of session state
        for ad_key in ['adata_initial_list', 'adata_merged_for_integration_or_qc', 
                       'adata_integrated', 'adata_for_qc', 'adata_processed', 
                       'adata_clustered', 'deg_results', 'qc_adata_before_processing',
                       'adata_annotated', 'marker_dict', 'raw_marker_df', 'marker_df_columns']:
            st.session_state[ad_key] = [] if ad_key == 'adata_initial_list' else ( [] if ad_key == 'marker_df_columns' else None)

        st.session_state.current_stage = "loading_data"
        files_to_process = st.session_state.get(current_uploader_key, []) # Get files from the CURRENT uploader
        
        adata_list_temp = []
        success_all_files = True
        if not files_to_process:
            success_all_files = False
            # st.warning("Δεν επιλέχθηκαν αρχεία.") # Optional: message if no files are selected but button is pressed

        for uploaded_file_item in files_to_process:
            try:
                adata_bytes = uploaded_file_item.getvalue()
                adata_item = sc.read_h5ad(io.BytesIO(adata_bytes))
                adata_list_temp.append(adata_item)
            except Exception as e:
                st.error(f"Σφάλμα ανάγνωσης {uploaded_file_item.name}: {e}")
                success_all_files = False
                break
        
        if success_all_files and adata_list_temp:
            st.session_state.adata_initial_list = adata_list_temp
            if len(st.session_state.adata_initial_list) > 1:
                try:
                    for i, ad_item in enumerate(st.session_state.adata_initial_list):
                        ad_item.obs['batch_upload'] = str(i) 
                    
                    st.session_state.adata_merged_for_integration_or_qc = sc.AnnData.concatenate(
                        *st.session_state.adata_initial_list, 
                        join='outer', 
                        batch_key='batch_original_upload', 
                        index_unique=None 
                    )
                    # Ensure obs_names are unique after concatenation
                    try:
                        st.session_state.adata_merged_for_integration_or_qc.obs_names_make_unique()
                        st.write("Observation names made unique after multi-file merge.") # Debug message
                    except Exception as e:
                        st.warning(f"Could not make observation names unique after multi-file merge: {e}")

                    if 'batch_upload' in st.session_state.adata_merged_for_integration_or_qc.obs:
                         st.session_state.adata_merged_for_integration_or_qc.obs['batch_upload'] = st.session_state.adata_merged_for_integration_or_qc.obs['batch_upload'].astype('category')
                    
                    st.success(f"Συνένωση {len(st.session_state.adata_initial_list)} αρχείων. Σχήμα: {st.session_state.adata_merged_for_integration_or_qc.shape}")
                except Exception as e:
                    st.error(f"Σφάλμα συνένωσης: {e}")
                    st.session_state.adata_merged_for_integration_or_qc = None # Ensure it's None on error
            elif len(st.session_state.adata_initial_list) == 1:
                st.session_state.adata_merged_for_integration_or_qc = st.session_state.adata_initial_list[0].copy()
                # Ensure obs_names are unique for single file as well
                try:
                    st.session_state.adata_merged_for_integration_or_qc.obs_names_make_unique()
                    st.write("Observation names made unique for single file.") # Debug message
                except Exception as e:
                    st.warning(f"Could not make observation names unique for single file: {e}")

                if 'batch_upload' not in st.session_state.adata_merged_for_integration_or_qc.obs:
                    st.session_state.adata_merged_for_integration_or_qc.obs['batch_upload'] = '0'
                    st.session_state.adata_merged_for_integration_or_qc.obs['batch_upload'] = st.session_state.adata_merged_for_integration_or_qc.obs['batch_upload'].astype('category')
                st.success(f"Φόρτωση αρχείου. Σχήμα: {st.session_state.adata_merged_for_integration_or_qc.shape}")

            if st.session_state.adata_merged_for_integration_or_qc is not None:
                # Ensure observation names are unique AFTER merging/loading
                # This might be redundant now but kept for safety, can be removed if above works.
                try:
                    st.session_state.adata_merged_for_integration_or_qc.obs_names_make_unique()
                    st.write("Observation names made unique.") # Temporary message for debugging
                except Exception as e:
                    st.warning(f"Could not make observation names unique: {e}")
                st.session_state.current_stage = "integration_params" 
        
        st.session_state.trigger_file_processing = False 
        st.rerun() 

    # Logic for Scanorama application or skipping to QC
    if st.session_state.get('adata_merged_for_integration_or_qc') is not None:
        # This button was previously outside the expander, and its logic is now duplicated inside.
        # We need to ensure that the logic tied to the run_scanorama_checkbox and scanorama_possible 
        # correctly uses the adata_for_qc and adata_integrated states, 
        # regardless of whether the button inside the expander was pressed or if it was handled by previous logic.
        # The original button outside the expander will be removed or its logic consolidated.

        # The core logic for setting adata_for_qc based on Scanorama choice should remain.
        # However, the action of *running* Scanorama is now tied to the button inside the expander.
        
        # If Scanorama is NOT selected, or not possible, or already run (adata_integrated exists)
        # This logic ensures adata_for_qc is set correctly
        if not (run_scanorama_checkbox and scanorama_possible):
            # If scanorama was not chosen or not possible, qc data is the merged one.
            # Also, if adata_integrated exists (scanorama was run) BUT then the user unchecks scanorama, 
            # we should revert to merged data for QC if no new scanorama run is triggered.
            if st.session_state.get('adata_integrated') is not None and not run_scanorama_checkbox:
                 st.session_state.adata_for_qc = st.session_state.adata_merged_for_integration_or_qc
                 st.session_state.adata_integrated = None # Clear integrated if deselected
                 st.info("Το Scanorama απενεργοποιήθηκε. Τα δεδομένα για QC είναι τα αρχικά συνενωμένα.")
            elif st.session_state.get('adata_integrated') is None: # No scanorama run yet, or it failed previously
                st.session_state.adata_for_qc = st.session_state.adata_merged_for_integration_or_qc

        elif run_scanorama_checkbox and scanorama_possible and st.session_state.get('adata_integrated') is not None:
            # If Scanorama was run and successful, and checkbox is still checked
            st.session_state.adata_for_qc = st.session_state.adata_integrated
        elif st.session_state.get('adata_merged_for_integration_or_qc') is not None and st.session_state.get('adata_for_qc') is None:
            # Fallback if previous logic didn't set adata_for_qc (e.g. after deselection of Scanorama without rerun)
            # or if it's the first time loading data and scanorama is possible but not run yet.
            st.session_state.adata_for_qc = st.session_state.adata_merged_for_integration_or_qc


    # Display data preview if data is ready for QC
    if st.session_state.get('adata_for_qc') is not None:
        with st.expander("Προεπισκόπηση Δεδομένων για QC", expanded=True):
            adata_display = st.session_state.adata_for_qc
            st.text(str(adata_display))
            st.dataframe(adata_display.obs.head())
            st.dataframe(adata_display.var.head())
            if 'X_scanorama' in adata_display.obsm:
                st.write("Εφαρμόστηκε Scanorama.")
            elif 'batch_original_upload' in adata_display.obs and adata_display.obs['batch_original_upload'].nunique() > 1:
                 st.write("Τα δεδομένα είναι συνενωμένα (χωρίς Scanorama).")
    elif not st.session_state.get(current_uploader_key, []): # If no files are uploaded yet (or after clearing)
        st.info("Παρακαλώ ανεβάστε αρχεία και πατήστε 'Επεξεργασία & Φόρτωση'.")


    # --- QC Section ---
    with st.sidebar.expander("2. Προεπεξεργασία (QC)", expanded=False):
        qc_ready = st.session_state.get('adata_for_qc') is not None
        qc_enabled = st.checkbox("Ενεργοποίηση Προεπεξεργασίας", value=True, disabled=not qc_ready, key="qc_enable_cb_expander")

        param_min_counts_gene = st.number_input("Ελάχ. counts/γονίδιο", 0, 100, 3, 1, disabled=not (qc_ready and qc_enabled), key="min_counts_gene_qc_expander") 
        param_min_n_genes_cell = st.number_input("Ελάχ. γονίδια/κύτταρο", 0, 1000, 200, 10, disabled=not (qc_ready and qc_enabled), key="min_n_genes_cell_qc_expander") 
        param_max_n_genes_cell = st.number_input("Μέγ. γονίδια/κύτταρο", 0, 20000, 7000, 100, disabled=not (qc_ready and qc_enabled), key="max_n_genes_cell_qc_expander") 
        param_pct_mito_threshold = st.slider("Μέγ. % μιτοχ.", 0.0, 100.0, 10.0, 0.5, disabled=not (qc_ready and qc_enabled), key="pct_mito_qc_expander") 
        param_target_sum = st.number_input("Target sum (norm.)", 100, 50000, 10000, 100, disabled=not (qc_ready and qc_enabled), key="target_sum_qc_expander") 
        param_n_top_genes = st.number_input("HVGs (n_top_genes)", 100, 5000, 2000, 100, disabled=not (qc_ready and qc_enabled), key="n_top_genes_qc_expander") 

        if st.button("Εκτέλεση Προεπεξεργασίας", key="run_qc_button_expander", disabled=not (qc_ready and qc_enabled)):
            if st.session_state.get('adata_for_qc') is not None:
                st.session_state.current_stage = "preprocessing_running"
                st.session_state.adata_processed = None
                st.session_state.adata_clustered = None
                st.session_state.deg_results = None
                st.session_state.adata_annotated = None
                
                st.session_state.qc_adata_before_processing = st.session_state.adata_for_qc.copy()
                
                with st.spinner("Εκτελείται η προεπεξεργασία..."):
                    try:
                        # Use the parameters from the expander widgets
                        st.session_state.adata_processed = preprocess_adata(
                            st.session_state.qc_adata_before_processing, 
                            min_counts_gene=param_min_counts_gene,
                            min_n_genes_cell=param_min_n_genes_cell,
                            max_n_genes_cell=param_max_n_genes_cell,
                            pct_mito_threshold=param_pct_mito_threshold,
                            target_sum=param_target_sum,
                            n_top_genes=param_n_top_genes,
                            debug=False 
                        )
                        st.session_state.current_stage = "dim_reduction_params"
                    except Exception as e:
                        st.error(f"Σφάλμα προεπεξεργασίας: {e}")
                        st.session_state.adata_processed = None 
                        st.session_state.current_stage = "preprocessing_params"
                st.rerun()

    # Display QC results if processing is done and successful
    if st.session_state.current_stage in ["dim_reduction_params", "deg_analysis_params", "annotation_params", "annotation_results_display", "deg_results_display"] and \
       st.session_state.get('qc_adata_before_processing') and st.session_state.get('adata_processed'):
        with st.container(): 
            st.header("2. Αποτελέσματα Προεπεξεργασίας")
            col_qc_before, col_qc_after = st.columns(2)
            with col_qc_before:
                st.subheader("QC Πριν")
                adata_to_plot_before = st.session_state.qc_adata_before_processing
                # Ensure QC metrics are calculated if not present
                if 'n_genes_by_counts' not in adata_to_plot_before.obs.columns or \
                   'total_counts' not in adata_to_plot_before.obs.columns or \
                   'pct_counts_mt' not in adata_to_plot_before.obs.columns:
                    adata_to_plot_before.var['mt'] = adata_to_plot_before.var_names.str.startswith(('MT-', 'mt-'))
                    sc.pp.calculate_qc_metrics(adata_to_plot_before, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
                
                fig_before, axs_before = plt.subplots(1, 3, figsize=(15, 4))
                sc.pl.violin(adata_to_plot_before, ['n_genes_by_counts'], jitter=0.4, ax=axs_before[0], show=False)
                sc.pl.violin(adata_to_plot_before, ['total_counts'], jitter=0.4, ax=axs_before[1], show=False)
                sc.pl.violin(adata_to_plot_before, ['pct_counts_mt'], jitter=0.4, ax=axs_before[2], show=False)
                plt.tight_layout()
                st.pyplot(fig_before)
                plt.close(fig_before) # Close figure
            
            with col_qc_after:
                st.subheader("QC Μετά")
                # QC metrics should be present in adata_processed from the function
                fig_after, axs_after = plt.subplots(1, 3, figsize=(15, 4))
                sc.pl.violin(st.session_state.adata_processed, ['n_genes_by_counts'], jitter=0.4, ax=axs_after[0], show=False)
                sc.pl.violin(st.session_state.adata_processed, ['total_counts'], jitter=0.4, ax=axs_after[1], show=False)
                sc.pl.violin(st.session_state.adata_processed, ['pct_counts_mt'], jitter=0.4, ax=axs_after[2], show=False)
                plt.tight_layout()
                st.pyplot(fig_after)
                plt.close(fig_after) # Close figure

            st.success("Η προεπεξεργασία ολοκληρώθηκε!")
            st.write(f"Σχήμα μετά QC: {st.session_state.adata_processed.shape}")


    # --- PCA, UMAP & Clustering Section ---
    with st.sidebar.expander("3. PCA, UMAP & Clustering", expanded=False):
        dimred_ready = st.session_state.get('adata_processed') is not None
        dimred_enabled = st.checkbox("Ενεργοποίηση PCA/UMAP/Clustering", value=True, disabled=not dimred_ready, key="dimred_enable_cb_expander")

        param_n_pcs = st.number_input("Αριθμός PCs", 5, 100, 30, 5, disabled=not (dimred_ready and dimred_enabled), key="n_pcs_dimred_expander")
        param_n_neighbors = st.number_input("Αριθμός Γειτόνων", 2, 50, 15, 1, disabled=not (dimred_ready and dimred_enabled), key="n_neighbors_dimred_expander") 
        param_umap_min_dist = st.slider("UMAP Min Distance", 0.0, 1.0, 0.5, 0.05, disabled=not (dimred_ready and dimred_enabled), key="min_dist_dimred_expander")
        param_leiden_resolution = st.slider("Leiden Resolution", 0.1, 2.0, 1.0, 0.1, disabled=not (dimred_ready and dimred_enabled), key="leiden_res_dimred_expander")
        param_use_hvg_pca = st.checkbox("Χρήση HVGs για PCA", value=True, disabled=not (dimred_ready and dimred_enabled), key="use_hvg_dimred_expander")

        run_dimred_button_pressed_expander = st.button("Εκτέλεση PCA, UMAP & Clustering", key="run_dimred_button_expander", disabled=not (dimred_ready and dimred_enabled))

    # Display UMAP plot if clustering is done
    if st.session_state.get('adata_clustered') is not None and \
       st.session_state.current_stage in ["deg_analysis_params", "annotation_params", "annotation_results_display", "deg_results_display"]:
        with st.container(): 
            st.header("3. Αποτελέσματα Μείωσης Διαστατικότητας & Clustering")
            if 'leiden' in st.session_state.adata_clustered.obs:
                st.write(f"Αποθηκεύτηκαν {len(st.session_state.adata_clustered.obs['leiden'].unique())} clusters στη στήλη 'leiden'.")
                
                genes_to_plot_str = st.text_input(
                    "Εμφάνιση έκφρασης γονιδίων στο UMAP (π.χ., CD74,LYZ) ή αφήστε κενό για χρωματισμό βάσει .obs:", 
                    key="umap_gene_input"
                )
                genes_to_plot_list = [gene.strip().upper() for gene in genes_to_plot_str.split(',') if gene.strip()]
                
                umap_color_options = ['leiden']
                if 'batch_upload' in st.session_state.adata_clustered.obs:
                    umap_color_options.append('batch_upload')
                if 'condition' in st.session_state.adata_clustered.obs: 
                    umap_color_options.append('condition')
                
                if st.session_state.get('adata_annotated') is not None:
                    wms_cols = [col for col in st.session_state.adata_annotated.obs.columns if col.startswith('wms_')]
                    umap_color_options.extend(wms_cols)
                
                selected_color_obs = None
                if not genes_to_plot_list:
                    selected_color_obs = st.selectbox(
                        "Χρωματισμός UMAP βάσει (.obs):", 
                        umap_color_options, 
                        index=umap_color_options.index('leiden') if 'leiden' in umap_color_options else 0, 
                        key="umap_color_select"
                    )
                
                color_param = None
                title_suffix = ""
                use_raw_for_gene_expr = False # Default to False (use .X which is lognorm data)

                if genes_to_plot_list:
                    # Check if genes exist (case-insensitive check against var_names)
                    adata_var_upper = [v.upper() for v in st.session_state.adata_clustered.var_names]
                    valid_genes_upper = [g for g in genes_to_plot_list if g in adata_var_upper]
                    invalid_genes = [g for g in genes_to_plot_list if g not in adata_var_upper]

                    if invalid_genes:
                        st.warning(f"Τα γονίδια: {', '.join(invalid_genes)} δεν βρέθηκαν στα δεδομένα (έγινε αναζήτηση με κεφαλαία).")

                    if not valid_genes_upper:
                        st.error("Κανένα από τα γονίδια που δώσατε δεν βρέθηκε. Εμφανίζεται ο χρωματισμός Leiden.")
                        color_param = 'leiden' # Fallback
                        title_suffix = "Leiden Clusters (fallback)"
                    else:
                        # Get actual gene names (with correct casing) from var_names
                        actual_gene_names_to_plot = []
                        for gene_upper in valid_genes_upper:
                            try:
                                idx = adata_var_upper.index(gene_upper)
                                actual_gene_names_to_plot.append(st.session_state.adata_clustered.var_names[idx])
                            except ValueError:
                                pass # Should not happen if valid_genes_upper is correct
                        
                        color_param = actual_gene_names_to_plot
                        title_suffix = f"Expression of {', '.join(color_param)}"
                        # For gene expression, Scanpy typically uses .raw if available and use_raw=True.
                        # If we want to plot from .X (lognormalized, scaled), use_raw should be False.
                        # The pipeline stores lognormalized in .X and HVG scaled in .X of .raw
                        # For simple gene expression visualization, .X (lognorm) is often preferred.
                        use_raw_for_gene_expr = False 
                                                
                elif selected_color_obs:
                    color_param = selected_color_obs
                    title_suffix = selected_color_obs
                
                if color_param:
                    st.subheader(f"UMAP χρωματισμένο κατά {title_suffix}")
                    if isinstance(color_param, list) and len(color_param) > 1: # Multiple genes
                        num_gene_plots = len(color_param)
                        cols_for_genes = st.columns(min(num_gene_plots, 3)) 
                        for i, gene_to_plot_actual_case in enumerate(color_param):
                            with cols_for_genes[i % 3]:
                                fig_gene_umap, ax_gene_umap = plt.subplots(figsize=(6, 5))
                                try:
                                    sc.pl.umap(
                                        st.session_state.adata_clustered, 
                                        color=gene_to_plot_actual_case, 
                                        ax=ax_gene_umap, 
                                        show=False, 
                                        legend_loc='on data', 
                                        title=f'UMAP by {gene_to_plot_actual_case}',
                                        use_raw=use_raw_for_gene_expr 
                                    )
                                    st.pyplot(fig_gene_umap)
                                    plt.close(fig_gene_umap)
                                except Exception as e:
                                    st.error(f"Σφάλμα UMAP για {gene_to_plot_actual_case}: {e}")
                    else: # Single gene or .obs column
                        fig_umap, ax_umap = plt.subplots(figsize=(7, 6))
                        try:
                            sc.pl.umap(
                                st.session_state.adata_clustered, 
                                color=color_param, 
                                ax=ax_umap, 
                                show=False, 
                                legend_loc='on data', 
                                title=f'UMAP by {title_suffix}',
                                use_raw=use_raw_for_gene_expr if isinstance(color_param, list) else False # use_raw only if it's a gene list
                            )
                            st.pyplot(fig_umap)
                            plt.close(fig_umap)
                        except Exception as e:
                            st.error(f"Σφάλμα UMAP για {title_suffix}: {e}")
            else:
                st.warning("Δεν βρέθηκαν Leiden clusters στο AnnData object.")

    # IMPORTANT: This `run_dimred_button_pressed` was tied to the old button outside the expander.
    # We need to use `run_dimred_button_pressed_expander` now for the logic.
    if run_dimred_button_pressed_expander: # Changed from run_dimred_button_pressed
        if st.session_state.get('adata_processed') is not None:
            st.session_state.current_stage = "dim_reduction_running"
            st.session_state.adata_clustered = None 
            with st.spinner("Εκτελείται PCA, UMAP και Clustering..."):
                adata_dimred_input = st.session_state.adata_processed.copy() 
                try:
                    # Use parameters from the expander's widgets
                    st.session_state.adata_clustered = run_pca_umap_leiden(
                        adata_dimred_input, 
                        n_pcs=param_n_pcs, # This should be the one from the expander
                        n_neighbors=param_n_neighbors, # This should be the one from the expander
                        umap_min_dist=param_umap_min_dist, # This should be the one from the expander
                        leiden_resolution=param_leiden_resolution, # This should be the one from the expander
                        use_highly_variable=param_use_hvg_pca # This should be the one from the expander
                    )
                    st.success("PCA, UMAP και Leiden clustering ολοκληρώθηκαν!")
                    st.session_state.current_stage = "deg_analysis_params" 
                except Exception as e:
                    st.error(f"Σφάλμα κατά την εκτέλεση PCA/UMAP/Clustering: {e}")
                    st.session_state.adata_clustered = None 
                    st.session_state.current_stage = "dim_reduction_params" 
                st.rerun()

    # --- DEG Analysis Section ---
    with st.sidebar.expander("4. Ανάλυση DEG", expanded=False):
        deg_ready_expander = st.session_state.get('adata_clustered') is not None # Use new var name to avoid conflict if old one is in scope
        deg_enabled_expander = st.checkbox("Ενεργοποίηση Ανάλυσης DEG", value=True, disabled=not deg_ready_expander, key="deg_enable_cb_expander")
        
        param_deg_groupby_expander = None
        param_deg_group1_expander = None
        param_deg_reference_expander = None

        if deg_ready_expander and deg_enabled_expander:
            adata_c_expander = st.session_state.adata_clustered
            groupby_options_expander = [col for col in adata_c_expander.obs.columns 
                               if adata_c_expander.obs[col].dtype.name in ['category', 'object', 'bool'] or 
                                  (pd.api.types.is_numeric_dtype(adata_c_expander.obs[col]) and adata_c_expander.obs[col].nunique() < 20 and adata_c_expander.obs[col].nunique() > 1) ]
            
            if groupby_options_expander:
                param_deg_groupby_expander = st.selectbox(
                    "Ομαδοποίηση DEG βάσει:", 
                    groupby_options_expander, 
                    index=groupby_options_expander.index('leiden') if 'leiden' in groupby_options_expander else 0, 
                    key="deg_groupby_expander"
                )
                if param_deg_groupby_expander:
                    try:
                        available_groups_expander = sorted(list(adata_c_expander.obs[param_deg_groupby_expander].astype(str).unique()))
                        available_groups_expander = [g for g in available_groups_expander if g.lower() != 'nan']
                    except Exception: 
                        available_groups_expander = sorted(list(adata_c_expander.obs[param_deg_groupby_expander].unique()))

                    param_deg_group1_expander = st.multiselect(
                        f"Ομάδα/ες 1 (από '{param_deg_groupby_expander}'):", 
                        available_groups_expander, 
                        default=[available_groups_expander[0]] if available_groups_expander else None, 
                        key="deg_group1_expander"
                    )
                    
                    reference_options_expander = ["rest"] + [g for g in available_groups_expander if g not in param_deg_group1_expander]
                    param_deg_reference_expander = st.selectbox(
                        "Ομάδα Αναφοράς DEG:", 
                        reference_options_expander, 
                        key="deg_reference_expander"
                    )
            else:
                st.warning("Δεν βρέθηκαν κατάλληλες στήλες για ομαδοποίηση DEG.")

            param_deg_method_expander = st.selectbox("Μέθοδος DEG:", ['wilcoxon', 't-test', 'logreg'], key="deg_method_expander")
            param_deg_n_genes_display_expander = st.number_input("Top N Γονίδια (Πίνακας):", 10, 500, 50, 10, key="deg_n_genes_display_expander")
            param_lfc_thresh_volcano_expander = st.slider("Volcano LFC Κατώφλι", 0.0, 5.0, 1.0, 0.1, key="volcano_lfc_expander")
            param_padj_thresh_volcano_expander = st.slider("Volcano Adj. P-val Κατώφλι", 0.001, 0.5, 0.05, 0.001, format="%.3f", key="volcano_padj_expander")

        run_deg_button_pressed_expander = st.button(
            "Εκτέλεση Ανάλυσης DEG", 
            key="run_deg_button_expander", 
            disabled=not (deg_ready_expander and deg_enabled_expander and param_deg_groupby_expander and param_deg_group1_expander and param_deg_reference_expander)
        )

    # Display DEG Results
    if st.session_state.get('deg_results') is not None and st.session_state.current_stage == "deg_results_display":
        with st.container():
            st.header("4. Αποτελέσματα Ανάλυσης DEG")
            st.subheader(f"Top {st.session_state.get('param_deg_n_genes_display_cache', param_deg_n_genes_display_expander)} Διαφορικά Εκφρασμένα Γονίδια") 
            st.dataframe(st.session_state.deg_results.head(st.session_state.get('param_deg_n_genes_display_cache', param_deg_n_genes_display_expander)))
            
            st.subheader("Volcano Plot")
            try:
                fig_volcano = plot_volcano(
                    st.session_state.deg_results, 
                    lfc_threshold=st.session_state.get('param_lfc_thresh_volcano_cache', param_lfc_thresh_volcano_expander), 
                    padj_threshold=st.session_state.get('param_padj_thresh_volcano_cache', param_padj_thresh_volcano_expander)
                )
                st.pyplot(fig_volcano)
                plt.close(fig_volcano) 
            except Exception as e:
                st.error(f"Σφάλμα Δημιουργίας Volcano Plot: {e}")

    if run_deg_button_pressed_expander: # Changed from run_deg_button_pressed
        if st.session_state.get('adata_clustered') is not None and param_deg_groupby_expander and param_deg_group1_expander:
            st.session_state.current_stage = "deg_running"
            st.session_state.deg_results = None 
            
            st.session_state.param_deg_n_genes_display_cache = param_deg_n_genes_display_expander
            st.session_state.param_lfc_thresh_volcano_cache = param_lfc_thresh_volcano_expander
            st.session_state.param_padj_thresh_volcano_cache = param_padj_thresh_volcano_expander

            with st.spinner("Εκτελείται ανάλυση DEG..."):
                group1_list_expander = list(param_deg_group1_expander) 
                ref_group_for_func_expander = param_deg_reference_expander
                if isinstance(ref_group_for_func_expander, str) and ref_group_for_func_expander != "rest":
                    ref_group_for_func_expander = [ref_group_for_func_expander]
                elif isinstance(ref_group_for_func_expander, list) and not ref_group_for_func_expander: 
                     ref_group_for_func_expander = "rest"

                try:
                    adata_for_deg_copy, deg_df = run_deg_analysis(
                        st.session_state.adata_clustered, 
                        groupby_key=param_deg_groupby_expander, 
                        group1=group1_list_expander,
                        reference_groups=ref_group_for_func_expander, 
                        method=param_deg_method_expander,
                        n_genes=st.session_state.adata_clustered.n_vars 
                    )
                    if deg_df is not None and not deg_df.empty:
                        st.session_state.deg_results = deg_df
                        st.success(f"DEG ολοκληρώθηκε: {param_deg_groupby_expander} {group1_list_expander} vs {param_deg_reference_expander}.")
                        st.session_state.current_stage = "deg_results_display"
                    elif deg_df is not None and deg_df.empty:
                        st.warning("Η ανάλυση DEG δεν επέστρεψε γονίδια. Ελέγξτε τις παραμέτρους και τις ομάδες.")
                        st.session_state.current_stage = "deg_analysis_params"
                    else: 
                        st.error("Η ανάλυση DEG δεν επέστρεψε DataFrame.")
                        st.session_state.current_stage = "deg_analysis_params"
                except Exception as e:
                    st.error(f"Σφάλμα DEG: {e}")
                    st.session_state.current_stage = "deg_analysis_params"
                st.rerun()

    # --- Cell Type Annotation (Decoupler) Section ---
    with st.sidebar.expander("5. Χαρ/σμός Κυτ. Τύπων (Decoupler)", expanded=False):
        annotation_ready_expander = st.session_state.get('adata_clustered') is not None 
        annotation_enabled_expander = st.checkbox("Ενεργοποίηση Χαρ/σμού Κυτ. Τύπων", value=True, disabled=not annotation_ready_expander, key="anno_enable_cb_expander")
        
        uploaded_marker_file_expander = None # Initialize here
        if annotation_ready_expander and annotation_enabled_expander:
            uploaded_marker_file_expander = st.file_uploader("Ανεβάστε αρχείο δεικτών (.csv)", type="csv", key="marker_file_uploader_expander")
            
            if uploaded_marker_file_expander is not None:
                if st.button("Επεξεργασία Αρχείου Δεικτών", key="process_marker_file_btn_expander"):
                    try:
                        df_markers_expander = pd.read_csv(uploaded_marker_file_expander)
                        st.session_state.marker_df_columns = list(df_markers_expander.columns) 
                        st.session_state.raw_marker_df = df_markers_expander 
                        st.success(f"Το αρχείο δεικτών '{uploaded_marker_file_expander.name}' φορτώθηκε.")
                        st.session_state.marker_dict = None 
                    except Exception as e:
                        st.error(f"Σφάλμα ανάγνωσης αρχείου δεικτών: {e}")
                        st.session_state.raw_marker_df = None
                        st.session_state.marker_df_columns = []
            
            if st.session_state.get('raw_marker_df') is not None and not st.session_state.raw_marker_df.empty:
                ct_col_options_expander = st.session_state.get('marker_df_columns', [])
                param_anno_cell_type_col_expander = st.selectbox(
                    "Στήλη Τύπου Κυττάρου:", ct_col_options_expander, 
                    index=0 if ct_col_options_expander else -1, 
                    key="anno_ct_col_expander"
                )
                param_anno_gene_col_expander = st.selectbox(
                    "Στήλη Συμβόλου Γονιδίου:", ct_col_options_expander, 
                    index=1 if len(ct_col_options_expander)>1 else (0 if ct_col_options_expander else -1), 
                    key="anno_gene_col_expander"
                )

                if param_anno_cell_type_col_expander and param_anno_gene_col_expander: 
                    if st.button("Δημιουργία Λεξικού Δεικτών", key="create_marker_dict_btn_expander"):
                        with st.spinner("Δημιουργία λεξικού δεικτών..."):
                            try:
                                st.session_state.marker_dict = create_marker_dict_from_df(
                                    st.session_state.raw_marker_df, 
                                    cell_type_col=param_anno_cell_type_col_expander, 
                                    gene_symbol_col=param_anno_gene_col_expander
                                )
                                st.success(f"Λεξικό δεικτών δημιουργήθηκε: {len(st.session_state.marker_dict)} τύποι κυττάρων.")
                            except Exception as e:
                                st.error(f"Σφάλμα δημιουργίας λεξικού: {e}")
                                st.session_state.marker_dict = None
            
            if st.session_state.get('marker_dict') is not None:
                param_anno_min_n_expander = st.number_input("Decoupler min_n:", 1, 50, 5, 1, key="anno_min_n_expander")
                run_ora_cb_expander = st.checkbox("Εκτέλεση ORA", value=True, key="anno_run_ora_expander")
                run_wms_cb_expander = st.checkbox("Εκτέλεση WMS (Weighted Mean Score)", value=True, key="anno_run_wms_expander")

                if st.button("Εκτέλεση Ανάλυσης Decoupler", key="run_decoupler_btn_expander", disabled=not (run_ora_cb_expander or run_wms_cb_expander)):
                    if st.session_state.get('adata_clustered') is not None:
                        st.session_state.current_stage = "annotation_running"
                        st.session_state.adata_annotated = None 
                        
                        adata_for_annotation_expander = st.session_state.adata_clustered.copy() 

                        if run_ora_cb_expander:
                            with st.spinner("Εκτελείται ORA..."):
                                try: 
                                    adata_for_annotation_expander = run_decoupler_ora(
                                        adata_for_annotation_expander, 
                                        st.session_state.marker_dict, 
                                        min_n=param_anno_min_n_expander
                                    )
                                except Exception as e: st.error(f"Σφάλμα ORA: {e}")
                        
                        if run_wms_cb_expander:
                            with st.spinner("Εκτελείται WMS..."):
                                try: 
                                    adata_for_annotation_expander = run_decoupler_wmean(
                                        adata_for_annotation_expander, 
                                        st.session_state.marker_dict, 
                                        min_n=param_anno_min_n_expander
                                    )
                                except Exception as e: st.error(f"Σφάλμα WMS: {e}")
                        
                        st.session_state.adata_annotated = adata_for_annotation_expander 
                        st.session_state.current_stage = "annotation_results_display"
                        st.rerun()
                    else:
                        st.error("Δεν υπάρχουν clustered δεδομένα για εκτέλεση Decoupler.") # Changed from st.sidebar.error

    # Display Annotation Results
    if st.session_state.get('adata_annotated') is not None and st.session_state.current_stage == "annotation_results_display":
        with st.container():
            st.header("5. Αποτελέσματα Χαρακτηρισμού Κυτταρικών Τύπων")
            adata_ann = st.session_state.adata_annotated

            if 'ora_estimate' in adata_ann.obsm:
                st.subheader("ORA Scores")
                try:
                    # Ensure decoupler (dc) is imported
                    ora_scores_df = dc.get_acts(adata_ann, obsm_key='ora_estimate').to_df() 
                    if 'leiden' in adata_ann.obs:
                        if adata_ann.obs.index.equals(ora_scores_df.index): # Check index alignment
                            ora_scores_df['leiden'] = adata_ann.obs['leiden'].values.astype(str) # Ensure leiden is string for grouping
                            ora_pivot = ora_scores_df.groupby('leiden').mean()
                            if ora_pivot.isnull().all().all():
                                st.warning("ORA scores (mean per cluster) are all NaN.")
                            else:
                                fig_ora_hm, ax_ora_hm = plt.subplots(figsize=(max(8, len(ora_pivot.columns)*0.8), max(6, len(ora_pivot.index)*0.5)))
                                sns.heatmap(ora_pivot.T, cmap="viridis", annot=False, ax=ax_ora_hm) # annot=True can be too crowded
                                ax_ora_hm.set_title("ORA Scores Heatmap (Mean per Leiden Cluster)")
                                st.pyplot(fig_ora_hm)
                                plt.close(fig_ora_hm)
                        else:
                            st.warning("Indices mismatch for ORA heatmap. Displaying raw ORA scores.")
                            st.dataframe(ora_scores_df.head())
                    else:
                        st.warning("Leiden clusters not found for ORA heatmap. Displaying raw ORA scores.")
                        st.dataframe(ora_scores_df.head())
                except Exception as e:
                    st.warning(f"Δεν ήταν δυνατή η δημιουργία heatmap/table για ORA: {e}")
            
            if 'wmean_estimate' in adata_ann.obsm or any(col.startswith('wms_') for col in adata_ann.obs.columns): # Check both .obsm and .obs for WMS results
                st.subheader("WMS Scores (Weighted Mean)")
                # WMS scores from run_decoupler_wmean are often added directly to .obs
                wms_cols_in_obs = [col for col in adata_ann.obs.columns if col.startswith('wms_')]

                if wms_cols_in_obs and 'leiden' in adata_ann.obs:
                    num_cell_types_to_plot = min(len(wms_cols_in_obs), 5) # Plot top 5 or fewer
                    if num_cell_types_to_plot > 0:
                        try:
                            if num_cell_types_to_plot == 1:
                                fig_wms_vln, axs_wms_vln_single = plt.subplots(figsize=(5, 4))
                                axs_wms_vln = [axs_wms_vln_single] # Make it a list for consistent indexing
                            else:
                                fig_wms_vln, axs_wms_vln = plt.subplots(1, num_cell_types_to_plot, figsize=(3*num_cell_types_to_plot, 4), sharey=True)
                            
                            for i, col in enumerate(wms_cols_in_obs[:num_cell_types_to_plot]):
                                current_ax = axs_wms_vln[i] if num_cell_types_to_plot > 1 else axs_wms_vln[0]
                                sc.pl.violin(adata_ann, keys=col, groupby='leiden', ax=current_ax, show=False, rotation=90)
                                current_ax.set_title(col.replace('wms_','')) # Clean title
                            plt.tight_layout()
                            st.pyplot(fig_wms_vln)
                            plt.close(fig_wms_vln)
                        except Exception as e:
                            st.error(f"Σφάλμα δημιουργίας WMS violin plots: {e}")
                elif 'wmean_estimate' in adata_ann.obsm: # Fallback to .obsm if not in .obs
                     st.dataframe(dc.get_acts(adata_ann, obsm_key='wmean_estimate').to_df().head())


    # --- Data Previews at the end ---
    if st.session_state.get('adata_processed', None) is not None and \
       st.session_state.current_stage not in ["load_data", "loading_data", "preprocessing_running"]:
        with st.expander("Προεπισκόπηση Προεπεξεργασμένου AnnData (στο τέλος)", expanded=False):
            st.text(str(st.session_state.adata_processed))
            st.dataframe(st.session_state.adata_processed.obs.head())
            st.dataframe(st.session_state.adata_processed.var.head())
            if 'highly_variable' in st.session_state.adata_processed.var.columns:
                st.write(f"Εντοπίστηκαν {st.session_state.adata_processed.var.highly_variable.sum()} ιδιαίτερα μεταβλητά γονίδια.")

    if st.session_state.get('adata_clustered', None) is not None and \
       st.session_state.current_stage not in ["load_data", "loading_data", "preprocessing_running", "dim_reduction_running"]:
        with st.expander("Προεπισκόπηση Clustered AnnData (στο τέλος)", expanded=False):
            st.text(str(st.session_state.adata_clustered))
            st.dataframe(st.session_state.adata_clustered.obs.head())
    
    # Debugging: Show current stage and session state items (optional)
    # st.sidebar.write("Current Stage (bottom):", st.session_state.current_stage)
    # st.write("Final SS (types):", {k: type(v).__name__ for k, v in st.session_state.items()})


with tab2:
    st.header("👥 Πληροφορίες Ομάδας")
    st.write("Όνομα Project: Ανάπτυξη Διαδραστικής Εφαρμογής scRNA-seq")
    st.write("Μέλη Ομάδας:")
    st.markdown("""
    - Στεργιούλας Γεώργιος (inf2021216)
    - Αναγνωστόπουλος Φίλιππος (inf2021014)
    """)
    st.write("Ημερομηνία: Μάιος 2025")

# Clear uploader if files are cleared by the user (X button)
# This is tricky because file_uploader doesn't have a direct "on_clear" callback
# One way is to check if uploaded_files_val becomes empty AFTER it had files
# However, the current flow with "Επεξεργασία & Φόρτωση" button makes this less critical
# as state is reset upon that button press anyway.