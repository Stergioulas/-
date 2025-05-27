import scanpy as sc
import pandas as pd
import decoupler as dc
import numpy as np # Θα χρειαστεί για το WMS αν κάνουμε pivot

def create_marker_dict_from_df(marker_df, cell_type_col='cell_name', gene_symbol_col='Symbol'):
    """
    Δημιουργεί ένα λεξικό γονιδίων-δεικτών από ένα pandas DataFrame.
    Κάθε κλειδί είναι ένας τύπος κυττάρου και κάθε τιμή είναι μια λίστα γονιδίων.
    """
    if not isinstance(marker_df, pd.DataFrame):
        raise ValueError("Το marker_df πρέπει να είναι pandas DataFrame.")
    if cell_type_col not in marker_df.columns:
        raise ValueError(f"Η στήλη '{cell_type_col}' δεν βρέθηκε στο DataFrame των δεικτών.")
    if gene_symbol_col not in marker_df.columns:
        raise ValueError(f"Η στήλη '{gene_symbol_col}' δεν βρέθηκε στο DataFrame των δεικτών.")

    marker_dict = {}
    for cell_type, group in marker_df.groupby(cell_type_col):
        marker_dict[str(cell_type)] = group[gene_symbol_col].unique().tolist()
    print(f"Δημιουργήθηκε λεξικό δεικτών με {len(marker_dict)} τύπους κυττάρων.")
    return marker_dict

def _marker_dict_to_df(marker_dict, source_col_name='source', target_col_name='target', weight_col_name='weight'):
    """Μετατρέπει ένα λεξικό δεικτών σε DataFrame κατάλληλο για το decoupler."""
    df_list = []
    for cell_type, genes in marker_dict.items():
        for gene in genes:
            df_list.append({source_col_name: cell_type, target_col_name: gene, weight_col_name: 1})
    return pd.DataFrame(df_list)

def run_decoupler_ora(
    adata_in,
    marker_dict,
    min_n=5
):
    adata = adata_in.copy() # Δουλεύουμε σε αντίγραφο
    if adata.raw is None:
        print("!!! Προσοχή: Το adata.raw είναι None για ORA. Χρησιμοποιείται το .X")
        # if not 'n_genes' in adata.obs.columns: 
        #     sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    print(f"Εκτέλεση ORA με min_n={min_n}...")

    print("--- DEBUGGING decoupler API (before ORA call) ---")
    print(f"Decoupler version: {dc.__version__ if hasattr(dc, '__version__') else 'unknown'}")
    print(f"Type of dc: {type(dc)}")
    print(f"Attributes of dc: {dir(dc)}")
    if hasattr(dc, 'mt'):
        print(f"Type of dc.mt: {type(dc.mt)}")
        print(f"Attributes of dc.mt: {dir(dc.mt)}")
    else:
        print("dc does NOT have attribute 'mt'")
    print("--- END DEBUGGING ---")

    try:
        # Μετατροπή του marker_dict σε DataFrame κατάλληλο για decoupler
        marker_df_for_decoupler = _marker_dict_to_df(marker_dict, source_col_name='source', target_col_name='target')
        # Corrected call for decoupler 2.0.1 based on debug output
        dc.mt.ora(
            data=adata, # Changed from mat, and source/target/use_raw are removed as they might not be direct args for mt.ora
            net=marker_df_for_decoupler,
            # source='source', # Assuming inferred from net
            # target='target', # Assuming inferred from net
            verbose=True,
            # use_raw=adata.raw is not None, # use_raw likely not a direct arg for internal mt.ora
            min_n=min_n
        )
        print("ORA ολοκληρώθηκε.")
        if 'ora_estimate' in adata.obsm and 'ora_pvals' in adata.obsm:
            print("ORA scores/pvals στο .obsm")
        else:
            print("!!! Δεν βρέθηκαν ora_estimate/ora_pvals στο .obsm")
        return adata
    except Exception as e:
        print(f"!!! Σφάλμα ORA: {e}")
        import traceback
        print(traceback.format_exc())
        return adata_in # Επιστροφή του αρχικού σε σφάλμα

def run_decoupler_wmean(
    adata_in, 
    marker_dict, 
    min_n=5
):
    adata = adata_in.copy() # Δουλεύουμε σε αντίγραφο
    print(f"Εκτέλεση WMS με min_n={min_n}...")

    try:
        # Μετατροπή του marker_dict σε DataFrame κατάλληλο για decoupler
        marker_df_for_decoupler = _marker_dict_to_df(marker_dict, source_col_name='source', target_col_name='target', weight_col_name='weight')
        # Using dc.mt.waggr based on debug output, assuming it handles weighted mean/sum
        dc.mt.waggr(
            data=adata, 
            net=marker_df_for_decoupler,
            verbose=True
            # min_n=min_n, # Removed, as it caused TypeError for _func_waggr
            # We might need to specify how aggregation is done (e.g., mean vs sum) if waggr is general
            # Also, waggr might require specific arguments like `groups_key` or `groupby` if it aggregates over obs groups
        )
        print("WMS (waggr) ολοκληρώθηκε.")
        if 'wmean_estimate' in adata.obsm:
            print("WMS scores στο .obsm")
            # Το dc.get_acts επιστρέφει ένα AnnData με τα activities στο .X
            # Μετατρέπουμε το .X σε DataFrame
            wms_activities = dc.get_acts(adata, obsm_key='wmean_estimate')
            wms_scores_df = pd.DataFrame(wms_activities.X, columns=wms_activities.var_names, index=wms_activities.obs_names)
            
            for col in wms_scores_df.columns:
                if col not in adata.obs.columns:
                    adata.obs[f'wms_{col}'] = wms_scores_df[col].values
                else:
                    print(f"Warning: Column wms_{col} already exists in .obs. Skipping.")
            print("WMS scores προστέθηκαν στο .obs")
        else:
            print("!!! Δεν βρέθηκε wmean_estimate στο .obsm")
        return adata
    except Exception as e:
        print(f"!!! Σφάλμα WMS: {e}")
        import traceback
        print(traceback.format_exc())
        return adata_in # Επιστροφή του αρχικού σε σφάλμα

# Παράδειγμα χρήσης
if __name__ == '__main__':
    # Δημιουργία ψεύτικου AnnData
    adata_test = sc.AnnData(np.random.rand(100, 500))
    adata_test.var_names = [f'gene_{i+1}' for i in range(500)]
    adata_test.obs_names = [f'cell_{i+1}' for i in range(100)]
    # Δημιουργία raw layer για δοκιμή ORA
    adata_test.raw = adata_test.copy()
    sc.pp.log1p(adata_test) # Το .X είναι τώρα λογαριθμισμένο

    # Δημιουργία ψεύτικου marker_df
    marker_data = {
        'cell_name': ['TypeA', 'TypeA', 'TypeB', 'TypeB', 'TypeB', 'TypeC'],
        'Symbol': ['gene_1', 'gene_2', 'gene_3', 'gene_4', 'gene_5', 'gene_1']
    }
    df_markers_test = pd.DataFrame(marker_data)
    marker_dict_test = create_marker_dict_from_df(df_markers_test)
    print("Δοκιμαστικό marker_dict:", marker_dict_test)

    print("\n--- Δοκιμή ORA ---")
    adata_ora = run_decoupler_ora(adata_test.copy(), marker_dict_test, min_n=1)
    if 'ora_estimate' in adata_ora.obsm:
        print("ORA estimates (head):")
        print(dc.get_acts(adata_ora, obsm_key='ora_estimate').to_df().head())
    
    print("\n--- Δοκιμή WMS ---")
    adata_wms = run_decoupler_wmean(adata_test.copy(), marker_dict_test, min_n=1)
    if 'wmean_estimate' in adata_wms.obsm:
        # Για τη δοκιμή, ας τυπώσουμε το DataFrame που προκύπτει
        wms_df_test = pd.DataFrame(dc.get_acts(adata_wms, obsm_key='wmean_estimate').X, 
                                   columns=dc.get_acts(adata_wms, obsm_key='wmean_estimate').var_names, 
                                   index=dc.get_acts(adata_wms, obsm_key='wmean_estimate').obs_names)
        print("WMS estimates (head) from .obsm (converted to DataFrame):")
        print(wms_df_test.head())
    if 'wms_TypeA' in adata_wms.obs:
        print("WMS scores στο .obs (head for wms_TypeA):")
        print(adata_wms.obs[['wms_TypeA', 'wms_TypeB', 'wms_TypeC']].head()) 