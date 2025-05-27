import scanpy as sc
import pandas as pd
import numpy as np

def run_deg_analysis(
    adata_clustered,
    groupby_key,        # Π.χ., 'leiden', 'condition'
    group1,             # Λίστα με τις τιμές της ομάδας1 (π.χ., ['0', '1'] για clusters ή ['disease'] για condition)
    reference_groups='rest', # 'rest' ή μια λίστα με τις τιμές της ομάδας αναφοράς
    method='wilcoxon',      # Μέθοδος για το rank_genes_groups
    n_genes=200,            # Αριθμός top γονιδίων ανά ομάδα για αποθήκευση από το rank_genes_groups
    corr_method='benjamini-hochberg' # Μέθοδος διόρθωσης για πολλαπλές συγκρίσεις
):
    """
    Εκτελεί ανάλυση διαφορικά εκφρασμένων γονιδίων (DEG).

    Args:
        adata_clustered: Το AnnData object που περιέχει τα clusters ή τις ομάδες σύγκρισης.
        groupby_key: Το όνομα της στήλης στο .obs που θα χρησιμοποιηθεί για την ομαδοποίηση.
        group1: Η ομάδα (ή λίστα ομάδων) για την οποία θα βρεθούν τα DEGs.
        reference_groups: Η ομάδα αναφοράς. 'rest' για σύγκριση με όλες τις υπόλοιπες, 
                          ή μια λίστα με συγκεκριμένες ομάδες αναφοράς.
        method: Η στατιστική μέθοδος που θα χρησιμοποιηθεί (π.χ., 'wilcoxon', 't-test').
        n_genes: Ο αριθμός των top γονιδίων που θα αναφερθούν ανά ομάδα (για το DataFrame εξόδου).
                 Η rank_genes_groups θα υπολογίσει για όλα τα γονίδια αν το n_genes στο sc.tl.rank_genes_groups είναι αρκετά μεγάλο.
        corr_method: Μέθοδος διόρθωσης για πολλαπλές συγκρίσεις.

    Returns:
        Ένα pandas DataFrame με τα αποτελέσματα των DEGs, ή None αν προκύψει σφάλμα.
        Το DataFrame περιέχει στήλες για 'names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj'.
        Επίσης, το anndata object ενημερώνεται inplace με τα αποτελέσματα στο .uns['rank_genes_groups'].
    """
    adata_copy = adata_clustered.copy() 
    print(f"Έναρξη DEG: {group1} vs {reference_groups} by '{groupby_key}' ({method}). use_raw=True")
    try:
        sc.tl.rank_genes_groups(
            adata_copy, 
            groupby=groupby_key,
            groups=group1, 
            reference=reference_groups,
            method=method,
            n_genes=adata_copy.n_vars, 
            corr_method=corr_method,
            use_raw=True # ΧΡΗΣΗ ΤΟΥ ADATA.RAW
        )
        print("rank_genes_groups ολοκληρώθηκε.")

        target_group_for_results = str(group1[0])
        rgg_results = adata_copy.uns['rank_genes_groups']
        required_keys = ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj']
        data_for_df = {}
        all_keys_present = True

        for key in required_keys:
            field_data = None
            if key in rgg_results: # Έλεγχος αν το κλειδί υπάρχει στο λεξικό rgg_results
                # Τα αποτελέσματα είναι structured arrays, προσπελάσιμα με το όνομα της ομάδας
                if isinstance(rgg_results[key], np.ndarray) and rgg_results[key].dtype.names is not None and target_group_for_results in rgg_results[key].dtype.names:
                    field_data = rgg_results[key][target_group_for_results]
                # Εναλλακτικά, αν το rgg_results[key] είναι ένα απλό numpy array (όταν groups='all')
                # ή αν η ομάδα είναι άμεσα προσπελάσιμη ως string key (λιγότερο συνηθισμένο πλέον)
                elif isinstance(rgg_results[key], dict) and target_group_for_results in rgg_results[key]:
                     field_data = rgg_results[key][target_group_for_results]
                elif isinstance(rgg_results[key], np.ndarray): # Αν δεν είναι structured αλλά απλό array (π.χ. από παλιότερη έκδοση ή λάθος κλήση)
                     print(f"Warning: Field '{key}' is a simple numpy array, not structured or dict as expected for group-specific results.")
                     # Προσπάθεια να το πάρουμε αν είναι μόνο μια ομάδα, αλλιώς παράλειψη
                     if len(group1) == 1: #  and rgg_results[key].shape[0] == adata_copy.n_vars (υποθέτοντας ότι είναι για όλα τα γονίδια)
                         field_data = rgg_results[key]
                     else:
                        print(f"!!! Δεν ήταν δυνατή η ανάκτηση του '{key}' για την ομάδα '{target_group_for_results}' από απλό array πολλαπλών ομάδων.")
                        all_keys_present = False
                        break
                else:
                    print(f"!!! Το πεδίο '{key}' για την ομάδα '{target_group_for_results}' δεν έχει την αναμενόμενη δομή (structured array or dict).")
                    all_keys_present = False
                    break
            else:
                print(f"!!! Το κλειδί '{key}' δεν βρέθηκε στο adata_copy.uns['rank_genes_groups']")
                all_keys_present = False
                break
            
            if field_data is not None:
                # Χειρισμός NaN/inf στο logfoldchanges
                if key == 'logfoldchanges':
                    field_data = np.nan_to_num(field_data, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
                data_for_df[key] = field_data[:n_genes]
            else:
                # Αυτό το block δεν θα έπρεπε να εκτελεστεί αν το all_keys_present είναι σωστό
                all_keys_present = False 
                break
        
        if all_keys_present:
            results_df = pd.DataFrame(data_for_df)
            # Εξασφάλιση ότι οι στήλες pval και pval_adj είναι αριθμητικές και όχι None/object
            for p_col in ['pvals', 'pvals_adj']:
                if p_col in results_df.columns:
                    results_df[p_col] = pd.to_numeric(results_df[p_col], errors='coerce').fillna(1.0)
            print(f"DEG DataFrame για '{target_group_for_results}': {results_df.shape}")
            return adata_copy, results_df 
        else:
            print(f"!!! Δεν εξήχθησαν όλα τα πεδία DEG για '{target_group_for_results}'.")
            return adata_copy, None

    except Exception as e:
        print(f"!!! Σφάλμα DEG: {e}")
        import traceback
        print(traceback.format_exc())
        return adata_clustered, None 

# Παράδειγμα χρήσης (μπορεί να αφαιρεθεί ή να γίνει comment out αργότερα)
if __name__ == '__main__':
    import numpy as np
    adata_test = sc.AnnData(np.random.rand(100, 500))
    adata_test.obs['leiden'] = np.random.choice(['0', '1', '2'], size=100).astype('category')
    adata_test.obs['condition'] = np.random.choice(['A', 'B'], size=100).astype('category')
    adata_test.var_names = [f'gene_{i}' for i in range(500)]
    # Προσομοίωση κανονικοποιημένων δεδομένων στο raw
    adata_test.X = np.random.poisson(5, size=(100,500)).astype(float) # Raw-like counts
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    adata_test.raw = adata_test.copy() # Store normalized in raw
    sc.pp.log1p(adata_test) # Logarithmize .X for other computations

    print("--- Δοκιμή της συνάρτησης run_deg_analysis (Leiden) ---")
    adata_res_leiden, degs_leiden = run_deg_analysis(adata_test, groupby_key='leiden', group1=['0'], reference_groups='rest', n_genes=10)
    if degs_leiden is not None:
        print("Top DEGs για το cluster '0' έναντι των υπολοίπων:")
        print(degs_leiden.head())
        print(degs_leiden.dtypes)
    if 'rank_genes_groups' in adata_res_leiden.uns:
        print("Το rank_genes_groups υπάρχει στο anndata μετά την ανάλυση Leiden.")
    
    print("\n--- Δοκιμή της συνάρτησης run_deg_analysis (Condition) ---")
    adata_res_cond, degs_condition = run_deg_analysis(adata_test, groupby_key='condition', group1=['A'], reference_groups=['B'], n_genes=10)
    if degs_condition is not None:
        print("Top DEGs για τη συνθήκη 'A' έναντι 'B':")
        print(degs_condition.head())
        print(degs_condition.dtypes)
    if 'rank_genes_groups' in adata_res_cond.uns:
        print("Το rank_genes_groups υπάρχει στο anndata μετά την ανάλυση Condition.") 