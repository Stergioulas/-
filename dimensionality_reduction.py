import scanpy as sc

def run_pca_umap_leiden(
    adata_processed,
    n_pcs=30, 
    n_neighbors=15,
    umap_min_dist=0.5,
    leiden_resolution=1.0,
    use_highly_variable=True
):
    """
    Εκτελεί PCA, υπολογισμό γειτόνων, UMAP και Leiden clustering.

    Args:
        adata_processed: Το προεπεξεργασμένο AnnData object (μετά το QC, normalize, log, HVG, scale).
        n_pcs: Αριθμός Principal Components για χρήση.
        n_neighbors: Αριθμός γειτόνων για τον υπολογισμό του UMAP/Leiden.
        umap_min_dist: Παράμετρος min_dist για το UMAP.
        leiden_resolution: Παράμετρος ανάλυσης για τον αλγόριθμο Leiden.
        use_highly_variable: Αν θα χρησιμοποιηθούν μόνο τα HVGs για το PCA.

    Returns:
        Το AnnData object με τα αποτελέσματα PCA, UMAP και Leiden clustering.
    """
    adata_dimred = adata_processed.copy()

    # 1. PCA
    # Το sc.pp.highly_variable_genes θα πρέπει να έχει τρέξει ήδη στην προεπεξεργασία
    # και τα δεδομένα να είναι κλιμακωμένα (sc.pp.scale)
    print(f"Εκτέλεση PCA με n_pcs={n_pcs}...")
    sc.tl.pca(adata_dimred, n_comps=n_pcs, use_highly_variable=use_highly_variable)
    
    # 2. Υπολογισμός Γειτόνων
    # Χρησιμοποιούμε την αναπαράσταση PCA για τον υπολογισμό των γειτόνων
    print(f"Υπολογισμός γειτόνων με n_neighbors={n_neighbors} χρησιμοποιώντας PCA...")
    sc.pp.neighbors(adata_dimred, n_neighbors=n_neighbors, n_pcs=n_pcs)
    
    # 3. UMAP
    print(f"Υπολογισμός UMAP με min_dist={umap_min_dist}...")
    sc.tl.umap(adata_dimred, min_dist=umap_min_dist)
    
    # 4. Leiden Clustering
    print(f"Εκτέλεση Leiden clustering με resolution={leiden_resolution}...")
    sc.tl.leiden(adata_dimred, resolution=leiden_resolution, key_added='leiden')
    
    print("Ολοκληρώθηκε η μείωση διαστατικότητας και το clustering.")
    return adata_dimred

# Παράδειγμα χρήσης (μπορεί να αφαιρεθεί ή να γίνει comment out αργότερα)
if __name__ == '__main__':
    import numpy as np
    # Δημιουργία ενός ψεύτικου AnnData (μετά από υποτιθέμενη προεπεξεργασία)
    # Θα πρέπει να έχει HVGs και να είναι scaled
    counts = np.random.poisson(1, size=(300, 1000))
    adata_test = sc.AnnData(counts)
    adata_test.var_names = [f'gene_{i}' for i in range(adata_test.n_vars)]
    adata_test.obs_names = [f'cell_{i}' for i in range(adata_test.n_obs)]
    
    # Προσομοίωση προηγούμενων βημάτων
    sc.pp.calculate_qc_metrics(adata_test, percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata_test)
    sc.pp.log1p(adata_test)
    sc.pp.highly_variable_genes(adata_test, n_top_genes=200)
    adata_test.raw = adata_test
    sc.pp.scale(adata_test)

    print("--- Δοκιμή της συνάρτησης run_pca_umap_leiden ---")
    adata_final = run_pca_umap_leiden(
        adata_test,
        n_pcs=20,
        n_neighbors=10,
        leiden_resolution=0.8
    )
    print("--- Η διαδικασία ολοκληρώθηκε για το δοκιμαστικό AnnData ---")
    print(f"Τελικό σχήμα: {adata_final.shape}")
    print(f"Διαθέσιμα UMAP embeddings: {'X_umap' in adata_final.obsm}")
    print(f"Διαθέσιμα Leiden clusters: {'leiden' in adata_final.obs}")
    if 'leiden' in adata_final.obs:
        print(f"Leiden clusters: {np.unique(adata_final.obs['leiden'])}") 