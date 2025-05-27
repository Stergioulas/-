import scanpy as sc
import numpy as np
# import pandas as pd # Δεν φαίνεται να χρησιμοποιείται άμεσα εδώ
# import hdf5plugin # Δεν χρειάζεται για την ίδια την προεπεξεργασία, μόνο για το write_h5ad
# import scipy as sci # Δεν φαίνεται να χρησιμοποιείται άμεσα εδώ

def preprocess_adata(
    adata_initial, 
    min_genes=200, 
    min_cells=3, 
    target_sum=1e4, # Για normalize_total
    n_top_genes=2000, # Για highly_variable_genes
    min_counts_gene=3, # Παράμετρος για sc.pp.filter_genes
    min_n_genes_cell=200, # Παράμετρος για sc.pp.filter_cells
    max_n_genes_cell=7000, # Παράμετρος για sc.pp.filter_cells
    pct_mito_threshold=10.0, # Παράμετρος για ποσοστό μιτοχονδριακών
    # n_genes_min=1000, # Παλιές παράμετροι, θα χρησιμοποιήσουμε τις νέες παραπάνω
    # n_genes_max=10000, 
    # n_counts_max=30000, 
    # pc_rib=25, # Δεν χρησιμοποιείται συχνά, μπορεί να προστεθεί αν χρειαστεί
    debug=True
):
    """
    Προεπεξεργασία ενός AnnData object.

    Args:
        adata_initial: Το αρχικό AnnData object.
        min_genes: Ελάχιστος αριθμός γονιδίων ανά κύτταρο (παράμετρος του sc.pp.filter_cells).
        min_cells: Ελάχιστος αριθμός κυττάρων ανά γονίδιο (παράμετρος του sc.pp.filter_genes).
        target_sum: Άθροισμα στόχος για κανονικοποίηση ανά κύτταρο.
        n_top_genes: Αριθμός των πιο μεταβλητών γονιδίων προς επιλογή.
        min_counts_gene: Ελάχιστος αριθμός counts για ένα γονίδιο ώστε να διατηρηθεί.
        min_n_genes_cell: Ελάχιστος αριθμός γονιδίων που εκφράζονται σε ένα κύτταρο ώστε να διατηρηθεί.
        max_n_genes_cell: Μέγιστος αριθμός γονιδίων που εκφράζονται σε ένα κύτταρο ώστε να διατηρηθεί (για doublets).
        pct_mito_threshold: Κατώφλι ποσοστού μιτοχονδριακών γονιδίων για φιλτράρισμα κυττάρων.
        debug: Εκτύπωση μηνυμάτων προόδου.

    Returns:
        Ένα νέο, προεπεξεργασμένο AnnData object.
    """
    
    adata = adata_initial.copy() # Δουλεύουμε σε αντίγραφο για να μην αλλάξουμε το αρχικό

    def prn(txt):
        if debug:
            print(txt)
    
    prn(f"Αρχικό σχήμα δεδομένων: {adata.shape}")
    
    # 1. Βασικό φιλτράρισμα γονιδίων και κυττάρων
    # sc.pp.filter_cells(adata, min_genes=min_genes) # Χρησιμοποιούμε το min_n_genes_cell παρακάτω
    # prn(f"Φιλτράρισμα κυττάρων με γονίδια < {min_genes}: {adata.shape}")
    
    # sc.pp.filter_genes(adata, min_cells=min_cells) # Χρησιμοποιούμε το min_counts_gene παρακάτω
    # prn(f"Φιλτράρισμα γονιδίων που εκφράζονται σε < {min_cells} κύτταρα: {adata.shape}")

    # 2. Υπολογισμός ποιοτικών μετρικών
    adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))  # Προσδιορισμός μιτοχονδριακών γονιδίων
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    prn("Υπολογίστηκαν τα QC metrics (π.χ., n_genes_by_counts, total_counts, pct_counts_mt)")

    # 3. Φιλτράρισμα βάσει QC metrics
    prn(f"Σχήμα πριν το φιλτράρισμα QC: {adata.shape}")
    sc.pp.filter_genes(adata, min_counts=min_counts_gene)
    prn(f"  Μετά το φιλτράρισμα γονιδίων (min_counts={min_counts_gene}): {adata.shape}")
    
    sc.pp.filter_cells(adata, min_genes=min_n_genes_cell)
    prn(f"  Μετά το φιλτράρισμα κυττάρων (min_genes={min_n_genes_cell}): {adata.shape}")
    
    adata = adata[adata.obs.n_genes_by_counts < max_n_genes_cell, :]
    prn(f"  Μετά το φιλτράρισμα κυττάρων (max_genes={max_n_genes_cell}): {adata.shape}")
    
    adata = adata[adata.obs.pct_counts_mt < pct_mito_threshold, :]
    prn(f"  Μετά το φιλτράρισμα κυττάρων (pct_mito < {pct_mito_threshold}%): {adata.shape}")
    prn(f"Τελικό σχήμα μετά το QC φιλτράρισμα: {adata.shape}")

    # 4. Κανονικοποίηση ΠΡΙΝ την αποθήκευση στο .raw και ΠΡΙΝ το log1p
    sc.pp.normalize_total(adata, target_sum=target_sum)
    prn(f"Έγινε normalize_total (target_sum={target_sum}).")

    # Αποθήκευση των κανονικοποιημένων (αλλά όχι log-transformed) δεδομένων στο .raw
    # Αυτό είναι συχνά καλύτερο για το rank_genes_groups αν χρησιμοποιηθεί το use_raw=True
    adata.raw = adata.copy()
    prn("Τα κανονικοποιημένα δεδομένα αποθηκεύτηκαν στο adata.raw")

    sc.pp.log1p(adata)
    prn("Έγινε log1p μετασχηματισμός.")

    # 5. Εντοπισμός ιδιαίτερα μεταβλητών γονιδίων (Highly Variable Genes - HVGs)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')
    prn(f"Εντοπίστηκαν {adata.var.highly_variable.sum()} ιδιαίτερα μεταβλητά γονίδια (n_top_genes={n_top_genes}).")

    # Το raw αποθηκεύεται πριν την κλιμάκωση για downstream αναλύσεις (π.χ. DEG)
    adata.raw = adata

    # 6. Κλιμάκωση (Scaling)
    # Κλιμάκωση μόνο των HVGs για καλύτερη απόδοση σε PCA/Clustering
    # adata_scaled = adata[:, adata.var.highly_variable].copy()
    # sc.pp.scale(adata_scaled, max_value=10)
    # adata[:, adata.var.highly_variable] = adata_scaled.X
    # Ή κλιμάκωση όλων των γονιδίων αν προτιμάται για κάποιες εφαρμογές:
    sc.pp.scale(adata, max_value=10)
    prn("Έγινε κλιμάκωση των δεδομένων (max_value=10).")
    
    return adata

# Παράδειγμα χρήσης (μπορεί να αφαιρεθεί ή να γίνει comment out αργότερα)
if __name__ == '__main__':
    # Δημιουργία ενός ψεύτικου AnnData για δοκιμή
    counts = np.random.poisson(1, size=(100, 2000))
    adata_test = sc.AnnData(counts)
    adata_test.var_names = [f'gene_{i}' for i in range(adata_test.n_vars)]
    adata_test.obs_names = [f'cell_{i}' for i in range(adata_test.n_obs)]
    # Προσθήκη μερικών μιτοχονδριακών γονιδίων για δοκιμή
    adata_test.var_names = ['MT-gene_1' if i == 0 else name for i, name in enumerate(adata_test.var_names)]
    adata_test.var_names = ['MT-gene_2' if i == 1 else name for i, name in enumerate(adata_test.var_names)]


    print("--- Δοκιμή της συνάρτησης preprocess_adata ---")
    adata_processed = preprocess_adata(
        adata_test,
        min_genes=50, 
        min_cells=2,
        target_sum=1e4,
        n_top_genes=500,
        min_counts_gene=2,
        min_n_genes_cell=50,
        max_n_genes_cell=1500,
        pct_mito_threshold=20.0
    )
    print("--- Η προεπεξεργασία ολοκληρώθηκε για το δοκιμαστικό AnnData ---")
    print(f"Τελικό επεξεργασμένο σχήμα: {adata_processed.shape}")
    print(f"HVGs: {adata_processed.var.highly_variable.sum()}")
    if adata_processed.raw is not None:
        print(f"Raw data shape: {adata_processed.raw.X.shape}") 