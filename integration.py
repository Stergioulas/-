import scanpy as sc
import scanpy.external as sce
import numpy as np
import pandas as pd
import streamlit as st # Προσθήκη Streamlit για το spinner

def run_scanorama_integration(
    adata_concatenated,
    batch_key='batch_upload', # Η στήλη στο obs που υποδεικνύει το batch
    dimred=50, # Παράμετρος για το scanorama.integrate_scanpy
    n_pcs_for_scanorama_input=50 # Αριθμός PCs για υπολογισμό πριν το Scanorama, αν χρειαστεί
):
    """
    Εκτελεί ενοποίηση δεδομένων με Scanorama.

    Args:
        adata_concatenated: Ένα AnnData object που περιέχει τα συνενωμένα δεδομένα από πολλαπλά batches.
                            Πρέπει να περιέχει μια στήλη στο .obs που ορίζεται από το batch_key.
        batch_key: Το όνομα της στήλης στο .obs που περιέχει την πληροφορία του batch.
        dimred: Ο αριθμός των διαστάσεων για την ενσωμάτωση Scanorama.
        n_pcs_for_scanorama_input: Αν τα δεδομένα δεν έχουν ήδη PCA, θα τρέξει PCA με τόσες συνιστώσες.

    Returns:
        Ένα νέο AnnData object με τα δεδομένα ενοποιημένα μέσω Scanorama.
        Το αποτέλεσμα της ενοποίησης αποθηκεύεται στο .obsm['X_scanorama'].
    """
    print(f"Έναρξη ενοποίησης με Scanorama χρησιμοποιώντας batch_key: '{batch_key}'")
    
    # Δημιουργία λίστας από AnnData objects, ένα για κάθε batch
    batches = adata_concatenated.obs[batch_key].cat.categories.tolist()
    adata_list = []
    for batch_val in batches:
        adata_batch = adata_concatenated[adata_concatenated.obs[batch_key] == batch_val, :].copy()
        # Βασική προεπεξεργασία ανά batch (αν δεν έχει γίνει ήδη εκτενώς)
        # Το Scanorama λειτουργεί καλύτερα με κανονικοποιημένα και λογαριθμισμένα δεδομένα
        # και συχνά με επιλεγμένα HVGs.
        if 'X_pca' not in adata_batch.obsm: # Απλή εκδοχή: αν δεν υπάρχει PCA, κάνε βασικά βήματα
            print(f"  Προεπεξεργασία για το batch {batch_val}...")
            sc.pp.normalize_total(adata_batch, target_sum=1e4)
            sc.pp.log1p(adata_batch)
            # sc.pp.highly_variable_genes(adata_batch, n_top_genes=2000, flavor='seurat_v3')
            # adata_batch = adata_batch[:, adata_batch.var.highly_variable].copy() # Κράτα μόνο HVGs
            # sc.pp.scale(adata_batch, max_value=10) # Το Scanorama δεν θέλει scaled data
        adata_list.append(adata_batch)
    
    print(f"Δημιουργήθηκε λίστα με {len(adata_list)} anndata objects (ένα ανά batch).")

    # Εκτέλεση Scanorama
    # Το scanorama.integrate_scanpy τροποποιεί τα anndata objects στη λίστα inplace,
    # προσθέτοντας το .obsm['X_scanorama']
    print(f"Εκτέλεση scanorama.integrate_scanpy με dimred={dimred}...")
    with st.spinner(f"Running Scanorama integration (dimred={dimred})..."):
        sce.pp.scanorama_integrate(adata_list, batch_key=batch_key, dimred=dimred, approx=True)
        # Το integrate_scanpy δεν υπάρχει πλέον, χρησιμοποιούμε το scanorama_integrate
        # sce.pp.scanorama_integrate(adata_list, dimred=dimred) 
        # Ήταν: scanorama.integrate_scanpy(adata_list, dimred=dimred)

        print("Η ενοποίηση Scanorama ολοκληρώθηκε για τα επιμέρους anndata objects.")

        # Συνένωση των anndata objects πίσω σε ένα, κρατώντας το X_scanorama
        # Πρώτα, βεβαιωνόμαστε ότι όλα έχουν το X_scanorama
        for i, ad in enumerate(adata_list):
            if 'X_scanorama' not in ad.obsm:
                raise ValueError(f"Το AnnData object για το batch {batches[i]} δεν περιέχει 'X_scanorama' μετά την ενοποίηση.")

        # Δημιουργούμε ένα νέο anndata για να κρατήσουμε την ενσωμάτωση
        # Θα πάρουμε το .obs από το αρχικό concatenated και θα προσθέσουμε το X_scanorama
        # Αυτό είναι λίγο πιο περίπλοκο γιατί το scanorama.integrate_scanpy δεν επιστρέφει ένα ενιαίο anndata
        
        # Παίρνουμε το X_scanorama από κάθε anndata και το συνδυάζουμε
        all_scanorama_embeddings = [ad.obsm['X_scanorama'] for ad in adata_list]
        
        # Για να τα συνδυάσουμε, πρέπει να ξέρουμε τη σειρά των κυττάρων στο αρχικό adata_concatenated
        # και να τα αντιστοιχίσουμε. Ο πιο απλός τρόπος είναι να ξαναφτιάξουμε το concatenated
        # και να του προσθέσουμε το X_scanorama.
        
        # Προσοχή: Η σειρά των κυττάρων στα adata_list πρέπει να διατηρηθεί όπως ήταν στο αρχικό adata_concatenated
        # όταν έγινε ο διαχωρισμός ανά batch. Ο παραπάνω τρόπος διαχωρισμού διατηρεί τη σχετική σειρά εντός των batches,
        # αλλά η σειρά των batches στην adata_list μπορεί να μην είναι η ίδια με τη σειρά εμφάνισης στο αρχικό.
        # Για απλότητα, θα τα συνενώσουμε ξανά και θα προσθέσουμε το ενιαίο embedding.

        # Δημιουργούμε ένα νέο AnnData object από την αρχή για το αποτέλεσμα
        # Χρησιμοποιούμε τα αρχικά δεδομένα έκφρασης από το adata_concatenated.X
        # και τις παρατηρήσεις από το adata_concatenated.obs
        # και τις μεταβλητές από το adata_concatenated.var
        # Το σημαντικό είναι να φτιάξουμε το ενιαίο .obsm['X_scanorama']

        # Σωστός τρόπος: Επανασυναρμολόγηση του X_scanorama με βάση τα αρχικά cell indices
        # Δημιουργούμε ένα κενό array για το τελικό X_scanorama
        final_x_scanorama = np.zeros((adata_concatenated.n_obs, dimred))
        current_idx = 0
        for i, batch_val in enumerate(batches):
            adata_batch_original_slice = adata_concatenated[adata_concatenated.obs[batch_key] == batch_val, :]
            num_cells_in_batch = adata_batch_original_slice.n_obs
            
            # Βρίσκουμε το αντίστοιχο anndata από την adata_list που έχει το X_scanorama
            # Αυτό υποθέτει ότι η σειρά των batches στην adata_list είναι ίδια με τη σειρά των categories
            adata_integrated_batch = adata_list[i] 
            
            if adata_integrated_batch.n_obs != num_cells_in_batch:
                print(f"Προσοχή: Ασυμφωνία στον αριθμό κυττάρων για το batch {batch_val}. Αυτό δεν θα έπρεπε να συμβαίνει.")
                # Αυτό μπορεί να συμβεί αν η σειρά των batches στην adata_list δεν είναι σωστή.
                # Προς το παρόν, συνεχίζουμε με την υπόθεση ότι είναι σωστή.

            final_x_scanorama[adata_concatenated.obs[batch_key] == batch_val] = adata_integrated_batch.obsm['X_scanorama']
            # current_idx += num_cells_in_batch # Αυτό δεν χρειάζεται αν αντιστοιχούμε με boolean indexing

        adata_integrated_final = adata_concatenated.copy() # Ξεκινάμε με ένα αντίγραφο του αρχικού συνενωμένου
        adata_integrated_final.obsm['X_scanorama'] = final_x_scanorama
    
    print(f"Το τελικό ενοποιημένο AnnData object δημιουργήθηκε με X_scanorama σχήματος: {adata_integrated_final.obsm['X_scanorama'].shape}")
    
    return adata_integrated_final


# Παράδειγμα χρήσης (μπορεί να αφαιρεθεί ή να γίνει comment out αργότερα)
if __name__ == '__main__':
    # Δημιουργία ψεύτικων δεδομένων με batches
    adata1 = sc.AnnData(pd.DataFrame(np.random.rand(50, 100), columns=[f'g{i}' for i in range(100)]))
    adata1.obs['batch_upload'] = '0'
    adata2 = sc.AnnData(pd.DataFrame(np.random.rand(60, 100), columns=[f'g{i}' for i in range(100)]))
    adata2.obs['batch_upload'] = '1'
    
    # Προσομοιώνουμε ένα ήδη συνενωμένο adata που θα έδινε ο χρήστης ή θα προέκυπτε από τη φόρτωση
    adata_concat_test = sc.AnnData.concatenate(adata1, adata2, batch_key='batch_original_upload')
    # Η συνάρτησή μας περιμένει η στήλη batch_key να είναι κατηγορική
    adata_concat_test.obs['batch_upload'] = adata_concat_test.obs['batch_original_upload'].astype('category')

    print("--- Δοκιμή της συνάρτησης run_scanorama_integration ---")
    adata_integrated = run_scanorama_integration(adata_concat_test, batch_key='batch_upload', dimred=30)
    
    print("--- Η ενοποίηση ολοκληρώθηκε για το δοκιμαστικό AnnData ---")
    print(f"Τελικό σχήμα: {adata_integrated.shape}")
    if 'X_scanorama' in adata_integrated.obsm:
        print(f"Διαθέσιμα Scanorama embeddings: {adata_integrated.obsm['X_scanorama'].shape}")
    else:
        print("!!! Δεν βρέθηκαν Scanorama embeddings στο τελικό αντικείμενο.") 