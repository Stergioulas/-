import scanpy as sc
import os
import hdf5plugin # Παρόλο που δεν καλείται απευθείας, χρειάζεται για το adata.write_h5ad με συμπίεση

# Λίστα των δειγμάτων και των συνθηκών τους
samples_info = [
    {"gsm_id": "GSM4483339", "condition": "control"},
    {"gsm_id": "GSM4483340", "condition": "control"},
    {"gsm_id": "GSM4483366", "condition": "disease"},
    {"gsm_id": "GSM4483367", "condition": "disease"},
]

# Βασικοί φάκελοι
base_path = "."
raw_data_dir = os.path.join(base_path, "data", "RAW")
h5ad_output_dir = os.path.join(base_path, "data", "h5ad")

# Δημιουργία του φακέλου εξόδου αν δεν υπάρχει
if not os.path.exists(h5ad_output_dir):
    os.makedirs(h5ad_output_dir)
    print(f"Δημιουργήθηκε ο φάκελος: {h5ad_output_dir}")

print(f"Έναρξη επεξεργασίας {len(samples_info)} δειγμάτων...")

for sample in samples_info:
    gsm_id = sample["gsm_id"]
    condition = sample["condition"]
    
    sample_raw_path = os.path.join(raw_data_dir, gsm_id)
    output_h5ad_path = os.path.join(h5ad_output_dir, f"{gsm_id}.h5ad")
    
    print(f"\nΕπεξεργασία δείγματος: {gsm_id} (Condition: {condition})")
    print(f"Αναζήτηση ακατέργαστων δεδομένων στον φάκελο: {sample_raw_path}")
    
    if not os.path.exists(sample_raw_path):
        print(f"!!! Σφάλμα: Ο φάκελος {sample_raw_path} δεν βρέθηκε. Παρακαλώ ελέγξτε την οργάνωση των αρχείων.")
        continue
        
    try:
        # Φόρτωση των δεδομένων 10x mtx
        adata = sc.read_10x_mtx(sample_raw_path, var_names='gene_symbols', cache=False) # cache=False για να είμαστε σίγουροι ότι διαβάζει πάντα τα αρχεία
        adata.obs["condition"] = condition
        print(f"  Φορτώθηκαν {adata.n_obs} κύτταρα και {adata.n_vars} γονίδια.")
        
        # Αποθήκευση του AnnData object
        adata.write_h5ad(output_h5ad_path, compression="gzip")
        print(f"  Το αρχείο AnnData αποθηκεύτηκε στο: {output_h5ad_path}")
        
    except Exception as e:
        print(f"!!! Σφάλμα κατά την επεξεργασία του δείγματος {gsm_id}: {e}")

print("\nΗ επεξεργασία όλων των δειγμάτων ολοκληρώθηκε.") 