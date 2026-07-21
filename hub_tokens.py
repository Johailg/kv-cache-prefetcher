import pickle, numpy as np
wiki_res = np.load(r"extracted_data\gate1.npz")["wiki_res"]
with open(r"extracted_data\assigned.pkl", "rb") as f:
    assigned = pickle.load(f)
clu = np.concatenate([c for _, c in assigned])
print((wiki_res < 0.1).mean())          # sanity: does 0.1 isolate the bump?
print((clu[wiki_res < 0.1] == 16).mean())