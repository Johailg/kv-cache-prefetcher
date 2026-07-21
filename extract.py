from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import random
from sklearn.cluster import KMeans
import os
from scipy.spatial.distance import pdist
import pickle
import collections
import matplotlib.pyplot as plt


# Experimental Constants
LAYER_IDX = 13
HEAD_IDX = 2
NUM_CLUSTERS = 64
MAX_POOL_SIZE = 50000
OUTPUT_DIR = "./extracted_data"
SEQ_LEN = 500

def load_model(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer

def load_text(path):
    with open(path, encoding = "utf-8") as f:
        return [ln for ln in f.read().split("\n") if ln.strip()]

def get_token_stream(tokenizer, text_lines):
    full_text = "\n".join(text_lines)
    return tokenizer(full_text, return_tensors='pt').input_ids[0]
    
def window_tokens(token_stream, L=SEQ_LEN):
    chunked_sequences = []
    for i in range(0, len(token_stream) - L + 1, L):
        chunk = token_stream[i : i + L].unsqueeze(0)
        chunked_sequences.append(chunk)
        
    return chunked_sequences

def collect_keys(model, chunks, pool_size = MAX_POOL_SIZE):
    pool = []         
    seen = 0          
    for input_ids in chunks:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache = True)
            
        keys = out.past_key_values.layers[LAYER_IDX].keys[0, HEAD_IDX].cpu().numpy()
        keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)

        for key in keys:
            seen += 1
            if len(pool) < pool_size:
                pool.append(key)
            else: 
                r = random.random()
                if r > pool_size / seen:
                    evict_idx = random.randint(0, pool_size-1)
                    pool[evict_idx] = key

    return np.array(pool)

def fit_centroids(pool, num_clusters = NUM_CLUSTERS):
    kmeans = KMeans(n_clusters = num_clusters, n_init = 'auto', random_state = 42)
    kmeans.fit(pool)
    return kmeans

def assign_clusters(model, chunks, kmeans):
    ret = []
    residuals = []
    for input_ids in chunks:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache = True)

        keys = out.past_key_values.layers[LAYER_IDX].keys[0, HEAD_IDX].cpu().numpy()
        keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
        
        dists = kmeans.transform(keys)
        cluster_ids = dists.argmin(axis = 1)
        residuals.append(dists.min(axis = 1))

        token_ids = input_ids[0].cpu().numpy()
        ret.append((token_ids, cluster_ids))

    d_val = float(np.concatenate(residuals).mean())
    res = np.concatenate(residuals)
    return ret, d_val, res

def design_b(assigned_pairs, tokenizer, token_str):
    target_id = tokenizer(token_str)['input_ids'][0]
    decoded = tokenizer.decode([target_id])
    pos_list, clu_list = [], []

    for token_ids, cluster_ids in assigned_pairs:
        hits = np.where(token_ids == target_id)[0]
        for pos in hits:
            pos_list.append(int(pos))
            clu_list.append(int(cluster_ids[pos]))

    if not clu_list:
        print(f"'{decoded}' (id {target_id}): not found.")
        return
        
    counter = collections.Counter(clu_list)
    modal_cluster, modal_n = counter.most_common(1)[0]
    purity = modal_n / len(clu_list)
    corr = np.corrcoef(pos_list, clu_list)[0, 1] if len(set(pos_list)) > 1 else float('nan')
    print(f"Token '{decoded}' (id {target_id}) | occurrences: {len(clu_list)}")
    print(f"  Modal purity:           {purity*100:.1f}%  (high = content survives)")
    print(f"  Position-cluster corr:  {corr:+.3f}  (near +/-1 = position dominates)")
    print(f"  Top 5 clusters:         {counter.most_common(5)}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, tokenizer = load_model("models\pythia-410m")
    
    # Wiki processing
    text = load_text("datasets\wiki_val.txt")[:100]
    wiki_stream = get_token_stream(tokenizer, text)
    wiki_chunks = window_tokens(wiki_stream)
    
    pool = collect_keys(model, wiki_chunks)
    kmeans = fit_centroids(pool)
    assigned, d_val, wiki_res = assign_clusters(model, wiki_chunks, kmeans)
    
    with open(os.path.join(OUTPUT_DIR, "assigned.pkl"), "wb") as f:
        pickle.dump(assigned, f)
    np.save(os.path.join(OUTPUT_DIR, "wiki_centroids.npy"), kmeans.cluster_centers_)
    
    # Code processing
    code_text = load_text("datasets\code_val.txt")[:700]
    code_stream = get_token_stream(tokenizer, code_text)
    code_chunks = window_tokens(code_stream)
    
    code_assigned, d_code, code_res = assign_clusters(model, code_chunks, kmeans)

    np.savez(os.path.join(OUTPUT_DIR, "gate1.npz"),
         wiki_res=wiki_res, code_res=code_res,
         centroids=kmeans.cluster_centers_,
         d_val=d_val, d_code=d_code, D=pdist(kmeans.cluster_centers_).mean())
    with open(os.path.join(OUTPUT_DIR, "code_assigned.pkl"), "wb") as f:
        pickle.dump(code_assigned, f)


    D = pdist(kmeans.cluster_centers_).mean()
    print(f"d_val:          {d_val:.4f}")
    print(f"d_code:         {d_code:.4f}")
    print(f"d_code / D:     {d_code/D:.3f}   (kill > 0.40)")
    print(f"d_code / d_val: {d_code/d_val:.3f}   (kill > 1.5)")