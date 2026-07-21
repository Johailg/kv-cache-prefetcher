import numpy as np, matplotlib.pyplot as plt
for tag in ("dialogue", "fiction", "math", "news", "code"):
    z = np.load(f"extracted_data/harvest_{tag}.npz")
    pos = np.where(z["clusters"] == 16)[1]
    plt.hist(pos, bins=50, alpha=.5, label=tag, density=True)
plt.legend(); plt.xlabel("window position of cluster-16 keys"); plt.show()