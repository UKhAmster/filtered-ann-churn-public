import hnswlib
import numpy as np

rng = np.random.default_rng(0)
data = rng.random((1000, 32), dtype=np.float32)
index = hnswlib.Index(space="l2", dim=32)
index.init_index(max_elements=2000, ef_construction=100, M=16)
index.add_items(data, np.arange(1000))
index.set_ef(50)
labels, _ = index.knn_query(data[:5], k=10)
print("smoke ok", labels.shape)
