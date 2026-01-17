import faiss
import numpy as np

def rank_resumes(jd_vector, resume_vectors, resume_names, top_k=10):
    dim = len(jd_vector)
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(resume_vectors))
    D, I = index.search(np.array([jd_vector]), top_k)

    results = []
    for rank, idx in enumerate(I[0]):
        results.append((rank + 1, resume_names[idx], float(D[0][rank])))
    return results
