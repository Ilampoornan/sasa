#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


@dataclass
class SizeStats:
    k: int
    min: int
    p01: int
    median: int
    p99: int
    max: int
    frac_in_range: float
    n_in_range: int


def _quantile_int(x: np.ndarray, q: float) -> int:
    return int(np.quantile(x.astype(np.float64), q))


def summarize_sizes(sizes: np.ndarray, lo: int, hi: int) -> SizeStats:
    sizes = sizes.astype(np.int64, copy=False)
    in_range = (sizes >= lo) & (sizes <= hi)
    return SizeStats(
        k=int(sizes.size),
        min=int(sizes.min()),
        p01=_quantile_int(sizes, 0.01),
        median=int(np.median(sizes)),
        p99=_quantile_int(sizes, 0.99),
        max=int(sizes.max()),
        frac_in_range=float(in_range.mean()),
        n_in_range=int(in_range.sum()),
    )


def effective_rank(
    X: np.ndarray, *, center: bool = True, eps: float = 1e-12
) -> Tuple[float, float, list[float]]:
    """
    Effective rank per Roy & Vetterli: eRank = exp(H(p)),
    p_i = s_i / sum_j s_j, where s_i are singular values.

    We compute s via eigenvalues of Gram matrix (m x m) for speed when m << d.

    Returns: (effective_rank, total_variance, top_singular_values)
    """
    m = X.shape[0]
    if m <= 1:
        return 1.0, 0.0, []

    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    G = Xc @ Xc.T  # (m, m)
    # Numerical issues can create tiny negatives
    evals = np.linalg.eigvalsh(G).astype(np.float64, copy=False)
    evals = np.clip(evals, 0.0, None)
    total_var = float(evals.sum())

    if total_var <= eps:
        return 1.0, total_var, []

    s = np.sqrt(evals)
    s_sum = float(s.sum())
    if s_sum <= eps:
        return 1.0, total_var, []

    p = s / s_sum
    # guard: p can contain zeros
    p_nz = p[p > 0]
    H = -float(np.sum(p_nz * np.log(p_nz)))
    erank = float(np.exp(H))

    top_s = np.sort(s)[::-1][:5].tolist()
    return erank, total_var, [float(v) for v in top_s]


def pca_dimension(
    X: np.ndarray, *, var_threshold: float = 0.9, center: bool = True, eps: float = 1e-12
) -> int:
    """
    PCA dimension = smallest k such that top-k eigenvalues explain >= var_threshold
    of total variance.

    We compute eigenvalues via Gram matrix (m x m) when m << d.
    """
    vt = float(var_threshold)
    if not (0.0 < vt <= 1.0):
        raise ValueError(f"var_threshold must be in (0, 1], got {var_threshold}")

    m = X.shape[0]
    if m <= 1:
        return 1

    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    G = Xc @ Xc.T
    evals = np.linalg.eigvalsh(G).astype(np.float64, copy=False)
    evals = np.clip(evals, 0.0, None)
    total = float(evals.sum())
    if total <= eps:
        return 1

    ev = np.sort(evals)[::-1]
    c = np.cumsum(ev) / total
    k = int(np.searchsorted(c, vt, side="left") + 1)
    return int(min(max(k, 1), int(ev.size)))

def iter_edges_ncol(path: Path) -> Iterable[Tuple[int, int, float]]:
    with path.open("r") as f:
        for line in f:
            a, b, w = line.split()
            yield int(a), int(b), float(w)


def _set_hf_cache_defaults(workspace_root: Path) -> None:
    """
    Avoid writing large caches to $HOME by defaulting to workspace cache.
    """
    cache_root = workspace_root / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "hf_datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hf_transformers"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hf_hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _slug(s: str) -> str:
    return "".join(c if (c.isalnum() or c in {"-", "_", "."}) else "_" for c in s)


def _infer_cuda_index(device: str | None) -> int | None:
    """
    Extract CUDA device index from strings like 'cuda:4'. Return None if not specified.
    """
    if not device:
        return None
    s = str(device).strip().lower()
    if not s.startswith("cuda"):
        return None
    if ":" not in s:
        return None
    try:
        return int(s.split(":", 1)[1])
    except Exception:
        return None


def _build_knn_edges_faiss_hnsw(
    *,
    X: np.ndarray,
    edge_path: Path,
    k_nn: int,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
    min_sim: float,
) -> None:
    import faiss

    n, d = X.shape
    index = faiss.IndexHNSWFlat(d, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(ef_construction)
    index.add(X)
    index.hnsw.efSearch = int(ef_search)

    S, I = index.search(X, int(k_nn) + 1)
    I = I[:, 1:]
    S = np.clip(S[:, 1:], 0.0, 1.0)

    ms = float(min_sim)
    with edge_path.open("w", buffering=1024 * 1024) as f:
        for i in range(n):
            for j, w in zip(I[i].tolist(), S[i].tolist()):
                if j != i and w > ms:
                    f.write(f"{i} {j} {w}\n")


def _build_knn_edges_torch_exact(
    *,
    X: np.ndarray,
    edge_path: Path,
    k_nn: int,
    device: str,
    batch: int,
    min_sim: float,
) -> None:
    """
    Exact cosine top-k using torch matmul in chunks.

    X must already be row-normalized (cosine == inner product).
    Writes a directed edge list; downstream igraph loads as undirected and simplifies.
    """
    import torch

    n, d = X.shape
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError(f"Requested device={device} but CUDA is not available.")

    dev = torch.device(device)
    X_t = torch.from_numpy(X).to(device=dev, dtype=torch.float16)
    k = int(k_nn)
    bsz = max(1, int(batch))
    ms = float(min_sim)

    with edge_path.open("w", buffering=1024 * 1024) as f:
        for start in range(0, n, bsz):
            end = min(n, start + bsz)
            Q = X_t[start:end]  # (b, d)
            # (b, n) similarities
            S = (Q @ X_t.T).to(dtype=torch.float32)
            # Mask self
            rows = torch.arange(end - start, device=dev)
            cols = torch.arange(start, end, device=dev)
            S[rows, cols] = float("-inf")

            vals, idx = torch.topk(S, k=k, dim=1, largest=True, sorted=False)
            idx_cpu = idx.to("cpu").numpy()
            vals_cpu = vals.to("cpu").numpy()
            for bi in range(end - start):
                i = start + bi
                for j, w in zip(idx_cpu[bi].tolist(), vals_cpu[bi].tolist()):
                    if w > ms:
                        f.write(f"{i} {int(j)} {float(w)}\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--pt-path",
        default=None,
        help="Path to a local SAE checkpoint (.pt) containing state_dict['W_dec'].",
    )
    src.add_argument(
        "--sae-release",
        default="gpt2-small-res-jb",
        help="SAE Lens release name (e.g. gpt2-small-res-jb).",
    )
    ap.add_argument(
        "--hook",
        default="blocks.7.hook_resid_pre",
        help="Hook name to load from SAE Lens (e.g. blocks.7.hook_resid_pre).",
    )
    ap.add_argument(
        "--device",
        default=None,
        help='Device for SAE Lens loading ("cpu", "cuda"). Default: auto.',
    )
    ap.add_argument(
        "--clusterer",
        type=str,
        default="leiden",
        choices=["leiden", "spectral"],
        help="Clustering algorithm: Leiden on kNN graph, or spectral embedding + k-means.",
    )
    ap.add_argument("--k-nn", type=int, default=100)
    ap.add_argument(
        "--knn-backend",
        type=str,
        default="faiss_hnsw",
        choices=["faiss_hnsw", "torch_exact"],
        help="How to build kNN graph. torch_exact computes exact top-k on the given device.",
    )
    ap.add_argument(
        "--knn-device",
        type=str,
        default=None,
        help='Device for torch_exact kNN (e.g. "cuda:4"). Default: infer from --device.',
    )
    ap.add_argument(
        "--torch-batch",
        type=int,
        default=1024,
        help="Batch size (queries) for torch_exact kNN.",
    )
    ap.add_argument(
        "--min-sim",
        type=float,
        default=0.0,
        help="Drop edges with cosine similarity <= min_sim.",
    )
    ap.add_argument(
        "--spectral-threshold",
        type=float,
        default=0.5,
        help="Additional similarity threshold for spectral clustering (drop edges < this).",
    )
    ap.add_argument(
        "--n-clusters",
        type=int,
        default=3072,
        help="Number of clusters for spectral clustering (k-means).",
    )
    ap.add_argument(
        "--spectral-n-components",
        type=int,
        default=256,
        help="Embedding dimension for spectral clustering (does not need to equal n_clusters).",
    )
    ap.add_argument(
        "--spectral-drop-first",
        action="store_true",
        help="Drop the first (near-constant) eigenvector in spectral embedding.",
    )
    ap.add_argument(
        "--spectral-eig-tol",
        type=float,
        default=1e-3,
        help="Tolerance for sparse eigensolver (smaller is slower but more accurate).",
    )
    ap.add_argument(
        "--spectral-kmeans-batch",
        type=int,
        default=4096,
        help="MiniBatchKMeans batch size for spectral clustering.",
    )
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=80)
    ap.add_argument("--ef-search", type=int, default=120)
    ap.add_argument(
        "--center",
        action="store_true",
        help="Center vectors within each cluster before eRank",
    )
    ap.add_argument(
        "--pca-var",
        type=float,
        default=0.9,
        help="Variance fraction for PCA dimension per cluster (e.g. 0.9 for 90%%).",
    )
    ap.add_argument(
        "--skip-erank",
        action="store_true",
        help="Save membership.npy and stop (no effective-rank computation).",
    )
    ap.add_argument(
        "--compute-erank-only",
        action="store_true",
        help="Load membership.npy from --out-dir and compute effective ranks only.",
    )
    ap.add_argument("--lo", type=int, default=8, help="Lower desired cluster size")
    ap.add_argument("--hi", type=int, default=16, help="Upper desired cluster size")
    ap.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Skip tuning and use this Leiden resolution",
    )
    ap.add_argument(
        "--tune",
        action="store_true",
        help="Tune Leiden resolution for size range objective",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--coarse-iters", type=int, default=2)
    ap.add_argument(
        "--max-coarse-pow2",
        type=int,
        default=24,
        help="Max exponent for resolution sweep (2^exp)",
    )
    ap.add_argument(
        "--out-dir", default="manifold-sae/cluster_results/leiden_wdec_knn100"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    membership_path = out_dir / "membership.npy"
    run_meta_path = out_dir / "run_meta.json"

    # Ensure repo root is on sys.path so we can import the local `saes` stubs
    # (checkpoint unpickling references `saes.config.*`).
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    _set_hf_cache_defaults(repo_root)

    # Thread caps (avoid oversubscription)
    os.environ.setdefault("OMP_NUM_THREADS", "32")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
    os.environ.setdefault("MKL_NUM_THREADS", "32")

    import torch
    import igraph as ig
    import leidenalg
    from contextlib import nullcontext

    source_meta: dict[str, object]
    if args.pt_path is not None:
        pt_path = Path(args.pt_path)
        # Torch >=2.6 defaults weights_only=True, and requires allowlisting the config class
        # embedded in the checkpoint. We use the safe_globals context manager when available.
        try:
            from saes.config import LanguageModelSAERunnerConfig

            try:
                ctx = torch.serialization.safe_globals([LanguageModelSAERunnerConfig])
            except Exception:
                # Older torch versions may not have the context manager.
                torch.serialization.add_safe_globals([LanguageModelSAERunnerConfig])
                ctx = nullcontext()

            with ctx:
                obj = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        except Exception:
            # If allowlisting fails for any reason, fall back to normal torch.load.
            # This is safe here because this checkpoint is local/trusted.
            obj = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        W_dec = state["W_dec"]
        source_meta = {"source": "pt", "pt_path": str(pt_path)}
        cache_tag = _slug(pt_path.stem)
    else:
        # SAE Lens pretrained SAE
        from sae_lens import SAE  # noqa: WPS433

        load_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        sae = SAE.from_pretrained(str(args.sae_release), str(args.hook), device=load_device)
        W_dec = sae.W_dec.detach().cpu()
        source_meta = {"source": "sae_lens", "sae_release": args.sae_release, "hook": args.hook}
        cache_tag = f"{_slug(str(args.sae_release))}_{_slug(str(args.hook))}"

    X = W_dec.detach().cpu().to(torch.float32)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-12)  # cosine
    X = X.numpy().astype("float32", copy=False)

    n, d = X.shape

    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", "/tmp"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    backend_tag = str(args.knn_backend)
    edge_path = (
        cache_dir
        / f"{cache_tag}_{backend_tag}_wdec_cosine_knn{int(args.k_nn)}_minsim{float(args.min_sim):.3g}.ncol"
    )

    # Build kNN edges if missing
    if not edge_path.exists():
        if args.knn_backend == "faiss_hnsw":
            print(
                f"Building cosine kNN graph (FAISS HNSW, CPU): k={int(args.k_nn)} efSearch={int(args.ef_search)}"
            )
            _build_knn_edges_faiss_hnsw(
                X=X,
                edge_path=edge_path,
                k_nn=int(args.k_nn),
                hnsw_m=int(args.hnsw_m),
                ef_construction=int(args.ef_construction),
                ef_search=int(args.ef_search),
                min_sim=float(args.min_sim),
            )
        else:
            knn_device = args.knn_device or args.device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            print(
                f"Building cosine kNN graph (torch_exact): k={int(args.k_nn)} device={knn_device} batch={int(args.torch_batch)}"
            )
            _build_knn_edges_torch_exact(
                X=X,
                edge_path=edge_path,
                k_nn=int(args.k_nn),
                device=str(knn_device),
                batch=int(args.torch_batch),
                min_sim=float(args.min_sim),
            )
        print("Wrote edge list:", edge_path)
    else:
        print("Using cached edge list:", edge_path)

    if args.compute_erank_only:
        if not membership_path.exists():
            raise FileNotFoundError(f"Missing membership file: {membership_path}")
        membership = np.load(membership_path)
        chosen_r = float("nan")
        g = None  # type: ignore[assignment]
        print("Loaded membership:", membership_path, "K=", int(membership.max() + 1))
        # Jump to eRank computation section below
        part = None  # type: ignore[assignment]
        if run_meta_path.exists():
            try:
                with run_meta_path.open("r") as f:
                    run_meta = json.load(f)
                chosen_r = float(run_meta.get("resolution", float("nan")))
            except Exception:
                pass
    if not args.compute_erank_only and args.clusterer == "leiden":
        print("Reading graph into igraph...")
        g = ig.Graph.Read_Ncol(str(edge_path), directed=False, weights=True)
        if g.vcount() < n:
            g.add_vertices(n - g.vcount())
        g.simplify(combine_edges={"weight": "max"})
        print("Graph:", g.vcount(), "vertices,", g.ecount(), "edges")

    if not args.compute_erank_only and args.clusterer == "leiden":

        def leiden_sizes(resolution: float, iters: int) -> np.ndarray:
            part0 = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=float(resolution),
                n_iterations=int(iters),
                seed=int(args.seed),
            )
            return np.array(part0.sizes(), dtype=np.int32)

        chosen_r: float
        if args.resolution is not None:
            chosen_r = float(args.resolution)
        elif args.tune:
            print(f"Tuning resolution for cluster sizes in [{args.lo}, {args.hi}]...")

            # Exponential search until median <= hi
            r = 8.0
            st = summarize_sizes(
                leiden_sizes(r, iters=args.coarse_iters), args.lo, args.hi
            )
            print(
                f"  r={r:.3g} -> K={st.k} median={st.median} frac_in_range={st.frac_in_range:.3f}"
            )
            r_prev, st_prev = r, st
            while st.median > args.hi and r < 1e7:
                r_prev, st_prev = r, st
                r *= 2.0
                st = summarize_sizes(
                    leiden_sizes(r, iters=args.coarse_iters), args.lo, args.hi
                )
                print(
                    f"  r={r:.3g} -> K={st.k} median={st.median} frac_in_range={st.frac_in_range:.3f}"
                )

            # If we never reached median <= hi, just pick the largest tried
            if st.median > args.hi:
                chosen_r = float(r)
            else:
                # Search between r_prev and r in log-space
                logs = np.linspace(math.log(r_prev), math.log(r), 13)
                best = None
                for lg in logs:
                    rr = float(math.exp(float(lg)))
                    st_rr = summarize_sizes(
                        leiden_sizes(rr, iters=args.coarse_iters), args.lo, args.hi
                    )
                    in_band = args.lo <= st_rr.median <= args.hi
                    score = (
                        1 if in_band else 0,
                        st_rr.frac_in_range,
                        -abs(st_rr.median - (args.lo + args.hi) / 2.0),
                    )
                    print(
                        f"  cand r={rr:.4g} -> K={st_rr.k} median={st_rr.median} frac_in_range={st_rr.frac_in_range:.3f}"
                    )
                    if best is None or score > best[0]:
                        best = (score, rr, st_rr)
                assert best is not None
                chosen_r = float(best[1])
                print("Chosen resolution:", chosen_r, "stats:", asdict(best[2]))
        else:
            raise SystemExit("Provide --resolution or pass --tune.")

        print("Running final Leiden partition...")
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(chosen_r),
            n_iterations=int(args.iters),
            seed=int(args.seed),
        )

        membership = np.array(part.membership, dtype=np.int32)
        np.save(membership_path, membership)
        K = int(membership.max() + 1) if membership.size else 0
        print("Saved membership:", membership_path, "K=", K)
        # Save lightweight run metadata for later eRank-only runs.
        with run_meta_path.open("w") as f:
            json.dump(
                {
                    **source_meta,
                    "clusterer": "leiden",
                    "k_nn": int(args.k_nn),
                    "ef_search": int(args.ef_search),
                    "resolution": float(chosen_r),
                    "leiden_iters": int(args.iters),
                    "edge_path": str(edge_path),
                    "n_vertices": int(n),
                    "dim": int(d),
                    "n_edges": int(g.ecount()) if g is not None else None,
                    "size_target_lo": int(args.lo),
                    "size_target_hi": int(args.hi),
                },
                f,
            )
        print("Saved run meta:", run_meta_path)
        if args.skip_erank:
            return
    elif not args.compute_erank_only and args.clusterer == "spectral":
        # Spectral clustering on sparse kNN graph: normalized Laplacian eigenmaps + MiniBatchKMeans.
        import scipy.sparse as sp  # noqa: WPS433
        import scipy.sparse.linalg as spla  # noqa: WPS433
        from sklearn.cluster import MiniBatchKMeans  # noqa: WPS433

        thr = float(args.spectral_threshold)
        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"--spectral-threshold must be in [0,1], got {thr}")

        print("Reading edge list into scipy sparse...")
        rows = []
        cols = []
        vals = []
        with edge_path.open("r") as f:
            for line in f:
                a, b, w = line.split()
                wv = float(w)
                if wv < thr:
                    continue
                rows.append(int(a))
                cols.append(int(b))
                vals.append(wv)

        if not rows:
            raise RuntimeError(
                f"No edges left after thresholding at {thr}. Lower --spectral-threshold."
            )

        W = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(n, n))
        # Symmetrize (keep max weight).
        W = W.maximum(W.T)
        W.eliminate_zeros()
        nnz = int(W.nnz)
        print(f"Sparse adjacency: nnz={nnz} (after threshold+sym)")

        deg = np.asarray(W.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
        # Avoid divide-by-zero for isolated nodes (deg==0): set invsqrt to 0.
        inv_sqrt = np.zeros_like(deg)
        nz = deg > 0
        inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        D_inv_sqrt = sp.diags(inv_sqrt.astype(np.float32), offsets=0, shape=(n, n), format="csr")

        # Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
        print("Building normalized Laplacian...")
        S = (D_inv_sqrt @ W) @ D_inv_sqrt
        L = sp.eye(n, format="csr", dtype=np.float32) - S

        n_components = int(args.spectral_n_components)
        if n_components < 2:
            raise ValueError("--spectral-n-components must be >= 2")

        # If dropping first eigenvector, compute one extra.
        k_eigs = n_components + (1 if bool(args.spectral_drop_first) else 0)
        k_eigs = min(k_eigs, n - 2)
        print(f"Computing {k_eigs} smallest eigenvectors (eigsh)...")
        # Smallest magnitude eigenvalues of L correspond to connected components / smoothest modes.
        evals, evecs = spla.eigsh(
            L,
            k=k_eigs,
            which="SM",
            tol=float(args.spectral_eig_tol),
        )
        order = np.argsort(evals)
        evecs = evecs[:, order]

        if bool(args.spectral_drop_first) and evecs.shape[1] > 1:
            evecs = evecs[:, 1 : (1 + n_components)]
        else:
            evecs = evecs[:, :n_components]

        # Row-normalize embedding (common in spectral clustering).
        emb = evecs.astype(np.float32, copy=False)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms

        n_clusters = int(args.n_clusters)
        if not (2 <= n_clusters <= n):
            raise ValueError(f"--n-clusters must be in [2, {n}], got {n_clusters}")

        print(f"Running MiniBatchKMeans: n_clusters={n_clusters} ...")
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=int(args.spectral_kmeans_batch),
            n_init="auto",
            random_state=int(args.seed),
            verbose=0,
        )
        membership = km.fit_predict(emb).astype(np.int32, copy=False)
        np.save(membership_path, membership)
        K = int(membership.max() + 1) if membership.size else 0
        chosen_r = float("nan")
        g = None  # type: ignore[assignment]
        print("Saved membership:", membership_path, "K=", K)
        with run_meta_path.open("w") as f:
            json.dump(
                {
                    **source_meta,
                    "clusterer": "spectral",
                    "edge_path": str(edge_path),
                    "n_vertices": int(n),
                    "dim": int(d),
                    "k_nn": int(args.k_nn),
                    "min_sim": float(args.min_sim),
                    "spectral_threshold": float(thr),
                    "spectral_n_components": int(n_components),
                    "n_clusters": int(n_clusters),
                    "spectral_eig_tol": float(args.spectral_eig_tol),
                    "spectral_drop_first": bool(args.spectral_drop_first),
                    "nnz": int(nnz),
                },
                f,
            )
        print("Saved run meta:", run_meta_path)
        if args.skip_erank:
            return
    else:
        K = int(membership.max() + 1) if membership.size else 0

    # Group members by cluster id efficiently
    order = np.argsort(membership, kind="stable")
    sorted_labels = membership[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    groups = np.split(order, boundaries)

    # Compute effective rank per cluster
    cluster_rows = []
    for cid, idxs in enumerate(groups):
        Xg = X[idxs]
        er, var, top_s = effective_rank(
            Xg.astype(np.float64, copy=False), center=bool(args.center)
        )
        pca_k = pca_dimension(
            Xg.astype(np.float64, copy=False),
            var_threshold=float(args.pca_var),
            center=bool(args.center),
        )
        cluster_rows.append(
            {
                "cluster_id": int(cid),
                "size": int(idxs.size),
                "effective_rank": float(er),
                "pca_dim": int(pca_k),
                "size_over_pca_dim": float(float(idxs.size) / float(pca_k))
                if pca_k > 0
                else float("nan"),
                "total_variance": float(var),
                "top_singular_values": top_s,
            }
        )
        if (cid + 1) % 1000 == 0:
            print(f"Computed effective ranks for {cid+1}/{len(groups)} clusters...")

    sizes = np.array([r["size"] for r in cluster_rows], dtype=np.int32)
    stats = summarize_sizes(sizes, args.lo, args.hi)

    # membership already saved to membership_path (or loaded in --compute-erank-only mode)

    meta = {
        **source_meta,
        "n_vectors": int(n),
        "dim": int(d),
        "k_nn": int(args.k_nn),
        "ef_search": int(args.ef_search),
        "resolution": float(chosen_r),
        "leiden_iters": int(args.iters),
        "size_target_lo": int(args.lo),
        "size_target_hi": int(args.hi),
        "center_for_erank": bool(args.center),
        "pca_var_threshold": float(args.pca_var),
        "size_stats": asdict(stats),
        "edge_path": str(edge_path),
        "n_edges": int(g.ecount()) if g is not None else None,
    }

    with (out_dir / "cluster_stats.json").open("w") as f:
        json.dump({"meta": meta, "clusters": cluster_rows}, f)

    print("Saved:")
    print("  ", membership_path)
    print("  ", out_dir / "cluster_stats.json")
    print("Resolution:", chosen_r)
    print("Size stats:", meta["size_stats"])


if __name__ == "__main__":
    main()
