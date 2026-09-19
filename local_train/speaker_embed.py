"""Шаг 5: голосовые эмбеддинги (WeSpeaker ResNet34-LM, onnx fp32) + кластеризация спикеров.

Валидация: половинки одного файла должны быть близки; наши 4 рефа — в разных кластерах.
Пилот: выборка из ru_mfa_manifest → эмбеддинги → agglomerative(cossine) → распределение кластеров.
"""
import argparse
import json
import os
import sys

import numpy as np

MODEL_DIR = r"C:\Users\ARTEM\.cache\huggingface\hub\models--onnx-community--wespeaker-voxceleb-resnet34-LM\snapshots\6a61a1833ff2583aabeba044f5c8221f00b67ceb\onnx\model.onnx"
_SESS = None


def _sess():
    global _SESS
    if _SESS is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("SPK_THREADS", "2"))
        opts.inter_op_num_threads = 1
        prov = os.environ.get("SPK_PROVIDER", "cpu").lower()
        if prov == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = [("CUDAExecutionProvider", {"device_id": int(os.environ.get("SPK_DEVICE", "0"))}),
                         "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        _SESS = ort.InferenceSession(MODEL_DIR, sess_options=opts, providers=providers)
    return _SESS


def load16k(path):
    import soundfile as sf
    import librosa

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x


def fbank80(x):
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    import torchaudio.compliance.kaldi as kaldi

    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
    f = kaldi.fbank(t, num_mel_bins=80, frame_length=25, frame_shift=10,
                    sample_frequency=16000, dither=0.0, energy_floor=0.0,
                    window_type="hamming", snip_edges=True, use_energy=False)
    f = f - f.mean(dim=0, keepdim=True)  # CMN
    return f.numpy()


def embed(x16k):
    f = fbank80(x16k)
    out = _sess().run(None, {"input_features": f[None, :, :]})[0]
    e = np.asarray(out).reshape(-1).astype(np.float32)
    n = np.linalg.norm(e) + 1e-9
    return e / n


def embed_path(path):
    return embed(load16k(path))


def self_check():
    L = r"G:\AI\AuK\local_tests"
    refs = {
        "user_male": os.path.join(L, "user_ref.wav"),
        "nat_female": os.path.join(L, "ref_ru.wav"),
        "alt_male": os.path.join(L, "ref_male2.wav"),
        "alt_female": os.path.join(L, "ref_female2.wav"),
    }
    print("== self-check: halves of same file vs different files ==")
    embs = {}
    for name, p in refs.items():
        x = load16k(p)
        half = len(x) // 2
        e1 = embed(x[:half])
        e2 = embed(x[half:])
        embs[name] = embed(x)
        print(f"{name}: half-similarity={float(np.dot(e1, e2)):.3f}")
    names = list(refs)
    print("cross similarities (should be lower):")
    for i in range(len(names)):
        row = []
        for j in range(len(names)):
            row.append(round(float(np.dot(embs[names[i]], embs[names[j]])), 2))
        print("  ", names[i], row)


def pilot(n=3000, seed=7):
    import random

    rng = random.Random(seed)
    man = r"D:\MediaForge_kyutai"
    src = r"G:\AI\kyutai-ru\data\ru_mfa_manifest.jsonl"
    picks = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            d = float(o.get("duration", 0))
            if 2.5 <= d <= 8.0:
                picks.append(o)
                if len(picks) >= n * 6:
                    break
    rng.shuffle(picks)
    picks = picks[:n]
    print(f"pilot: {len(picks)} clips selected")
    rows = []
    embs = []
    for i, o in enumerate(picks):
        path = o["path"].replace("X:\\MediaForge\\kyutai-ru-data\\", "G:\\AI\\kyutai-ru\\data\\")
        try:
            e = embed_path(path)
        except Exception as exc:
            continue
        embs.append(e)
        rows.append({"path": path, "duration": o.get("duration"), "transcript": o.get("transcript", "")[:80]})
        if (i + 1) % 500 == 0:
            print(f"  embedded {i + 1}/{len(picks)}", flush=True)
    E = np.stack(embs)
    print("embeddings:", E.shape)
    from sklearn.cluster import AgglomerativeClustering

    out = {}
    for th in (0.45, 0.55, 0.65):
        cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=th).fit(E)
        sizes = np.bincount(cl.labels_)
        big = sizes[sizes >= 5]
        out[str(th)] = {"clusters": int(cl.n_clusters_), "singletons": int((sizes == 1).sum()),
                        "clusters_ge5": int(len(big)), "top_sizes": sorted(sizes.tolist(), reverse=True)[:10]}
        print(f"th={th}: clusters={cl.n_clusters_} singletons={(sizes==1).sum()} top={sorted(sizes.tolist(), reverse=True)[:8]}")
        np.save(os.path.join(r"G:\AI\AuK\local_train\corpus", f"pilot_labels_th{th}.npy"), cl.labels_)
    json.dump({"rows": rows, "stats": out},
              open(r"G:\AI\AuK\local_train\corpus\pilot_speakers.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("PILOT_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "pilot"], default="check")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    if args.mode == "check":
        self_check()
    else:
        pilot(args.n)
