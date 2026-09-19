"""Докачка RESD_Annotated с resume (Range) — файл большой (390 МБ),HTTP рвётся."""
import os
import time
import urllib.request

BASE = "https://huggingface.co/datasets/Aniemore/RESD_Annotated/resolve/main/data"
OUT_DIR = r"G:\AI\_datasets\aniemore_resd\data"
FILES = ("train-00000-of-00001-1f5fe73d1293189c.parquet",
         "test-00000-of-00001-a2b788d59856c4ae.parquet")


def fetch_range(url, start, retries=5):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"})
            r = urllib.request.urlopen(req, timeout=120)
            return r.read()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in FILES:
        dst = os.path.join(OUT_DIR, f)
        url = f"{BASE}/{f}"
        # размер
        req = urllib.request.Request(url, method="HEAD")
        total = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
        print(f, "total", total, flush=True)
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        while have < total:
            chunk = fetch_range(url, have)
            if not chunk:
                raise RuntimeError("empty chunk")
            with open(dst, "ab") as fh:
                fh.write(chunk)
            have += len(chunk)
            print(f"  {have}/{total} ({100*have//total}%)", flush=True)
        print(f, "DONE", have, flush=True)


if __name__ == "__main__":
    main()
