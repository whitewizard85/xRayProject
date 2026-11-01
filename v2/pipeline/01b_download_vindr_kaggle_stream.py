"""
01b_download_vindr_kaggle_stream.py -- VinDr-CXR (competizione Kaggle VinBigData) SENZA scaricare lo zip.

Come funziona
-------------
1. UNA sola chiamata all'API Kaggle restituisce l'indirizzo (firmato, temporaneo) dello zip completo
   della competizione sullo storage di Kaggle.
2. Lo zip viene letto "a pezzi" con richieste HTTP Range direttamente dallo storage: prima l'indice
   (central directory), poi SOLO i membri train/*.dicom che servono.
3. Ogni DICOM viene decodificato in memoria e salvato direttamente nella cache 512 px: nessun DICOM e
   nessuno zip finiscono su disco.
Le richieste allo storage NON passano dall'API Kaggle (niente errori 429 per singolo file). Se l'indirizzo
firmato scade, ne viene chiesto uno nuovo (una chiamata API, con attesa automatica in caso di 429).

    python 01b_download_vindr_kaggle_stream.py                    # tutte le 15.000 immagini train
    python 01b_download_vindr_kaggle_stream.py --max-images 6000  # sottoinsieme casuale uniforme (seed 0)
    python 02_prepare_data.py --skip-nih                          # poi: manifest ed etichette

Rilanciabile: le immagini gia' in cache vengono saltate. Prerequisiti: token Kaggle (~/.kaggle/access_token
o KAGGLE_API_TOKEN) e Rules della competizione accettate.
"""
import io
import os
import sys
import time
import glob
import shutil
import zipfile
import argparse
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from common import load_config, shared_data_dir

COMP = "vinbigdata-chest-xray-abnormalities-detection"


# ---------------------------------------------------------------------------
# Indirizzo firmato dello zip completo (1 chiamata API)
# ---------------------------------------------------------------------------
class UrlProvider:
    def __init__(self, fetch_fn):
        self.fetch_fn, self.url, self.lock, self.version = fetch_fn, None, threading.Lock(), 0

    def get(self):
        with self.lock:
            if self.url is None:
                self._refresh()
            return self.url, self.version

    def invalidate(self, version):
        with self.lock:
            if version == self.version:   # un solo thread rinnova
                self._refresh()

    def _refresh(self):
        wait = 60
        while True:
            try:
                self.url = self.fetch_fn()
                self.version += 1
                print(f"[VinDr] nuovo indirizzo firmato ottenuto (v{self.version})", flush=True)
                return
            except Exception as e:
                msg = str(e)
                if "429" in msg:
                    print(f"[VinDr] API Kaggle: 429 Too Many Requests, attendo {wait // 60} min...", flush=True)
                    time.sleep(wait)
                    wait = min(wait * 2, 3600)
                    continue
                raise


def kaggle_zip_url():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as k:
        req = ApiDownloadDataFilesRequest()
        req.competition_name = COMP
        resp = k.competitions.competition_api_client.download_data_files(req)
        url = resp.url
        try:
            resp.close()
        except Exception:
            pass
    if not url or "api.kaggle.com" in url:
        raise RuntimeError(f"indirizzo di download inatteso: {url[:120] if url else url}")
    return url


# ---------------------------------------------------------------------------
# File remoto "seekable" via HTTP Range (con blocchi in cache)
# ---------------------------------------------------------------------------
class RemoteFile(io.RawIOBase):
    def __init__(self, urls, size=None, block=4 << 20, session=None):
        self.urls, self.block, self.pos = urls, block, 0
        self.sess = session or requests.Session()
        self.cache = {}
        self.size = size if size is not None else self._probe_size()

    def _get(self, start, end):
        last = None
        for attempt in range(8):
            url, ver = self.urls.get()
            try:
                r = self.sess.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            except requests.RequestException as e:
                last = repr(e)
                time.sleep(min(2 ** attempt, 60))
                continue
            last = r.status_code
            if r.status_code == 206:
                return r.content
            if r.status_code == 200 and start == 0:
                return r.content[: end + 1]
            if r.status_code in (400, 401, 403, 410):      # firma scaduta
                self.urls.invalidate(ver)
                continue
            time.sleep(min(2 ** attempt, 60))
        raise IOError(f"lettura remota fallita bytes={start}-{end} (ultimo esito {last})")

    def _probe_size(self):
        for attempt in range(8):
            url, ver = self.urls.get()
            r = self.sess.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
            if r.status_code == 206 and "Content-Range" in r.headers:
                return int(r.headers["Content-Range"].split("/")[-1])
            if r.status_code in (400, 401, 403, 410):
                self.urls.invalidate(ver)
                continue
            if r.status_code == 200:
                raise IOError("il server non supporta richieste Range: questo metodo non e' utilizzabile")
            time.sleep(min(2 ** attempt, 60))
        raise IOError("impossibile determinare la dimensione dello zip remoto")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            b = self.pos // self.block
            if b not in self.cache:
                if len(self.cache) > 8:
                    self.cache.pop(next(iter(self.cache)))
                s = b * self.block
                self.cache[b] = self._get(s, min(s + self.block, self.size) - 1)
            data = self.cache[b]
            off = self.pos - b * self.block
            chunk = data[off: off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--url", help=argparse.SUPPRESS)   # solo test: indirizzo fisso
    a = ap.parse_args()
    cfg = load_config(a.config)
    prep = importlib.import_module("02_prepare_data")
    cache_size, geom = cfg["data"]["cache_size"], cfg["data"]["vindr_geometry"]
    kdir = cfg["paths"]["vindr_kaggle_dir"]
    os.makedirs(kdir, exist_ok=True)
    for d in glob.glob(os.path.join(kdir, "vindr_dl_*")):       # residui della versione precedente
        shutil.rmtree(d, ignore_errors=True)
    cdir = os.path.join(shared_data_dir(cfg), "vindr")
    os.makedirs(cdir, exist_ok=True)

    urls = UrlProvider((lambda: a.url) if a.url else kaggle_zip_url)
    rf = RemoteFile(urls)
    print(f"[VinDr] zip remoto: {rf.size / 1e9:.1f} GB (non viene scaricato per intero)")
    zf = zipfile.ZipFile(rf)
    infos = {i.filename: i for i in zf.infolist()}
    print(f"[VinDr] voci nello zip: {len(infos)}")

    csv = os.path.join(kdir, "train.csv")
    if not os.path.exists(csv):
        with zf.open(infos["train.csv"]) as src, open(csv + ".tmp", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(csv + ".tmp", csv)
    ids = sorted(pd.read_csv(csv)["image_id"].astype(str).unique())
    print(f"[VinDr] immagini annotate in train.csv: {len(ids)}")
    if a.max_images and a.max_images < len(ids):
        ids = sorted(np.random.default_rng(0).choice(ids, a.max_images, replace=False).tolist())
        print(f"[VinDr] sottoinsieme uniforme: {len(ids)} immagini (seed 0)")
    missing = [i for i in ids if f"train/{i}.dicom" not in infos]
    if missing:
        print(f"[VinDr][ATTENZIONE] {len(missing)} immagini di train.csv non presenti nello zip (es. {missing[:2]})")
    todo = [i for i in ids if f"train/{i}.dicom" in infos and not (
        os.path.exists(os.path.join(cdir, i + ".png")) and os.path.exists(os.path.join(cdir, i + ".png.geom.npy")))]
    tot_gb = sum(infos[f"train/{i}.dicom"].compress_size for i in todo) / 1e9
    print(f"[VinDr] gia' in cache: {len(ids) - len(todo)} | da leggere: {len(todo)} (~{tot_gb:.0f} GB di traffico)")

    local = threading.local()

    def one(iid):
        try:
            if not hasattr(local, "zf"):
                local.zf = zipfile.ZipFile(RemoteFile(urls, size=rf.size))
            info = local.zf.getinfo(f"train/{iid}.dicom")
            with local.zf.open(info) as fh:
                data = fh.read()
            tmp = os.path.join(cdir, f".{iid}.dicom.tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            try:
                prep._vindr_worker((tmp, os.path.join(cdir, iid + ".png"), cache_size, geom))
            finally:
                os.remove(tmp)
            return iid, None
        except Exception as e:
            return iid, f"{type(e).__name__}: {e}"

    t0, done, failed, gb = time.time(), 0, [], 0.0
    ex = ThreadPoolExecutor(a.threads)
    futs = {ex.submit(one, i): i for i in todo}
    try:
        for f in as_completed(futs):
                iid, err = f.result()
                done += 1
                if err:
                    failed.append(f"{iid}\t{err}")
                    if len(failed) <= 3:
                        print(f"[VinDr] errore su {iid}: {err}", flush=True)
                else:
                    gb += infos[f"train/{iid}.dicom"].compress_size / 1e9
                if done % 100 == 0 or done == len(todo):
                    el = max(time.time() - t0, 1e-6)
                    eta = (len(todo) - done) / (done / el) / 3600
                    print(f"[VinDr] {done}/{len(todo)} | {done / el:.2f} img/s | {gb / el * 1000:.0f} MB/s | "
                          f"ETA {eta:.1f} h | falliti {len(failed)}", flush=True)
                if done >= 200 and len(failed) == done:
                    print("[VinDr] tutti i tentativi falliscono: mi fermo (vedi gli errori sopra).")
                    break
    except KeyboardInterrupt:
        print("\n[VinDr] interrotto: annullo i download in coda ed esco (le immagini gia' in cache restano).", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        os._exit(130)
    ex.shutdown(wait=False, cancel_futures=True)
    fp = os.path.join(shared_data_dir(cfg), "vindr_download_failed.txt")
    with open(fp, "w") as fh:
        fh.write("\n".join(failed))
    print(f"[VinDr] fatto. Falliti: {len(failed)} (elenco in {fp}; rilancia lo script per ritentarli).")
    print("[VinDr] prossimo passo: python 02_prepare_data.py --skip-nih")


if __name__ == "__main__":
    main()
