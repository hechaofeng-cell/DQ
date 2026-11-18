#!/usr/bin/env python3
"""Extract frozen target basenames from complete, validated gzip/tar streams.

No full new archives or range chunks are written to disk. Images extracted from
an incomplete package are never marked source-verified. No model calls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import threading
import time

from PIL import Image
import requests

from prepare_fresh_images_20261112 import ROOT, READY, OUT, prepare, read, dump, write, sha


class NetworkBudget:
    def __init__(self, maximum, existing=0, persist=None):
        self.maximum=maximum;self.used=existing;self.lock=threading.Lock();self.persist=persist

    def reserve(self, size):
        with self.lock:
            if self.used+size>self.maximum:raise RuntimeError("Frozen network byte budget exhausted")
            self.used+=size
            if self.persist:self.persist(self.used)


class RangeReader(io.RawIOBase):
    """Ordered, memory-bounded prefetch; exact Content-Range validation per chunk."""
    def __init__(self,url,size,budget,chunk=8*1024**2,workers=8,request=None):
        self.url=url;self.size=size;self.budget=budget;self.chunk=chunk;self.workers=workers
        self.pool=ThreadPoolExecutor(max_workers=workers);self.request=request or requests.get
        self.futures={};self.buffer=b"";self.next_index=0;self.scheduled=0;self.position=0
        self.hash=hashlib.sha256();self.total=(size+chunk-1)//chunk
        self.started=time.monotonic();self.last_progress=self.started;self._schedule()

    def _schedule(self):
        while len(self.futures)<self.workers and self.scheduled<self.total:
            i=self.scheduled;self.futures[i]=self.pool.submit(self.fetch,i);self.scheduled+=1

    def fetch(self,i):
        start=i*self.chunk;end=min(self.size,start+self.chunk)-1
        for attempt in range(3):
            self.budget.reserve(end-start+1)
            try:
                with self.request(self.url+f"?qe-fresh-range={start}-{end}",headers={"Range":f"bytes={start}-{end}"},
                                  stream=True,timeout=(20,60)) as r:
                    r.raise_for_status()
                    expected=f"bytes {start}-{end}/{self.size}"
                    if r.status_code!=206 or r.headers.get("Content-Range")!=expected:
                        raise ValueError("Range response does not identify the requested bytes")
                    chunks=[];count=0
                    for block in r.iter_content(1024*1024):
                        count+=len(block)
                        if count>end-start+1:raise ValueError("Range response exceeded requested length")
                        chunks.append(block)
                    data=b"".join(chunks)
                    if len(data)!=end-start+1:raise ValueError("Truncated range response")
                    return data
            except (requests.RequestException,ValueError):
                if attempt==2:raise

    def readable(self):return True

    def read(self,size=-1):
        if size<0:raise ValueError("Unbounded stream reads are forbidden")
        chunks=[];needed=size
        while needed:
            if not self.buffer:
                if self.next_index==self.total:break
                self.buffer=self.futures.pop(self.next_index).result()
                self.next_index+=1;self._schedule()
            n=min(needed,len(self.buffer));data=self.buffer[:n];self.buffer=self.buffer[n:]
            self.hash.update(data);self.position+=n;chunks.append(data);needed-=n
            if time.monotonic()-self.last_progress>=30:
                elapsed=time.monotonic()-self.started;self.last_progress=time.monotonic()
                print(json.dumps(dict(stage="range_progress",compressed_bytes=self.position,total_bytes=self.size,
                    MiB_per_second=round(self.position/max(elapsed,1)/1024**2,2))),flush=True)
        return b"".join(chunks)

    def close(self):
        if not self.closed:
            for future in self.futures.values():future.cancel()
            self.pool.shutdown(wait=True,cancel_futures=True)
        super().close()


def extract_stream(raw,wanted,directory):
    """Only frozen basenames; duplicate names, truncation and bad gzip CRC fail."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    found={};seen=set()
    # gzip.GzipFile validates CRC/EOF. tarfile's r|gz alone can stop at tar's
    # terminator without checking the complete gzip stream, so drain gzip too.
    with gzip.GzipFile(fileobj=raw,mode="rb") as gz:
        with tarfile.open(fileobj=gz,mode="r|") as archive:
            for member in archive:
                name=Path(member.name).name
                if not member.isfile() or name not in wanted:continue
                if name in seen:raise ValueError("Repeated frozen basename in source archive")
                seen.add(name);target=directory/name
                if member.size<=0 or member.size>64*1024**2:raise ValueError("Implausible original image member size")
                with archive.extractfile(member) as source:
                    content=source.read(member.size+1)
                if len(content)!=member.size:raise ValueError("Truncated archive member")
                with Image.open(io.BytesIO(content)) as im:
                    im.verify()
                record=wanted[name]
                with Image.open(io.BytesIO(content)) as im:
                    if im.size!=(int(record["width"]),int(record["height"])):
                        raise ValueError("Official/image dimensions disagree")
                digest=hashlib.sha256(content).hexdigest()
                if target.exists():
                    if sha(target)!=digest:raise ValueError("Existing extracted image differs from source")
                else:
                    temp=target.with_suffix(".extracting");temp.write_bytes(content);temp.replace(target)
                found[name]=dict(path=str(target.resolve()),sha256=digest,bytes=member.size)
        while gz.read(1024*1024):pass
    if set(found)!=set(wanted):raise ValueError("Frozen source members missing: "+str(len(set(wanted)-set(found))))
    return found


def acquire(package_only=None):
    prepare()
    code=OUT/"acquisition_code_freeze.json"
    binding={str(Path(__file__)):sha(__file__)}
    if code.exists():
        if json.loads(code.read_text())!=binding:raise ValueError("Acquisition implementation changed")
    else:dump("acquisition_code_freeze.json",binding)
    plan=read(READY/"alternative_provisional_plan.csv")
    packages=read(READY/"alternative_packages.csv")
    sizes={r["package"]:r for r in json.loads((READY/"package_size_audit.json").read_text())["records"]}
    frozen=json.loads((READY/"frozen_validation_protocol.json").read_text())
    budget_path=OUT/"network_budget.json"
    previous=json.loads(budget_path.read_text())["reserved_bytes"] if budget_path.exists() else 0
    budget=NetworkBudget(frozen["data_gate"]["maximum_network_payload_bytes"],previous,
        lambda n:dump("network_budget.json",dict(reserved_bytes=n,meaning="conservative attempted range bytes, including retries and prefetched ranges")))
    for package in packages:
        name=package["package"]
        if package_only and name!=package_only:continue
        marker=OUT/"package_audits"/(name.replace("/","_")+".json")
        if marker.exists() and json.loads(marker.read_text())["status"]=="pass":continue
        if shutil.disk_usage(OUT).free<frozen["data_gate"]["minimum_free_disk_after_staging_bytes"]:
            raise RuntimeError("Frozen minimum free disk gate failed")
        wanted={Path(r["source_file_name"]).name:r for r in plan if r["package"]==name}
        print(json.dumps(dict(stage="package_start",package=name,targets=len(wanted))),flush=True)
        started=time.monotonic();record=sizes[name];local=Path(package["local_archive"])
        try:
            if local.is_file():
                metadata=json.loads(local.with_suffix(".gz.download.json").read_text())
                if local.stat().st_size!=record["bytes"]:raise ValueError("Changed local archive size")
                if sha(local)!=metadata["sha256"]:raise ValueError("Changed local archive SHA")
                with local.open("rb") as raw:found=extract_stream(raw,wanted,OUT/"images")
                archive_sha=metadata["sha256"];mode="verified_local_archive"
            else:
                with RangeReader(record["url"],record["bytes"],budget) as raw:
                    found=extract_stream(raw,wanted,OUT/"images")
                    while raw.read(1024*1024):pass
                    if raw.position!=record["bytes"]:raise ValueError("Incomplete compressed source")
                    archive_sha=raw.hash.hexdigest()
                mode="validated_ranges_complete_gzip_tar"
            value=dict(status="pass",package=name,mode=mode,archive_bytes=record["bytes"],archive_sha256=archive_sha,
                targets=len(wanted),found=found,seconds=round(time.monotonic()-started,3),network_reserved_bytes=budget.used)
            dump("package_audits/"+marker.name,value)
            print(json.dumps({k:v for k,v in value.items() if k!="found"}),flush=True)
        except Exception as exc:
            dump("package_failures/"+name.replace("/","_")+".json",dict(package=name,status="failed",
                error=type(exc).__name__+":"+str(exc)[:200],network_reserved_bytes=budget.used,
                extracted_images_not_source_verified=True))
            raise
    results=[json.loads(p.read_text()) for p in (OUT/"package_audits").glob("*.json")]
    dump("acquisition_progress.json",dict(packages_passed=len(results),packages_required=len(packages),
        extracted_source_verified=sum(r["targets"] for r in results),network_reserved_bytes=budget.used,
        new_VLM_calls=0,new_classifier_calls=0,all_packages_complete=len(results)==len(packages)))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--package");args=p.parse_args()
    acquire(args.package)
