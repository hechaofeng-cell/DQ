#!/usr/bin/env python3
"""Download official VOC archives with bounded range retries and known checksums."""
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
from voc_external_common import BASE,ARCHIVE_DIR,ARCHIVES,common

SPECS=(('VOCtrainval_06-Nov-2007.tar',460032000,'c52e279531787c972589f7e41ab4ae64'),
       ('VOCtest_06-Nov-2007.tar',451020800,'b6e924de25625d8de591ea690078ad9f'))

def digest(path):
    value=hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()

def main():
    ARCHIVE_DIR.mkdir(parents=True,exist_ok=True);BASE.mkdir(parents=True,exist_ok=True)
    lock=threading.Lock();jobs=[];chunk_size=8*1024*1024
    for name,size,md5 in SPECS:
        target=ARCHIVE_DIR/name
        if target.exists():
            if digest(target)!=md5:raise ValueError('Existing archive checksum mismatch: '+name)
            continue
        folder=ARCHIVE_DIR/(name+'.parts');folder.mkdir(exist_ok=True)
        original=ARCHIVE_DIR/(name+'.partial')
        if original.exists():
            with original.open('rb') as stream:
                for start in range(0,min(size,original.stat().st_size),chunk_size):
                    end=min(size,start+chunk_size);data=stream.read(end-start)
                    part=folder/f'{start:012d}.part'
                    if len(data)==end-start and not part.exists():part.write_bytes(data)
        for start in range(0,size,chunk_size):
            end=min(size,start+chunk_size)-1;part=folder/f'{start:012d}.part'
            if not part.exists() or part.stat().st_size!=end-start+1:jobs.append((name,size,start,end,part))
    def fetch(job):
        name,size,start,end,path=job;url='https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/'+name
        for attempt in range(1,7):
            try:
                offset=path.stat().st_size if path.exists() else 0
                if offset>end-start+1:raise ValueError('Oversized download part')
                if offset==end-start+1:return
                current=start+offset
                with requests.get(url+f'?range_start={current}',headers={'Range':f'bytes={current}-{end}'},stream=True,timeout=(30,60)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {current}-{end}/{size}':raise ValueError('Range response mismatch')
                    with path.open('ab') as stream:
                        for data in response.iter_content(128*1024):stream.write(data)
                if path.stat().st_size!=end-start+1:raise ValueError('Truncated part')
                return
            except (requests.RequestException,OSError,ValueError):
                if attempt==6:raise
                time.sleep(min(15,attempt*3))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures={executor.submit(fetch,j):j for j in jobs}
        for index,future in enumerate(as_completed(futures),1):
            future.result();job=futures[future]
            print(f'[{common.now()}] downloaded range {index}/{len(jobs)} {job[0]} {job[2]}-{job[3]}',flush=True)
            common.write_json(BASE/'download_progress.json',dict(status='running',completed_ranges=index,total_ranges=len(jobs),heartbeat=common.now()))
    for name,size,md5 in SPECS:
        target=ARCHIVE_DIR/name
        if target.exists():continue
        assembled=ARCHIVE_DIR/(name+'.assembled')
        with assembled.open('wb') as stream:
            for start in range(0,size,chunk_size):
                path=ARCHIVE_DIR/(name+'.parts')/f'{start:012d}.part'
                with path.open('rb') as part:
                    for data in iter(lambda:part.read(1024*1024),b''):stream.write(data)
        if assembled.stat().st_size!=size or digest(assembled)!=md5:raise ValueError('Archive size/checksum mismatch: '+name)
        assembled.rename(target)
    common.write_json(BASE/'download_audit.json',dict(status='pass',source='official Oxford VOC2007 trainval/test',
        archives=[dict(path=str(ARCHIVE_DIR/name),bytes=size,md5=md5,sha256=common.sha256(ARCHIVE_DIR/name)) for name,size,md5 in SPECS]))
    print('Both official VOC2007 archives verified.',flush=True)

if __name__=='__main__':main()
