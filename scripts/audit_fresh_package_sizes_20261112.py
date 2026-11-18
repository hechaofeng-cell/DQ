#!/usr/bin/env python3
"""Bounded HTTP size probes; never download dataset payloads or call models."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
import shutil

import requests

from audit_fresh_confirmation_20261112 import ROOT, OUT, read, dump, now

BASE = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/"


def probe(row):
    value = dict(package=row["package"], archive_exists=row["archive_exists"] == "True")
    if value["archive_exists"] and row["recorded_bytes"]:
        value.update(bytes=int(row["recorded_bytes"]), basis="historical source audit; no new HTTP probe", status="recorded")
        return value
    url = BASE + row["package"].split("/")[-1] + ".tar.gz"
    try:
        with requests.get(url + "?qe-fresh-size=20261112", headers={"Range":"bytes=0-0"}, stream=True, timeout=(15,30)) as r:
            r.raise_for_status()
            match = re.fullmatch(r"bytes 0-0/(\d+)", r.headers.get("Content-Range", ""))
            if r.status_code == 206 and match:
                size = int(match[1])
            elif r.status_code == 200 and r.headers.get("Content-Length"):
                size = int(r.headers["Content-Length"])
            else:
                raise ValueError("No verified total size in response headers")
            value.update(bytes=size, basis="HTTP response headers only; streamed body closed unread", status="verified_headers", url=url)
    except (requests.RequestException, ValueError) as exc:
        # Error type/status only; never persist proxy credentials or request headers.
        value.update(bytes=None, status="size_unresolved", error=type(exc).__name__)
    return value


def main():
    if (OUT / "package_size_audit.json").exists():
        raise FileExistsError("Size audit already exists")
    rows = read(OUT / "alternative_packages.csv")
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(probe, rows))
    missing = [r for r in records if not r["archive_exists"]]
    complete = all(r["bytes"] is not None for r in missing)
    size = sum(r["bytes"] for r in missing if r["bytes"] is not None)
    free = shutil.disk_usage(ROOT).free
    result = dict(completed_at=now(), records=records, header_requests=len(missing),
        all_missing_package_sizes_resolved=complete, verified_missing_bytes=size,
        bytes_are_lower_bound_if_unresolved=not complete, free_disk_bytes=free,
        staging_budget_bytes=80*1024**3, reserve_free_disk_bytes=20*1024**3,
        within_80GiB_download_budget=complete and size <= 80*1024**3,
        can_stage_all_missing_archives_with_20GiB_free=complete and size+20*1024**3 <= free,
        dataset_payload_downloaded=False, new_VLM_calls=0, new_classifier_calls=0)
    dump("package_size_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
