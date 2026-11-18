#!/usr/bin/env python3
"""Inspect, verify or restore the frozen fresh Objects365 originals on a new host.

Uses only Python's standard library. Downloads are explicit and never run models.
Archive streams are fully hashed; only frozen image members are written to disk.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'reproducibility/migration_20261118'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:
        raise ValueError('Expected a project-relative path: ' + relative)
    root = Path(root).resolve()
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError('Path escapes the selected root: ' + relative)
    return target


def verify_bundle(bundle):
    records = json.loads((bundle / 'manifest.sha256.json').read_text())
    for name, digest in records.items():
        p = safe_path(bundle, name)
        if not p.is_file() or sha(p) != digest:
            raise ValueError('Migration bundle changed or incomplete: ' + name)
    return len(records)


def load_packages(bundle):
    data = json.loads((bundle / 'objects365_fresh_downloads.json').read_text())
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported manifest version')
    seen = set()
    ids = set()
    for package in data['packages']:
        if package['package'] in ids:
            raise ValueError('Duplicate source package')
        ids.add(package['package'])
        url = urllib.parse.urlsplit(package['url'])
        if url.scheme != 'https' or url.username or url.password:
            raise ValueError('Expected a public HTTPS source URL')
        for record in package['images']:
            safe_path(Path.cwd(), record['relative_path'])
            if record['relative_path'] in seen or Path(record['name']).name != record['name']:
                raise ValueError('Duplicate output or invalid source member')
            seen.add(record['relative_path'])
            if not 0 < record['bytes'] <= 64 * 1024**2:
                raise ValueError('Unexpected image size')
    return data


def matches(root, record):
    p = safe_path(root, record['relative_path'])
    return p.is_file() and p.stat().st_size == record['bytes'] and sha(p) == record['sha256']


def check_records(root, records):
    counts = dict(valid=0, missing=0, mismatch=0)
    problems = []
    for record in records:
        p = safe_path(root, record['relative_path'])
        status = 'missing' if not p.exists() else 'valid' if matches(root, record) else 'mismatch'
        counts[status] += 1
        if status != 'valid':
            problems.append(dict(path=record['relative_path'], status=status))
    return dict(**counts, problems=problems)


class HashingReader:
    def __init__(self, stream, maximum, progress=None):
        self.stream = stream
        self.maximum = maximum
        self.count = 0
        self.hash = hashlib.sha256()
        self.progress = progress
        self.last_progress = time.monotonic()

    def read(self, size=-1):
        if size < 0:
            size = 1024 * 1024
        # Read one extra byte at the bound to detect an oversized response.
        size = min(size, self.maximum - self.count + 1)
        value = self.stream.read(size)
        self.count += len(value)
        if self.count > self.maximum:
            raise ValueError('Archive exceeded frozen byte length')
        self.hash.update(value)
        if self.progress and time.monotonic() - self.last_progress >= 30:
            self.last_progress = time.monotonic()
            self.progress(self.count)
        return value


def store_member(source, target, record):
    """Never overwrite existing data; atomically publish a verified new image."""
    if target.exists():
        if target.is_file() and target.stat().st_size == record['bytes'] and sha(target) == record['sha256']:
            return
        raise FileExistsError('Existing image differs; move it aside yourself: ' + str(target))
    content = source.read(record['bytes'] + 1)
    if len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
        raise ValueError('Image size/SHA256 differs: ' + record['name'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.qe-restore-', delete=False) as stream:
        temp = Path(stream.name)
        stream.write(content)
    try:
        # Hard link publishes without replacing a concurrently created target.
        try:
            target.hardlink_to(temp)
        except FileExistsError:
            if target.stat().st_size != record['bytes'] or sha(target) != record['sha256']:
                raise
    finally:
        temp.unlink(missing_ok=True)


def restore_archive(stream, package, root):
    wanted = {r['name']: r for r in package['images']}
    if len(wanted) != len(package['images']):
        raise ValueError('Repeated frozen basename')
    raw = HashingReader(stream, package['bytes'], progress=lambda count: print(json.dumps(
        dict(package=package['package'], compressed_bytes=count, total_bytes=package['bytes'])), flush=True))
    found = set()
    with gzip.GzipFile(fileobj=raw, mode='rb') as gz:
        with tarfile.open(fileobj=gz, mode='r|') as archive:
            for member in archive:
                name = PurePosixPath(member.name).name
                if name not in wanted:
                    continue
                if name in found or not member.isfile() or '..' in PurePosixPath(member.name).parts or PurePosixPath(member.name).is_absolute():
                    raise ValueError('Repeated or unsafe requested archive member: ' + member.name)
                record = wanted[name]
                if member.size != record['bytes']:
                    raise ValueError('Image member size differs: ' + name)
                with archive.extractfile(member) as source:
                    store_member(source, safe_path(root, record['relative_path']), record)
                found.add(name)
        # Check the complete gzip CRC/EOF even after the tar terminator.
        while gz.read(1024 * 1024):
            pass
    while raw.read(1024 * 1024):
        pass
    if raw.count != package['bytes'] or raw.hash.hexdigest() != package['sha256']:
        raise ValueError('Complete archive size/SHA256 differs: ' + package['package'])
    if found != set(wanted):
        raise ValueError('Frozen image members missing: ' + str(len(set(wanted) - found)))
    return dict(package=package['package'], archive_bytes=raw.count, archive_sha256=raw.hash.hexdigest(),
                source_verified=True, restored_images=len(found))


@contextmanager
def source_stream(package, root, archive_dir, timeout):
    basename = package['package'].replace('/', '_') + '.tar.gz'
    local = (Path(archive_dir) / basename) if archive_dir else (
        root / 'artifacts/objects365_independent_confirmation_20261005/data/packages' / basename)
    if local.is_file():
        with local.open('rb') as stream:
            yield stream
    else:
        request = urllib.request.Request(package['url'], headers={'User-Agent': 'QE-data-restore/1'})
        with urllib.request.urlopen(request, timeout=timeout) as stream:
            if stream.status != 200:
                raise ValueError('Expected a complete HTTP 200 archive response')
            yield stream


def download(packages, root, archive_dir=None, timeout=90):
    for package in packages:
        current = check_records(root, package['images'])
        if current['mismatch']:
            raise FileExistsError('Existing images differ: ' + json.dumps(current['problems'][:3]))
        if current['missing'] == 0:
            print(json.dumps(dict(package=package['package'], status='existing_images_hash_verified',
                                  images=current['valid'])), flush=True)
            continue
        root.mkdir(parents=True, exist_ok=True)
        required = sum(r['bytes'] for r in package['images']) + 20 * 1024**3
        if shutil.disk_usage(root).free < required:
            raise RuntimeError('Insufficient free space; retain 20 GiB beyond the selected package images')
        print(json.dumps(dict(package=package['package'], status='reading_complete_source_archive',
                              archive_bytes=package['bytes'])), flush=True)
        started = time.monotonic()
        with source_stream(package, root, archive_dir, timeout) as stream:
            audit = restore_archive(stream, package, root)
        audit['seconds'] = round(time.monotonic() - started, 2)
        audit_path = safe_path(root, 'artifacts/migration_restore_20261118/source_audits/' + package['package'].replace('/', '_') + '.json')
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        # A new audit describes this execution; never alters original experiment audits.
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps(audit), flush=True)


def localize(bundle, root, output):
    """Write distinct runtime manifests; preserve all original bytes and hashes."""
    provenance = json.loads((bundle / 'provenance.json').read_text())
    sources = [safe_path(bundle, p) for p in provenance['portable_manifests']]
    output = Path(output).resolve()
    # Do not allow derived manifests to overwrite the bundle or original data.
    if output.is_relative_to(bundle.resolve()) or any(output.is_relative_to((root / p).resolve()) for p in ['data', 'artifacts/fresh_image_qualification_20261112']):
        raise ValueError('Choose a separate new output directory for localized manifests')
    output.mkdir(parents=True, exist_ok=False)
    outputs = []
    for source in sources:
        with source.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            rows = list(reader); fields = reader.fieldnames
        for row in rows:
            for field, value in row.items():
                if value and (field.endswith('_path') or field == 'path'):
                    if Path(value).is_absolute():
                        raise ValueError('Unmapped absolute input path: ' + value)
                    row[field] = str(safe_path(root, value))
        name = source.relative_to(bundle / 'manifests')
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        outputs.append(dict(input=str(source.relative_to(bundle)), input_sha256=sha(source),
                            output=name.as_posix(), output_sha256=sha(destination)))
    (output / 'relocation.json').write_text(json.dumps(dict(root=str(root), manifests=outputs,
        boundary='Path relocation only. Does not install historical freezes, regenerate rendered views, restore annotations or certify a new data audit.'), indent=2) + '\n')
    return dict(output=str(output), localized_manifests=len(outputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'download', 'localize', 'verify-bundle'])
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--root', type=Path, default=ROOT, help='New QE checkout or separate data root')
    parser.add_argument('--package', action='append', help='Download/check only this package, e.g. v2/patch29; repeatable')
    parser.add_argument('--archive-dir', type=Path, help='Existing archives named v2_patch29.tar.gz, etc.')
    parser.add_argument('--output', type=Path, help='New directory for path-localized CSV copies')
    parser.add_argument('--checkpoints', action='store_true', help='With check: additionally verify 15 original trained checkpoints')
    parser.add_argument('--training-crops', action='store_true', help='With check: also verify historical training/validation crops and YOLO label archive')
    parser.add_argument('--timeout', type=int, default=90)
    args = parser.parse_args()
    bundle = args.bundle.resolve(); root = args.root.resolve()
    verified = verify_bundle(bundle)
    data = load_packages(bundle); packages = data['packages']
    if args.package:
        known = {p['package'] for p in packages}
        if not set(args.package) <= known:
            parser.error('Unknown package; run plan to list frozen package IDs')
        packages = [p for p in packages if p['package'] in args.package]
    if args.command == 'verify-bundle':
        print(json.dumps(dict(verified_bundle_files=verified, status='pass')))
    elif args.command == 'plan':
        print(json.dumps(dict(packages=[dict(package=p['package'], url=p['url'], bytes=p['bytes'], images=len(p['images'])) for p in packages],
            selected_packages=len(packages), originals=sum(len(p['images']) for p in packages),
            archive_gib=round(sum(p['bytes'] for p in packages)/1024**3, 3),
            extracted_gib=round(sum(r['bytes'] for p in packages for r in p['images'])/1024**3, 3),
            root=str(root), network_accessed=False), ensure_ascii=False, indent=2))
    elif args.command == 'check':
        images = check_records(root, [r for p in packages for r in p['images']])
        result = dict(images=images)
        if args.checkpoints:
            checkpoints = json.loads((bundle / 'trained_checkpoints.json').read_text())['checkpoints']
            result['trained_checkpoints'] = check_records(root, checkpoints)
        if args.training_crops:
            training = json.loads((bundle / 'training_crops.json').read_text())
            result['training_inputs'] = check_records(root, training['files'] + [training['labels_archive']])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if any(v['missing'] or v['mismatch'] for v in result.values()):
            return 1
    elif args.command == 'download':
        download(packages, root, args.archive_dir, args.timeout)
    elif args.command == 'localize':
        if args.output is None:
            parser.error('localize requires --output pointing to a new directory')
        print(json.dumps(localize(bundle, root, args.output), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(1)
