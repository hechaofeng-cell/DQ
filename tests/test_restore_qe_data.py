"""Recovery contracts: exact data membership, full-stream checks and no overwrites."""
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('qe_restore', Path(__file__).resolve().parents[1] / 'scripts/restore_qe_data.py')
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixture(members=None):
    payload = b'fixture-image-bytes'
    members = members if members is not None else [('images/photo.jpg', payload), ('unused.txt', b'skip')]
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as archive:
        for name, content in members:
            info = tarfile.TarInfo(name); info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    content = gzip.compress(raw.getvalue())
    record = dict(name='photo.jpg', bytes=len(payload), sha256=digest(payload), relative_path='artifacts/images/photo.jpg')
    package = dict(package='v2/patch1', url='https://example.invalid/patch1.tar.gz',
                   bytes=len(content), sha256=digest(content), images=[record])
    return content, package


class ArchiveTests(unittest.TestCase):
    def test_only_frozen_members_are_extracted_and_full_archive_is_verified(self):
        content, package = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = restore.restore_archive(io.BytesIO(content), package, root)
            self.assertTrue(result['source_verified'])
            self.assertEqual(result['archive_bytes'], len(content))
            self.assertEqual(restore.check_records(root, package['images'])['valid'], 1)
            self.assertFalse((root/'unused.txt').exists())

    def test_whole_archive_mismatch_fails_even_when_image_hash_matches(self):
        content, package = fixture(); package['sha256'] = '0'*64
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Complete archive'):
                restore.restore_archive(io.BytesIO(content), package, Path(directory))

    def test_gzip_crc_is_checked_after_tar_end(self):
        content, package = fixture()
        changed = bytearray(content); changed[-8] ^= 1
        content = bytes(changed); package['sha256'] = digest(content)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises((OSError, EOFError)):
                restore.restore_archive(io.BytesIO(content), package, Path(directory))

    def test_truncated_archive_cannot_pass(self):
        content, package = fixture()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises((OSError, EOFError, ValueError)):
                restore.restore_archive(io.BytesIO(content[:-9]), package, Path(directory))

    def test_wrong_image_content_is_not_published(self):
        content, package = fixture(); package['images'][0]['sha256'] = '0'*64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Image size/SHA256'):
                restore.restore_archive(io.BytesIO(content), package, root)
            self.assertFalse((root/'artifacts/images/photo.jpg').exists())

    def test_duplicate_basename_is_rejected(self):
        content, package = fixture([('a/photo.jpg', b'fixture-image-bytes'), ('b/photo.jpg', b'fixture-image-bytes')])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Repeated or unsafe'):
                restore.restore_archive(io.BytesIO(content), package, Path(directory))

    def test_requested_traversal_member_is_rejected(self):
        content, package = fixture([('../photo.jpg', b'fixture-image-bytes')])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                restore.restore_archive(io.BytesIO(content), package, Path(directory))

    def test_existing_different_file_is_never_overwritten(self):
        content, package = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root/'artifacts/images/photo.jpg'
            target.parent.mkdir(parents=True); target.write_bytes(b'existing-user-file')
            with self.assertRaises(FileExistsError):
                restore.restore_archive(io.BytesIO(content), package, root)
            self.assertEqual(target.read_bytes(), b'existing-user-file')

    def test_complete_existing_images_skip_network(self):
        content, package = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restore.restore_archive(io.BytesIO(content), package, root)
            with patch.object(restore.urllib.request, 'urlopen') as request:
                restore.download([package], root)
                request.assert_not_called()

    def test_missing_images_download_and_emit_a_separate_source_audit(self):
        content, package = fixture()
        response = io.BytesIO(content); response.status = 200
        disk = type('Disk', (), dict(free=100*1024**3))()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(restore.urllib.request, 'urlopen', return_value=response) as request, patch.object(restore.shutil, 'disk_usage', return_value=disk):
                restore.download([package], root)
                self.assertEqual(request.call_count, 1)
            audit = root/'artifacts/migration_restore_20261118/source_audits/v2_patch1.json'
            self.assertTrue(json.loads(audit.read_text())['source_verified'])


class ManifestTests(unittest.TestCase):
    def test_relative_paths_and_symlink_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'root'; root.mkdir()
            for name in ['../escape.jpg', '/tmp/escape.jpg', 'a/../../b', 'a\\b']:
                with self.assertRaises(ValueError): restore.safe_path(root, name)
            (root/'outside').symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError): restore.safe_path(root, 'outside/escape.jpg')

    def test_equal_size_changed_file_fails_check(self):
        _, package = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root/'artifacts/images/photo.jpg'
            target.parent.mkdir(parents=True); target.write_bytes(b'x'*package['images'][0]['bytes'])
            self.assertEqual(restore.check_records(root, package['images'])['mismatch'], 1)

    def test_localize_preserves_source_and_refuses_reused_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = root/'bundle'; source = bundle/'manifests/artifacts/targets.csv'
            source.parent.mkdir(parents=True)
            source.write_text('sample_id,image_path,image_sha256\nx,data/image.jpg,unchanged\n')
            before = source.read_bytes()
            (bundle/'provenance.json').write_text(json.dumps(dict(portable_manifests=['manifests/artifacts/targets.csv'])))
            output = root/'localized'
            restore.localize(bundle, root/'new-checkout', output)
            with (output/'artifacts/targets.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['image_path'], str(root/'new-checkout/data/image.jpg'))
            self.assertEqual(row['image_sha256'], 'unchanged')
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(FileExistsError): restore.localize(bundle, root/'new-checkout', output)

    def test_bundle_corruption_fails_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory); (bundle/'record.json').write_text('original')
            (bundle/'manifest.sha256.json').write_text(json.dumps({'record.json': digest(b'original')}))
            self.assertEqual(restore.verify_bundle(bundle), 1)
            (bundle/'record.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed or incomplete'):
                restore.verify_bundle(bundle)


if __name__ == '__main__':
    unittest.main()
