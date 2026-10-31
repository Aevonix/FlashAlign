# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Anonymous runtime distribution: integrity, isolation and hostile archives."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import tempfile
import threading
import unittest
from unittest import mock

from test_setup import wizard
from test_native_setup import make_bundle
from owner_mode import engine_assets as assets
from owner_mode.quickstart import download_engine_bundle
from owner_mode.setup_runtime import SetupError


class RuntimeAssets(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root/'home'
        self.home.mkdir()
        self.catalog = self.root/'catalog.json'
        bundle = make_bundle(self.root/'native-fixture', platform_id='linux-x86_64')
        manifest = json.loads((bundle/'manifest.json').read_text())
        manifest['model_types'] = ['fixture']
        (bundle/'manifest.json').write_text(json.dumps(manifest))
        self.files = {str(p.relative_to(bundle)): p.read_bytes() for p in bundle.rglob('*') if p.is_file()}
        self.write_catalog(self.archive(self.files))

    def archive(self, files, extra=()):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, content in files.items():
                item = tarfile.TarInfo(name)
                item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
            for item, content in extra:
                archive.addfile(item, io.BytesIO(content) if content is not None else None)
        return stream.getvalue()

    def write_catalog(self, data, *, files=None):
        files = self.files if files is None else files
        self.data = data
        self.entry = {'id': 'fixture-v1', 'model_types': ['fixture'],
                      'platforms': ['linux-x86_64'],
                      'url': 'https://github.com/Aevonix/FlashAlign/releases/download/v1.0.0/fixture.tar.gz',
                      'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                      'files': {name: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
                                for name, value in files.items()}}
        self.catalog.write_text(json.dumps({'version': 2, 'packages': [self.entry]}))

    def download(self):
        with (mock.patch.object(assets.platform, 'system', return_value='Linux'),
              mock.patch.object(assets.platform, 'machine', return_value='x86_64')):
            return assets.download_runtime_bundle(self.home, 'fixture', catalog_path=self.catalog)

    def test_no_login_or_profile_download_and_verified_cache_reuse(self):
        with (mock.patch.object(assets, '_fetch', return_value=self.data) as fetch,
              mock.patch.dict(os.environ, {'GH_TOKEN': 'unused-synthetic-token'}),
              mock.patch('subprocess.run', side_effect=AssertionError('no GitHub CLI'))):
            first = self.download()
            second = self.download()
        self.assertEqual(first, second)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual((first/'bin/tensorfold-native').read_bytes(), self.files['bin/tensorfold-native'])
        self.assertFalse(list(self.home.rglob('profile.json')))
        self.assertFalse(list(self.home.rglob('profile.safetensors')))
        self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((first/'manifest.json').stat().st_mode), 0o600)

    def test_exact_runtime_legal_documents_are_pinned_and_extracted(self):
        for name in assets.RUNTIME_DOCUMENTS:
            self.files[name] = ('Legal fixture: '+name+'\n').encode()
        self.write_catalog(self.archive(self.files))
        with mock.patch.object(assets, '_fetch', return_value=self.data):
            package = self.download()
        for name in assets.RUNTIME_DOCUMENTS:
            self.assertEqual((package/name).read_bytes(), self.files[name])

    def test_legal_directory_is_not_an_arbitrary_archive_allowlist(self):
        for name in ('legal/private.key', 'legal/extra.md', 'legal/../secret', 'CLA-INDIVIDUAL.md',
                     'legal/CLA-ENTITY.md', 'legal/subdir/CLA-TERMS.md'):
            with self.subTest(path=name), self.assertRaises(SetupError):
                assets._runtime_path(name)

    def test_public_wizard_uses_the_anonymous_runtime_path(self):
        with mock.patch('owner_mode.quickstart.download_runtime_bundle', return_value=self.root) as download:
            self.assertEqual(download_engine_bundle(self.home, 'fixture'), self.root)
        download.assert_called_once_with(self.home, 'fixture', platform_id=None)
        self.assertIn('engine_assets.py', wizard.MODULES)
        self.assertIn('engine_catalog.json', wizard.MODULES)

    def test_corrupt_or_truncated_asset_does_not_create_cache(self):
        for data in (self.data[:-1], b'x'+self.data[1:]):
            with self.subTest(size=len(data)), mock.patch.object(assets, '_fetch', return_value=data):
                with self.assertRaisesRegex(SetupError, 'checksum or size'):
                    self.download()
                self.assertFalse((self.home/'.local').exists())

    def test_profile_payload_is_rejected_before_download(self):
        files = {**self.files, 'assets/profile.safetensors': b'not permitted'}
        self.write_catalog(self.archive(files), files=files)
        with mock.patch.object(assets, '_fetch') as fetch:
            with self.assertRaisesRegex(SetupError, 'profiles'):
                self.download()
        fetch.assert_not_called()

    def test_pinned_but_invalid_gzip_is_reported_without_publishing_cache(self):
        self.write_catalog(self.data[:len(self.data)//2])
        with mock.patch.object(assets, '_fetch', return_value=self.data):
            with self.assertRaises(SetupError):
                self.download()
        destination = self.home/'.local/share/flashalign/engine-bundles'/self.entry['sha256']
        self.assertFalse(destination.exists())

    def test_traversal_and_links_do_not_write_outside_the_staging_directory(self):
        cases = []
        item = tarfile.TarInfo('../outside')
        item.size = 1
        cases.append((item, b'x'))
        item = tarfile.TarInfo('assets/link')
        item.type = tarfile.SYMTYPE
        item.linkname = str(self.root/'outside')
        cases.append((item, None))
        for item in cases:
            data = self.archive(self.files, [item])
            self.write_catalog(data)
            with self.subTest(path=item[0].name), mock.patch.object(assets, '_fetch', return_value=data):
                with self.assertRaises(SetupError):
                    self.download()
                self.assertFalse((self.root/'outside').exists())
                self.assertFalse((self.home/'.local/share/flashalign/engine-bundles'/self.entry['sha256']).exists())

    def test_repeated_archive_member_is_rejected(self):
        item = tarfile.TarInfo('manifest.json')
        item.size = len(self.files['manifest.json'])
        data = self.archive(self.files, [(item, self.files['manifest.json'])])
        self.write_catalog(data)
        with mock.patch.object(assets, '_fetch', return_value=data):
            with self.assertRaisesRegex(SetupError, 'repeats'):
                self.download()

    def test_expanded_size_limit_applies_before_payload_read(self):
        item = tarfile.TarInfo('assets/oversized')
        item.size = assets.MAX_EXPANDED+1
        data = gzip.compress(item.tobuf()+b'\0'*1024)
        self.write_catalog(data)
        with mock.patch.object(assets, '_fetch', return_value=data):
            with self.assertRaisesRegex(SetupError, 'limit'):
                self.download()

    def test_directory_headers_count_toward_the_entry_limit(self):
        item = tarfile.TarInfo('assets')
        item.type = tarfile.DIRTYPE
        self.write_catalog(self.archive(self.files, [(item, None)]*assets.MAX_FILES))
        with mock.patch.object(assets, '_fetch', return_value=self.data):
            with self.assertRaisesRegex(SetupError, 'entry limit'):
                self.download()

    def test_pax_metadata_counts_toward_the_expanded_size_limit(self):
        item = tarfile.TarInfo('assets/pax-fixture')
        item.pax_headers = {'comment': 'x'*(64*1024)}
        self.write_catalog(self.archive(self.files, [(item, b'')]))
        with (mock.patch.object(assets, '_fetch', return_value=self.data),
              mock.patch.object(assets, 'MAX_EXPANDED', 32*1024)):
            with self.assertRaisesRegex(SetupError, 'expanded size limit'):
                self.download()

    def test_tampering_with_both_cache_file_and_receipt_does_not_bypass_catalog(self):
        with mock.patch.object(assets, '_fetch', return_value=self.data):
            package = self.download()
        target = package/'bin/tensorfold-native'
        target.write_bytes(b'changed by the test')
        path = package/'asset-receipt.json'
        receipt = json.loads(path.read_text())
        receipt['files'][str(target.relative_to(package))] = {
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size}
        path.write_text(json.dumps(receipt))
        with mock.patch.object(assets, '_fetch') as fetch:
            with self.assertRaisesRegex(SetupError, 'catalog'):
                self.download()
        fetch.assert_not_called()
        self.assertEqual(target.read_bytes(), b'changed by the test')

    def test_a_symlinked_cache_parent_is_not_used(self):
        outside = self.root/'outside'
        outside.mkdir()
        (self.home/'.local').symlink_to(outside, target_is_directory=True)
        with mock.patch.object(assets, '_fetch', return_value=self.data):
            with self.assertRaisesRegex(SetupError, 'owned directory'):
                self.download()
        self.assertFalse(list(outside.iterdir()))

    def test_cached_package_rechecks_parent_ownership_before_reuse(self):
        with mock.patch.object(assets, '_fetch', return_value=self.data):
            package = self.download()
        package.parent.chmod(0o777)
        with mock.patch.object(assets, '_fetch') as fetch:
            with self.assertRaisesRegex(SetupError, 'owned directory'):
                self.download()
        fetch.assert_not_called()

    def test_cached_files_and_nested_directories_cannot_be_writable_by_others(self):
        with mock.patch.object(assets, '_fetch', return_value=self.data):
            package = self.download()
        targets = [(package/'bin', 0o777, 0o700),
                   (package/'bin/tensorfold-native', 0o666, 0o600)]
        for target, unsafe, restored in targets:
            target.chmod(unsafe)
            with self.subTest(path=target.name), mock.patch.object(assets, '_fetch') as fetch:
                with self.assertRaisesRegex(SetupError, 'not writable by others'):
                    self.download()
                fetch.assert_not_called()
            target.chmod(restored)

    def test_concurrent_downloads_publish_one_complete_cache(self):
        barrier = threading.Barrier(2)
        def fetch(entry):
            barrier.wait(timeout=3)
            return self.data
        with mock.patch.object(assets, '_fetch', side_effect=fetch):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: self.download(), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertTrue((results[0]/'asset-receipt.json').is_file())
        self.assertFalse(list(results[0].parent.glob('.bundle-*')))

    def test_unknown_family_and_platform_do_not_download(self):
        with mock.patch.object(assets, '_fetch') as fetch:
            self.assertIsNone(assets.download_runtime_bundle(self.home, 'unknown', catalog_path=self.catalog))
            with mock.patch.object(assets.platform, 'system', return_value='Unsupported'):
                self.assertIsNone(assets.download_runtime_bundle(self.home, 'fixture', catalog_path=self.catalog))
        fetch.assert_not_called()

    def test_https_release_requests_carry_no_authentication_headers(self):
        response = io.BytesIO(self.data)
        response.geturl = lambda: 'https://release-assets.githubusercontent.com/asset?signature=synthetic'
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(assets, 'build_opener', return_value=opener):
            self.assertEqual(assets._fetch(self.entry), self.data)
        request = opener.open.call_args.args[0]
        self.assertNotIn('authorization', {name.lower() for name in request.headers})
        self.assertNotIn('cookie', {name.lower() for name in request.headers})

    def test_untrusted_initial_urls_and_redirects_are_rejected(self):
        for url in ('http://github.com/Aevonix/FlashAlign/releases/download/v1/a.tar.gz',
                    'https://example.com/a.tar.gz', 'https://token@github.com/Aevonix/FlashAlign/releases/download/v1/a.tar.gz',
                    'https://github.com/other/project/releases/download/v1/a.tar.gz'):
            with self.subTest(url=url), self.assertRaises(SetupError):
                assets._https_url(url, initial=True)
        redirect = assets._ReleaseRedirects()
        with self.assertRaises(SetupError):
            redirect.redirect_request(None, None, 302, '', {}, 'http://github.com/unsafe')
        with self.assertRaises(SetupError):
            redirect.redirect_request(None, None, 302, '', {}, 'https://example.com/unsafe')


if __name__ == '__main__':
    unittest.main()
