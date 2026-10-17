# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Download pinned Gyro ranges with fresh URLs, durable resume, and SHA256 verification."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen

manifest_path, state = map(Path, sys.argv[1:])
p = json.loads(manifest_path.read_text())
state.mkdir(parents=True, exist_ok=True)
cache = Path(p['cache'])
model = cache / ('models--' + p['repo'].replace('/', '--'))
blob = model / 'blobs' / p['sha256']
partial = blob.with_name(blob.name + '.incomplete')
snapshot = model / 'snapshots' / p['revision'] / p['filename']
blob.parent.mkdir(parents=True, exist_ok=True)

def now():
    return datetime.now(timezone.utc).isoformat()

def save(name, value):
    temporary = state / (name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(state / name)

record = {**p, 'started_utc': now(), 'status': 'downloading', 'transport': 'fresh-url HTTP ranges',
          'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
save('progress.json', record)
try:
    offset = partial.stat().st_size if partial.exists() else 0
    assert 0 <= offset <= p['size']
    assert shutil.disk_usage(cache).free >= p['size'] - offset + 20 * 1024**3, 'Insufficient disk space'
    print(f'GYRO_DOWNLOAD_BEGIN revision={p["revision"]} bytes={p["size"]} resume_bytes={offset}', flush=True)
    if not blob.exists():
        with partial.open('r+b' if partial.exists() else 'w+b') as stream:
            while offset < p['size']:
                end = min(p['size'] - 1, offset + 256 * 1024**2 - 1)
                for attempt in range(1, 9):
                    try:
                        url = (f'https://huggingface.co/{p["repo"]}/resolve/{p["revision"]}/{p["filename"]}'
                               f'?download=true&fresh={time.time_ns()}')
                        request = Request(url, headers={'Range': f'bytes={offset}-{end}',
                                          'Cache-Control': 'no-cache', 'Accept-Encoding': 'identity'})
                        with urlopen(request, timeout=60) as response:
                            expected = f'bytes {offset}-{end}/{p["size"]}'
                            assert response.status == 206 and response.headers.get('Content-Range') == expected, 'Invalid range response'
                            stream.seek(offset)
                            count = 0
                            while count < end - offset + 1:
                                chunk = response.read(min(4 * 1024**2, end - offset + 1 - count))
                                if not chunk:
                                    raise IOError('Incomplete range response')
                                stream.write(chunk)
                                count += len(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                        break
                    except Exception as error:
                        stream.seek(offset)
                        stream.truncate()
                        stream.flush()
                        print(f'GYRO_RANGE_RETRY offset={offset} attempt={attempt} error={type(error).__name__}', flush=True)
                        if attempt == 8:
                            raise RuntimeError(f'Range at {offset} failed after eight attempts') from None
                        time.sleep(min(30, attempt * 3))
                offset = end + 1
                record.update(downloaded_bytes=offset, updated_utc=now())
                save('progress.json', record)
                print(f'GYRO_STEP step={offset // (256 * 1024**2)} bytes={offset} total={p["size"]}', flush=True)
    source = blob if blob.exists() else partial
    assert source.stat().st_size == p['size'], 'File size mismatch'
    record.update(status='verifying', downloaded_bytes=p['size'])
    save('progress.json', record)
    with source.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == p['sha256'], 'SHA256 mismatch'
    if source == partial:
        partial.replace(blob)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
    assert snapshot.resolve() == blob.resolve()
    record.update(status='completed', path=str(snapshot), sha256_verified=True, completed_utc=now())
    save('progress.json', record)
    save('complete.json', record)
    print('GYRO_COMPLETED status=completed_sha256_verified', flush=True)
except BaseException as error:
    record.update(status='failed', error=f'{type(error).__name__}: {error}', failed_utc=now())
    save('progress.json', record)
    raise
