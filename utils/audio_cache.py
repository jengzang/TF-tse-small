"""Optional node-local cache of immutable source audio; originals stay read-only."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def local_audio_path(source):
    root=os.environ.get('TFTSE_AUDIO_CACHE')
    if not root: return source
    digest=hashlib.sha256(str(source).encode()).hexdigest()
    folder=Path(root)/digest[:2];folder.mkdir(parents=True,exist_ok=True)
    target=folder/(digest+Path(source).suffix)
    if target.is_file(): return str(target)
    if shutil.disk_usage(folder).free < 20 * 1024**3:
        return source
    fd,tmp=tempfile.mkstemp(prefix=digest+'.',suffix='.tmp',dir=folder)
    os.close(fd)
    try:
        shutil.copyfile(source,tmp)
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return str(target)
