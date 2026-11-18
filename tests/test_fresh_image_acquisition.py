import gzip
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from acquire_fresh_images_20261112 import extract_stream, RangeReader, NetworkBudget
from prepare_fresh_images_20261112 import resolve_path, ROOT


def archive(names):
    picture=io.BytesIO();Image.new("RGB",(32,24),(123,55,99)).save(picture,format="JPEG")
    content=picture.getvalue();result=io.BytesIO()
    with tarfile.open(fileobj=result,mode="w") as z:
        for name in names:
            entry=tarfile.TarInfo(name);entry.size=len(content);z.addfile(entry,io.BytesIO(content))
    return gzip.compress(result.getvalue()),content


class FakeResponse:
    status_code=206
    def __init__(self,data,start,end,total,wrong=False):
        self.data=data[start:end+1];self.headers={"Content-Range":f"bytes {start}-{end}/{total+int(wrong)}"}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):pass
    def iter_content(self,size):yield self.data


class FreshImageAcquisitionTests(unittest.TestCase):
    def test_frozen_basename_cannot_escape_destination(self):
        data,content=archive(["../../target.jpg","unselected.jpg"])
        with tempfile.TemporaryDirectory() as d:
            result=extract_stream(io.BytesIO(data),{"target.jpg":{"width":"32","height":"24"}},Path(d)/"images")
            self.assertEqual(set(result),{"target.jpg"})
            self.assertEqual((Path(d)/"images/target.jpg").read_bytes(),content)
            self.assertFalse((Path(d)/"target.jpg").exists())
            self.assertFalse((Path(d)/"images/unselected.jpg").exists())

    def test_gzip_trailer_is_checked_after_tar_terminator(self):
        data,_=archive(["target.jpg"])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((EOFError,OSError)):
                extract_stream(io.BytesIO(data[:-8]),{"target.jpg":{"width":32,"height":24}},d)

    def test_duplicate_source_basename_is_rejected(self):
        data,_=archive(["dir1/target.jpg","dir2/target.jpg"])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"Repeated"):
                extract_stream(io.BytesIO(data),{"target.jpg":{"width":32,"height":24}},d)

    def test_wrong_official_dimensions_do_not_create_image(self):
        data,_=archive(["target.jpg"])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"dimensions"):
                extract_stream(io.BytesIO(data),{"target.jpg":{"width":33,"height":24}},d)
            self.assertFalse((Path(d)/"target.jpg").exists())

    def test_range_chunks_reassemble_and_hash_exactly(self):
        data=bytes(range(251))*37
        def request(url,headers,**kwargs):
            start,end=map(int,headers["Range"].split("=")[1].split("-"))
            return FakeResponse(data,start,end,len(data))
        budget=NetworkBudget(len(data))
        with RangeReader("https://example.test/file",len(data),budget,chunk=127,workers=3,request=request) as r:
            parts=[]
            while b:=r.read(311):parts.append(b)
            self.assertEqual(b"".join(parts),data)
            self.assertEqual(r.hash.hexdigest(),hashlib.sha256(data).hexdigest())
        self.assertEqual(budget.used,len(data))

    def test_wrong_range_headers_cannot_be_concatenated(self):
        data=b"source bytes"
        def request(url,headers,**kwargs):
            start,end=map(int,headers["Range"].split("=")[1].split("-"))
            return FakeResponse(data,start,end,len(data),wrong=True)
        with RangeReader("https://example.test/file",len(data),NetworkBudget(1000),chunk=4,workers=1,request=request) as r:
            with self.assertRaisesRegex(ValueError,"Range response"):
                r.read(1)

    def test_portable_relative_path_resolves_against_manifest_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p=Path(d)/"images/a.jpg";p.parent.mkdir();p.write_bytes(b"test")
            self.assertEqual(resolve_path(d,"images/a.jpg"),str(p.resolve()))


if __name__=="__main__":unittest.main()
