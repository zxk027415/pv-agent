from pathlib import Path

from pvdiag.data.download import (
    S3Object,
    download_object,
    local_path,
    parse_list_page,
    system_prefix,
)

PAGE_TRUNCATED = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>oedi-data-lake</Name>
  <IsTruncated>true</IsTruncated>
  <NextContinuationToken>tok123</NextContinuationToken>
  <Contents><Key>pvdaq/2023-solar-data-prize/2107_OEDI/data/</Key><Size>0</Size></Contents>
  <Contents><Key>pvdaq/2023-solar-data-prize/2107_OEDI/data/a.csv</Key><Size>42</Size></Contents>
  <Contents><Key>pvdaq/2023-solar-data-prize/2107_OEDI/metadata/2107_system_metadata.json</Key><Size>7</Size></Contents>
</ListBucketResult>"""

PAGE_LAST = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents><Key>pvdaq/2023-solar-data-prize/2107_OEDI/data/b.csv</Key><Size>5</Size></Contents>
</ListBucketResult>"""


def test_parse_truncated_page_skips_dir_markers():
    objs, token = parse_list_page(PAGE_TRUNCATED)
    assert token == "tok123"
    assert [o.key.rsplit("/", 1)[-1] for o in objs] == ["a.csv", "2107_system_metadata.json"]
    assert objs[0].size == 42


def test_parse_last_page_has_no_token():
    objs, token = parse_list_page(PAGE_LAST)
    assert token is None
    assert len(objs) == 1


def test_local_path_keeps_relative_layout(tmp_path: Path):
    prefix = system_prefix(2107)
    obj = S3Object(prefix + "data/a.csv", 42)
    assert local_path(obj, prefix, tmp_path, 2107) == tmp_path / "2107" / "data" / "a.csv"


def test_download_skips_existing_same_size(tmp_path: Path):
    path = tmp_path / "x.csv"
    path.write_bytes(b"12345")

    class Boom:
        def stream(self, *a, **k):
            raise AssertionError("should not hit network")

    assert download_object(Boom(), S3Object("k/x.csv", 5), path) is False
