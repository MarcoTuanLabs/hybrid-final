"""Retrieve the four development user records unchanged from the pinned publisher file."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

REVISION = "cb04336aa1b732d4b24f5186c552456b4099806e"
SOURCE = f"https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/{REVISION}/HaluMem-Medium.jsonl"
EXPECTED = "654aaf58177b00121c52094c863af773f6bf7d833d449b1415cf86ac1062326a"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("destination", type=Path)
    path = ap.parse_args().destination
    if path.exists():
        with path.open("rb") as f:
            if hashlib.sha256(b"".join(f.readline() for _ in range(4))).hexdigest() == EXPECTED:
                return
        raise ValueError("The destination already contains different data.")
    req = Request(SOURCE, headers={"Range": "bytes=0-7340031", "User-Agent": "hybrid-final-research"})
    with urlopen(req, timeout=60) as stream:
        raw = b"".join(stream.readline(7340032) for _ in range(4))
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    with path.open("xb") as f:
        f.write(raw)


if __name__ == "__main__":
    main()
