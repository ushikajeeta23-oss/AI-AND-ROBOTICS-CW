"""Reacquire exact Experiment 2 image bytes from its saved Wikimedia Commons URLs."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from scripts.experiment2_integrity import check_dataset


def restore(evidence, output, contact):
    with (evidence / "source_manifest.csv").open(encoding="utf-8", newline="") as stream:
        records = [row for row in csv.DictReader(stream) if row["new_split"]]
    for index, row in enumerate(records, 1):
        target = (output / row["new_path"].replace(chr(92), "/")).resolve()
        if not target.is_relative_to(output) or row["sha256"] != row["download_sha256"]:
            raise ValueError(f"Unsafe path or inconsistent source checksum for {row['source_page']}")
        if not target.exists():
            url = row["download_url"]
            if urlsplit(url).hostname != "upload.wikimedia.org" or not url.startswith("https://"):
                raise ValueError(f"Unexpected Wikimedia asset URL: {url}")
            request = Request(url, headers={"User-Agent": f"OfficeMateAI/2.0 ({contact})"})
            with urlopen(request, timeout=45) as response:
                data = response.read(32 * 1024 * 1024 + 1)
            if sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Source bytes changed or are unavailable: {row['source_page']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
        if sha256(target.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Existing image differs from recorded bytes: {target}")
        if index % 100 == 0:
            print(f"Verified {index}/{len(records)} files.", flush=True)

    with (evidence / "experiment2_duplicate_edges.csv").open(encoding="utf-8", newline="") as stream:
        print(check_dataset(output, list(csv.DictReader(stream))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("results/experiment2/dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/experiment2"))
    parser.add_argument("--contact", required=True, help="Contact email for the Commons HTTP User-Agent")
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / "manifest.csv").is_file():
        parser.error("Place the supplied Experiment 2 manifest.csv in the output directory first.")
    restore(args.evidence, output, args.contact)
