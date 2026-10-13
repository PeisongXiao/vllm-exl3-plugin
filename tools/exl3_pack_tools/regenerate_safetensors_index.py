"""Rebuild model.safetensors.index.json from every safetensors header in a pack.

Only headers are read. The index is replaced atomically, including when the
existing index is a symlink; the symlink's target stays unchanged.
"""

import argparse
import json
import os
import struct
import sys
import tempfile
from pathlib import Path


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def build_index(pack: Path) -> dict:
    shards = sorted(pack.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No *.safetensors files under {pack}")

    weight_map = {}
    total_size = 0
    for shard in shards:
        tensors = {name: meta for name, meta in read_header(shard).items()
                   if name != "__metadata__"}
        for name, meta in tensors.items():
            if name in weight_map:
                raise ValueError(
                    f"Tensor {name} appears in both {weight_map[name]} and {shard.name}"
                )
            weight_map[name] = shard.name
            start, end = meta["data_offsets"]
            total_size += int(end) - int(start)

    return {"metadata": {"total_size": total_size}, "weight_map": weight_map}


def write_index(pack: Path, index: dict) -> None:
    index_path = pack / "model.safetensors.index.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=pack, prefix=index_path.name + ".",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, index_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        index = build_index(args.pack)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    files = set(index["weight_map"].values())
    print(f"scanned {len(files)} shard(s), {len(index['weight_map'])} tensor(s), "
          f"total_size={index['metadata']['total_size']}")
    if not args.dry_run:
        write_index(args.pack, index)
        print(f"wrote {args.pack / 'model.safetensors.index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
