"""Link a Qwen EXL3 pack with a derived serving config and a complete index."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from regenerate_safetensors_index import build_index, write_index


def create_view(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    tensor_files = list(source.glob("*.safetensors"))
    if not tensor_files:
        raise ValueError("Source contains no safetensors files")
    output.mkdir(parents=True)
    try:
        for path in sorted(source.iterdir()):
            if not path.is_file() or path.name in {
                "config.json", "quantization_config.json", "pack_scan.json",
                "model.safetensors.index.json", "serving-view.json",
                "config.json.native", "model.safetensors.index.json.native",
            }:
                continue
            (output / path.name).symlink_to(path)
        write_index(output, build_index(output))
        shutil.copyfile(source / "config.json", output / "config.json")
        tools = Path(__file__).resolve().parent
        subprocess.run([sys.executable, str(tools / "qwen_pack_scan.py"), str(output)], check=True)
        subprocess.run([sys.executable, str(tools / "qwen_pack_config.py"), str(output)], check=True)
        config = json.loads((output / "config.json").read_text())
        quantization = config["quantization_config"]
        (output / "quantization_config.json").write_text(json.dumps(quantization, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return {"view": str(output), "tensor_files": len(tensor_files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(create_view(args.source, args.output)))


if __name__ == "__main__":
    main()
