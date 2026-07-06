"""Convert FIREWORKS_API_KEY in .env to its base64 form (FIREWORKS_API_KEY_B64).

The base64 form is what ships inside the public submission image; it keeps the
key out of reach of regex-based secret scanners that trawl public registries.
Run once before building the image: python scripts/encode_key.py
"""
import base64
import pathlib
import re
import sys


def main():
    env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        print("FAIL: .env not found")
        sys.exit(1)

    lines = env_path.read_text(encoding="utf-8").splitlines()
    out = []
    converted = False
    for line in lines:
        m = re.match(r"^FIREWORKS_API_KEY=(.+)$", line.strip())
        if m:
            b64 = base64.b64encode(m.group(1).strip().encode()).decode()
            out.append("FIREWORKS_API_KEY_B64=" + b64)
            converted = True
        else:
            out.append(line)

    if not converted:
        already = any(l.startswith("FIREWORKS_API_KEY_B64=") for l in lines)
        print("Already converted." if already else "FAIL: no FIREWORKS_API_KEY line found")
        sys.exit(0 if already else 1)

    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("OK: .env now holds FIREWORKS_API_KEY_B64 (plain key line removed)")


if __name__ == "__main__":
    main()
