# generate_glyphs.py
# Usage:
#   python generate_glyphs.py --mode advanced --min 0 --max 100 --size 96 --out out
#   python generate_glyphs.py --mode simple   --min 1 --max 100 --size 128 --out out
#   python render_glyphs.py --mode all --min 1 --max 100 --size 192 --out out

import argparse
import os
from io import BytesIO

import mglyph as mg

# import your glyph dictionaries from a python file, e.g. glyph_set.py
# Ensure this script is in the same folder or PYTHONPATH can find it.
from glyph_set import SIMPLE_GLYPHS, ADVANCED_GLYPHS


def render_png(glyph_fn, x: float, size: int) -> bytes:
    """
    Renders a single glyph function into PNG bytes using mglyph.
    """
    result = mg.render(glyph_fn, (size, size), [x])
    pil_img = result[0]["pil"]
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-render mglyph glyphs into PNGs (x in [min,max]).")
    parser.add_argument("--mode", choices=["simple", "advanced", "all"], default="all",
                        help="Which glyph set to render.")
    parser.add_argument("--min", type=int, default=0, help="Minimum x value (inclusive).")
    parser.add_argument("--max", type=int, default=100, help="Maximum x value (inclusive).")
    parser.add_argument("--size", type=int, default=96, help="PNG width/height in pixels.")
    parser.add_argument("--out", type=str, default="out", help="Output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs.")
    args = parser.parse_args()

    if args.min > args.max:
        raise SystemExit("--min must be <= --max")

    glyphs = {}
    if args.mode in ("simple", "all"):
        glyphs.update(SIMPLE_GLYPHS)
    if args.mode in ("advanced", "all"):
        glyphs.update(ADVANCED_GLYPHS)

    safe_mkdir(args.out)

    total = 0
    failed = 0

    for glyph_name, glyph_fn in glyphs.items():
        glyph_dir = os.path.join(args.out, glyph_name)
        safe_mkdir(glyph_dir)

        for x in range(args.min, args.max + 1):
            out_path = os.path.join(glyph_dir, f"{x}.png")

            if (not args.overwrite) and os.path.exists(out_path):
                continue

            try:
                png = render_png(glyph_fn, float(x), args.size)
                with open(out_path, "wb") as f:
                    f.write(png)
                total += 1
            except Exception as e:
                failed += 1
                print(f"[FAIL] {glyph_name} x={x}: {e}")

        print(f"[OK] {glyph_name} done.")

    print(f"Rendered: {total} PNGs. Failed: {failed}. Output: {os.path.abspath(args.out)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
