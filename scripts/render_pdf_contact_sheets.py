"""Create labeled contact sheets from Poppler-rendered manuscript pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("page_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pages-per-sheet", type=int, default=4)
    args = parser.parse_args()

    page_paths = sorted(args.page_dir.glob("page-*.png"))
    if not page_paths:
        raise RuntimeError(f"No page PNGs found in {args.page_dir}")
    if args.pages_per_sheet != 4:
        raise ValueError("This QA renderer currently uses a fixed 2x2 layout")

    output_dir = args.output_dir or args.page_dir / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=22)
    thumb_width = 650
    label_height = 38
    gutter = 24

    for sheet_index in range(0, len(page_paths), 4):
        group = page_paths[sheet_index : sheet_index + 4]
        rendered: list[tuple[Path, Image.Image]] = []
        for path in group:
            with Image.open(path) as source:
                page = source.convert("RGB")
            height = round(page.height * thumb_width / page.width)
            rendered.append((path, page.resize((thumb_width, height), Image.Resampling.LANCZOS)))

        thumb_height = max(image.height for _, image in rendered)
        canvas = Image.new(
            "RGB",
            (2 * thumb_width + 3 * gutter, 2 * (thumb_height + label_height) + 3 * gutter),
            "#d8dde6",
        )
        draw = ImageDraw.Draw(canvas)
        for offset, (path, page) in enumerate(rendered):
            row, col = divmod(offset, 2)
            x = gutter + col * (thumb_width + gutter)
            y = gutter + row * (thumb_height + label_height + gutter)
            draw.text((x, y), path.stem.replace("page-", "Page "), fill="#111827", font=font)
            canvas.paste(page, (x, y + label_height))
        first = sheet_index + 1
        last = sheet_index + len(group)
        canvas.save(output_dir / f"contact-{first:02d}-{last:02d}.png", optimize=True)

    print(
        "PDF_CONTACT_SHEETS_READY",
        {"pages": len(page_paths), "sheets": (len(page_paths) + 3) // 4, "output": str(output_dir)},
    )


if __name__ == "__main__":
    main()
