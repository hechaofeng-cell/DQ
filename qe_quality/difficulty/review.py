"""Contact sheets for the sampled label audit; never fed back as model inputs."""

import io
import zipfile
from pathlib import Path

from .io import new_output, read_csv, read_json, seal_run, verify_run, write_csv


def contact_sheets(prepared, output):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    prepared = Path(prepared).resolve()
    verify_run(prepared)
    config = read_json(prepared / "config.json")
    queue = sorted(read_csv(prepared / "review_queue.csv"), key=lambda r: r["original_id"])
    names = {c["class_id"]: c["class_name"] for c in config["classes"]}
    output = new_output(output, [prepared, Path(config["prediction_dir"]).parent])
    font = ImageFont.load_default(size=16)
    mapping = []
    with zipfile.ZipFile(config["archive"]) as archive:
        for start in range(0, len(queue), 12):
            page = Image.new("RGB", (1200, 1200), "white")
            draw = ImageDraw.Draw(page)
            for offset, row in enumerate(queue[start:start + 12]):
                x, y = (offset % 3) * 400, (offset // 3) * 300
                with Image.open(io.BytesIO(archive.read(row["path"]))) as image:
                    thumb = ImageOps.contain(image.convert("RGB"), (390, 235))
                    page.paste(thumb, (x + (400 - thumb.width) // 2, y))
                draw.text((x + 5, y + 237), f"{start + offset + 1}: {row['original_id']}", fill="black", font=font)
                draw.text((x + 5, y + 258), names[row["inherited_class_id"]], fill="black", font=font)
                draw.text((x + 5, y + 278), row["review_reasons"].replace("stratified_random", "random")[:44], fill="black", font=font)
                mapping.append({"number": start + offset + 1, "original_id": row["original_id"],
                                "sheet": f"sheet_{start // 12 + 1:02d}.jpg"})
            page.save(output / f"sheet_{start // 12 + 1:02d}.jpg", quality=95)
    write_csv(output / "index.csv", mapping, ["number", "original_id", "sheet"])
    seal_run(output)
    return output
