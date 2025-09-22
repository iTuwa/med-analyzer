# app/utils.py
from io import BytesIO
import base64
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

def draw_boxes_on_image(image_bytes: bytes, regions: List[Dict[str, Any]], outline_width: int = 3) -> bytes:
    """
    Draw bounding boxes and labels on the image and return PNG bytes.

    regions: list of dicts with keys:
        - 'bbox': [x_min, y_min, x_max, y_max]
        - 'label': optional string
        - 'score': optional float (0-1)
    """
    # Load image
    with BytesIO(image_bytes) as bio:
        img = Image.open(bio).convert("RGBA")

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    w, h = img.size

    for region in regions:
        bbox = region.get("bbox") or region.get("box")  # be lenient about key name
        if not bbox or len(bbox) != 4:
            continue
        # Ensure ints and clamp to image size
        x_min = max(0, int(bbox[0]))
        y_min = max(0, int(bbox[1]))
        x_max = min(w, int(bbox[2]))
        y_max = min(h, int(bbox[3]))

        # Draw rectangle (RGBA: we draw outline and a translucent fill)
        rect_color = (255, 0, 0, 180)  # red with some alpha
        outline_color = (255, 0, 0, 255)
        # translucent fill (slightly)
        fill_color = (255, 0, 0, 40)

        # create overlay for translucent fill
        overlay = Image.new("RGBA", img.size, (255,255,255,0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x_min, y_min, x_max, y_max], fill=fill_color)
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        # draw thicker border by drawing multiple rectangles
        for i in range(outline_width):
            draw.rectangle([x_min - i, y_min - i, x_max + i, y_max + i], outline=outline_color)

        # label + score
        label = region.get("label", "Concern")
        score = region.get("score")
        if score is not None:
            try:
                score_display = f"{float(score)*100:.0f}%"
            except Exception:
                score_display = str(score)
            text = f"{label} ({score_display})"
        else:
            text = label

        # text background
        text_size = draw.textsize(text, font=font) if font else draw.textsize(text)
        text_w, text_h = text_size
        text_x = x_min
        text_y = max(0, y_min - text_h - 4)

        # small background rectangle behind text for readability
        draw.rectangle([text_x, text_y, text_x + text_w + 6, text_y + text_h + 4], fill=(0,0,0,160))
        draw.text((text_x + 3, text_y + 2), text, fill=(255,255,255,255), font=font)

    # Save to PNG bytes
    with BytesIO() as out_bio:
        # convert back to RGB to drop alpha if you prefer JPEG, but use PNG to preserve overlays
        img.convert("RGB").save(out_bio, format="PNG")
        out_bytes = out_bio.getvalue()

    return out_bytes

def encode_image_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    """
    Return a data URL (base64) string for embedding in JSON or HTML.
    Example: data:image/png;base64,AAA...
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"
