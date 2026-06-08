# main.py — Dream Tees background-removal service
#
# A tiny FastAPI app with one job: take an image, run RemBG (u2net — the exact
# model validated in the torture-test protocol), and return a transparent PNG.
#
# The Vercel app calls POST /remove with the generated image as base64; this
# service returns the cut-out image as base64. Generation and removal stay as
# two separate steps so a removal failure never breaks generation.
#
# Endpoints:
#   GET  /        -> health check (also used to warm the service)
#   POST /remove  -> { "image": "<base64 or data URL>" } -> { "image": "data:image/png;base64,..." }

import base64
import io
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rembg import remove, new_session
from PIL import Image

app = FastAPI(title="Dream Tees RemBG Service")

# Allow the browser app (on Vercel) to call this service directly. Tighten the
# allow_origins list to your real domains once they're stable.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load the model ONCE at startup and reuse it for every request. We use the
# lightweight "u2netp" variant (~4MB vs u2net's ~170MB) so the service fits in
# Render's 512MB Starter tier. If quality on hard cases (smoke/hair/glow) isn't
# good enough, the fix is more RAM (Standard tier, 2GB) running full "u2net".
SESSION = new_session("u2netp")

_DATA_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,")


class RemoveRequest(BaseModel):
    image: str  # base64 string, with or without a data: URL prefix


def _decode_image(b64: str) -> bytes:
    """Strip an optional data-URL prefix and decode base64 to raw bytes."""
    cleaned = _DATA_URL_RE.sub("", b64.strip())
    return base64.b64decode(cleaned)


@app.get("/")
def health():
    # Hitting this endpoint also wakes the service from sleep on the free tier.
    return {"status": "ok", "service": "dream-tees-rembg", "model": "u2netp"}


@app.post("/remove")
def remove_background(req: RemoveRequest):
    try:
        raw = _decode_image(req.image)
    except Exception as e:
        return {"error": f"Could not decode image: {e}"}

    try:
        # Open and ensure RGBA.
        input_image = Image.open(io.BytesIO(raw)).convert("RGBA")

        # Cap working size to limit peak memory on the 512MB tier. gpt-image
        # output is typically 1024px; this only shrinks unusually large inputs.
        MAX_SIDE = 1024
        if max(input_image.size) > MAX_SIDE:
            input_image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

        output_image = remove(input_image, session=SESSION)

        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"image": f"data:image/png;base64,{out_b64}"}
    except Exception as e:
        # On any failure, signal the app so it can fall back to the original.
        return {"error": f"Removal failed: {e}"}
