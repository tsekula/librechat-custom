import base64
import os

import cairosvg
import requests
from mcp.server.fastmcp import FastMCP, Image

mcp = FastMCP("image-gen", host="0.0.0.0", port=8000)

RASTER_MODEL = os.environ.get("IMAGE_GEN_MODEL", "x-ai/grok-imagine-image-quality")
VECTOR_MODEL = os.environ.get("SVG_GEN_MODEL", "recraft/recraft-v4.1-vector")


def _call_openrouter_images(model: str, prompt: str) -> bytes:
    api_key = os.environ["OPENROUTER_API_KEY"]
    response = requests.post(
        url="https://openrouter.ai/api/v1/images",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "prompt": prompt},
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    images = result.get("data", [])
    if not images:
        raise ValueError("Image generation failed: no image returned.")
    return base64.b64decode(images[0]["b64_json"])


@mcp.tool()
def generate_image(prompt: str, vector: bool = False) -> list:
    """Generate an image from a text prompt.

    By default generates a rasterized PNG (via Grok Imagine). Set vector=True
    to generate a scalable vector graphic instead (via Recraft) -- use this
    for logos, icons, and illustrations that need to be resized or edited
    without quality loss. Vector requests return both a PNG preview (rendered
    inline in chat) and the original SVG (LibreChat forces SVG attachments to
    download rather than render inline, since SVGs can embed scripts --
    deliberate XSS protection, not a bug).
    """
    if vector:
        svg_bytes = _call_openrouter_images(VECTOR_MODEL, prompt)
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=1024)
        return [
            Image(data=png_bytes, format="png"),
            Image(data=svg_bytes, format="svg+xml"),
        ]

    png_bytes = _call_openrouter_images(RASTER_MODEL, prompt)
    return [Image(data=png_bytes, format="png")]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
