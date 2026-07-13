import base64
import os

import cairosvg
import requests
from mcp.server.fastmcp import FastMCP, Image

mcp = FastMCP("svg-gen", host="0.0.0.0", port=8000)

MODEL = os.environ.get("SVG_GEN_MODEL", "recraft/recraft-v4.1-vector")


@mcp.tool()
def generate_svg(prompt: str) -> list:
    """Generate a vector SVG image from a text prompt using Recraft (via OpenRouter).

    Returns a rasterized PNG preview (rendered inline in chat) plus the raw SVG
    markup as text, since LibreChat forces SVG attachments to download rather
    than render inline (SVGs can embed scripts, so this is a deliberate XSS
    protection, not a bug).
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    response = requests.post(
        url="https://openrouter.ai/api/v1/images",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "prompt": prompt},
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    images = result.get("data", [])
    if not images:
        raise ValueError("Image generation failed: no image returned.")

    svg_bytes = base64.b64decode(images[0]["b64_json"])
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=1024)
    svg_text = svg_bytes.decode("utf-8")

    return [
        Image(data=png_bytes, format="png"),
        f"Raw SVG source (save this as a `.svg` file to keep it vector):\n\n```svg\n{svg_text}\n```",
    ]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
