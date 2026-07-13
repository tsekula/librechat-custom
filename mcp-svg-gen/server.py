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

    Returns two attachments: a rasterized PNG preview (rendered inline in chat)
    and the original SVG (LibreChat forces SVG attachments to download rather
    than render inline, since SVGs can embed scripts -- deliberate XSS
    protection, not a bug). Both are attached directly without depending on
    the chat model to relay them, since large tool-result text is often
    dropped or paraphrased by the model instead of reproduced verbatim.
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

    return [
        Image(data=png_bytes, format="png"),
        Image(data=svg_bytes, format="svg+xml"),
    ]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
