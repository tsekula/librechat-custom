import base64
import os

import requests
from mcp.server.fastmcp import FastMCP, Image

mcp = FastMCP("image-gen", host="0.0.0.0", port=8000)

MODEL = os.environ.get("IMAGE_GEN_MODEL", "x-ai/grok-imagine-image-quality")


@mcp.tool()
def generate_image(prompt: str) -> Image:
    """Generate an image from a text prompt using Grok Imagine (via OpenRouter)."""
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

    image_bytes = base64.b64decode(images[0]["b64_json"])
    return Image(data=image_bytes, format="png")


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
