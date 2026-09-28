import base64
import io

import numpy as np
from PIL import Image, ImageDraw

from .costmap import LAYER_SYMBOLS, UNKNOWN, LocalCostmap
from .palette import BASELINE, CELL_COLORS, COMMAND, SURFACE, rgb

CELL_PIXELS = 64
TOKEN_PIXELS = 32
MIN_IMAGE_TOKENS = 1024
MAX_IMAGE_TOKENS = 4096
UNKNOWN_SYMBOL = LAYER_SYMBOLS[UNKNOWN]
DOT_SHARE = 0.1
ROBOT_HALF_SHARE = 0.8


def image_tokens(size: int, cell: int) -> int:
    return (size * cell // TOKEN_PIXELS) ** 2


def check_cell(size: int, cell: int) -> None:
    tokens = image_tokens(size, cell)
    if cell % TOKEN_PIXELS or not MIN_IMAGE_TOKENS <= tokens <= MAX_IMAGE_TOKENS:
        valid = [
            pixels
            for pixels in range(TOKEN_PIXELS, 8 * TOKEN_PIXELS + 1, TOKEN_PIXELS)
            if MIN_IMAGE_TOKENS <= image_tokens(size, pixels) <= MAX_IMAGE_TOKENS
        ]
        raise ValueError(
            f"--image-cell {cell}: a {size}-cell costmap needs a multiple of {TOKEN_PIXELS} px "
            f"giving {MIN_IMAGE_TOKENS}-{MAX_IMAGE_TOKENS} image tokens, so the model sees it "
            f"unresized; use {' or '.join(map(str, valid))}"
        )


def render(costmap: LocalCostmap, cell: int = CELL_PIXELS) -> Image.Image:
    symbols = costmap.symbols(robot=False)
    palette = {symbol: rgb(color) for symbol, color in CELL_COLORS.items()}
    cells = np.array([[palette[symbol] for symbol in row] for row in symbols], dtype=np.uint8)
    image = Image.fromarray(np.repeat(np.repeat(cells, cell, axis=0), cell, axis=1))
    draw = ImageDraw.Draw(image)
    dot = max(1.0, cell * DOT_SHARE)
    for row, col in np.argwhere(symbols == UNKNOWN_SYMBOL):
        x, y = (col + 0.5) * cell, (row + 0.5) * cell
        draw.ellipse([x - dot, y - dot, x + dot, y + dot], fill=rgb(BASELINE))
    _draw_robot(draw, costmap.spec.center, cell)
    return image


def _draw_robot(draw: ImageDraw.ImageDraw, center: int, cell: int) -> None:
    middle = (center + 0.5) * cell
    half = cell * ROBOT_HALF_SHARE
    apex = (middle, middle - half)
    left = (middle - half * 0.8, middle + half * 0.7)
    right = (middle + half * 0.8, middle + half * 0.7)
    draw.polygon(
        [apex, left, right], fill=rgb(COMMAND), outline=rgb(SURFACE), width=max(1, cell // 16)
    )


def png_base64(costmap: LocalCostmap, cell: int = CELL_PIXELS) -> str:
    buffer = io.BytesIO()
    render(costmap, cell).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
