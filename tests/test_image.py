import base64
import io

import numpy as np
import pytest
from PIL import Image

from jevnav.costmap_image import CELL_PIXELS, check_cell, image_tokens, png_base64, render
from jevnav.deciders import HeuristicDecider, JevDecider
from jevnav.navigator import Navigator, NavigatorConfig
from jevnav.palette import BASELINE, CELL_COLORS, COMMAND, UNKNOWN_FILL, rgb
from jevnav.prompt import EvidenceFormat

from .test_simulation import simulation_request


def pixel(image: Image.Image, row: float, col: float, cell: int = CELL_PIXELS):
    return image.getpixel((int((col + 0.5) * cell), int((row + 0.5) * cell)))


def first_cell(symbols: np.ndarray, symbol: str) -> tuple[int, int]:
    return tuple(int(value) for value in np.argwhere(symbols == symbol)[0])


def test_one_square_of_pixels_per_costmap_cell(wall_costmap):
    image = render(wall_costmap)
    assert image.size == (21 * CELL_PIXELS, 21 * CELL_PIXELS)
    assert render(wall_costmap, 16).size == (336, 336)


@pytest.mark.parametrize("symbol", ["#", "x", "+", ".", "*", "G"])
def test_cells_wear_the_viewer_colors(wall_costmap, symbol):
    symbols = wall_costmap.symbols(robot=False)
    row, col = first_cell(symbols, symbol)
    assert pixel(render(wall_costmap), row, col) == rgb(CELL_COLORS[symbol])


def test_unknown_cells_are_pale_with_a_dot(wall_costmap):
    image = render(wall_costmap)
    row, col = first_cell(wall_costmap.symbols(robot=False), "?")
    assert pixel(image, row, col) == rgb(BASELINE)
    assert pixel(image, row - 0.4, col - 0.4) == rgb(UNKNOWN_FILL)


def test_the_robot_is_an_orange_triangle_pointing_ahead(wall_costmap):
    image = render(wall_costmap)
    centre = wall_costmap.spec.center
    assert pixel(image, centre, centre) == rgb(COMMAND)
    assert pixel(image, centre - 0.6, centre) == rgb(COMMAND)
    assert pixel(image, centre + 0.45, centre - 0.45) == rgb(COMMAND)
    assert pixel(image, centre - 0.45, centre - 0.45) != rgb(COMMAND)


def test_the_image_is_sized_the_way_qwen_vl_expects(env_factory):
    assert image_tokens(21, CELL_PIXELS) == 1764
    check_cell(21, 64)
    check_cell(21, 96)
    for cell in (32, 48, 128):
        with pytest.raises(ValueError, match="use 64 or 96"):
            check_cell(21, cell)
    config = NavigatorConfig(evidence_format=EvidenceFormat.IMAGE, image_cell=32)
    with pytest.raises(ValueError, match="--image-cell 32"):
        Navigator(env_factory("open"), HeuristicDecider(), config)


def test_png_travels_as_plain_base64(wall_costmap):
    data = base64.b64decode(png_base64(wall_costmap), validate=True)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert Image.open(io.BytesIO(data)).size == (1344, 1344)


def test_image_evidence_is_the_simulation_request_plus_the_picture(commands):
    simulation = simulation_request(commands, wall=0.6)
    image = simulation_request(commands, wall=0.6, evidence_format=EvidenceFormat.IMAGE)
    assert simulation.images == ()
    assert len(image.images) == 1
    assert image.evidence.startswith(simulation.evidence + "\n")
    assert "It spans 4.2 m by 4.2 m, 0.2 m per cell" in image.evidence
    assert image.instructions == simulation.instructions
    assert image.criteria() == simulation.criteria()


def test_jev_decider_sends_the_image_only_with_image_evidence(commands):
    decider = JevDecider()
    image = simulation_request(commands, wall=0.6, evidence_format=EvidenceFormat.IMAGE)
    assert decider.request_body(image)["images"] == list(image.images)
    assert "images" not in decider.request_body(simulation_request(commands, wall=0.6))


def test_heuristic_reaches_the_goal_through_the_image_pipeline(env_factory):
    config = NavigatorConfig(hold=1.7, evidence_format=EvidenceFormat.IMAGE)
    report = Navigator(env_factory("slalom"), HeuristicDecider(), config, scenario="slalom").run()
    assert report.outcome == "arrived"
    assert report.safety_stops == 0
    assert all(record.image_png for record in report.decisions)
    assert report.trace()[0]["image_png"] == report.decisions[0].image_png
