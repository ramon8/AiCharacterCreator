import numpy as np
from PIL import Image

from make.texturing.psx import reduce_to_psx


def colors(image):
    return {c for _, c in image.convert("RGB").getcolors(maxcolors=1 << 24)}


def noisy(size=512, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def test_reduces_to_128_px_with_at_most_256_colors():
    texture = reduce_to_psx(noisy())
    assert texture.size == (128, 128)
    assert len(colors(texture)) <= 256


def test_flat_regions_keep_their_exact_colors_without_dithering():
    # Two flat halves aligned to the 128 px grid: no new (blended or dithered) colors may appear.
    image = Image.new("RGB", (512, 512), (200, 30, 40))
    image.paste((20, 60, 200), (0, 0, 256, 512))
    texture = reduce_to_psx(image)
    assert colors(texture) == {(20, 60, 200), (200, 30, 40)}
    assert texture.getpixel((63, 64)) == (20, 60, 200)
    assert texture.getpixel((64, 64)) == (200, 30, 40)


def test_is_deterministic():
    a, b = reduce_to_psx(noisy(seed=3)), reduce_to_psx(noisy(seed=3))
    assert a.tobytes() == b.tobytes()
