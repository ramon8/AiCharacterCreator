"""Final PSX reduction of a baked texture: resolution and palette.

Spike values, fixed in code until the PSX contract file arrives (#2, wired in #11).
"""

from __future__ import annotations

from PIL import Image

TEXTURE_SIZE = 128
MAX_COLORS = 256


def reduce_to_psx(texture: Image.Image, size: int = TEXTURE_SIZE, colors: int = MAX_COLORS) -> Image.Image:
    """Downscale to `size` px and quantize to at most `colors` colors, no dithering.

    The downscale here is an area average (`Image.Resampling.BOX`), so a texel is
    the mean of the bake texels it covers instead of an aliased point sample; it
    is *not* a nearest-neighbour resize. "Nearest" in the PSX contract is how the
    texture is *sampled* at render time: the material's Image Texture
    interpolation, set to Closest when `pack` puts the texture into the .blend.
    """
    small = texture.convert("RGB").resize((size, size), Image.Resampling.BOX)
    quantized = small.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return quantized.convert("RGB")
