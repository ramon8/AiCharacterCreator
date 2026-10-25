from PIL import Image

from make.generation import Conditions, FakeGenerationClient


def test_fake_client_returns_its_fixed_image_at_the_render_size_with_mask_alpha():
    fixed = Image.new("RGB", (64, 64), (9, 99, 199))
    mask = Image.new("L", (256, 128), 0)
    mask.paste(255, (0, 0, 128, 128))
    render = Image.new("RGB", (256, 128))
    client = FakeGenerationClient(fixed)

    image = client.generate(Conditions(depth=render, normal=render, prompt="x", seed=1, mask=mask))

    assert image.size == (256, 128)
    assert image.getpixel((10, 10)) == (9, 99, 199, 255)
    assert image.getpixel((200, 10))[3] == 0
