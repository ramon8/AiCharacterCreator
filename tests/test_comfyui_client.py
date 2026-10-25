import io

import pytest
from PIL import Image, ImageChops

from make.generation import ComfyUIClient, Conditions, GenerationError


def solid(color, size=(512, 384)):
    return Image.new("RGB", size, color)


def conditions(**overrides):
    values = dict(
        depth=solid((10, 10, 10)),
        normal=solid((128, 128, 255)),
        references=(solid((255, 0, 0), (300, 300)),),
        prompt="red knight armour",
        seed=1234,
    )
    values.update(overrides)
    return Conditions(**values)


def test_returns_the_generated_image_at_the_render_size(comfy, tmp_path):
    image = ComfyUIClient(comfy.url, work_dir=tmp_path).generate(conditions())
    assert image.size == (512, 384)
    assert image.getpixel((0, 0))[:3] == comfy.output_color


def uploaded(comfy, name):
    return Image.open(io.BytesIO(comfy.uploads[name])).convert("RGB")


def node_titled(graph, title):
    return next(n for n in graph.values() if n["_meta"]["title"] == title)["inputs"]


def test_submits_prompt_seed_and_each_conditioning_image(comfy, tmp_path):
    depth, normal, reference = solid((1, 2, 3)), solid((4, 5, 6)), solid((7, 8, 9), (300, 300))
    ComfyUIClient(comfy.url, work_dir=tmp_path).generate(
        conditions(depth=depth, normal=normal, references=(reference,), prompt="blue robe", seed=42)
    )
    [graph] = comfy.prompts
    assert node_titled(graph, "input:prompt")["text"] == "blue robe"
    assert node_titled(graph, "input:seed")["seed"] == 42
    for title, image in [("input:depth", depth), ("input:normal", normal), ("input:reference:0", reference)]:
        sent = uploaded(comfy, node_titled(graph, title)["image"])
        assert ImageChops.difference(sent, image).getbbox() is None, title


def test_mask_becomes_the_alpha_channel(comfy, tmp_path):
    mask = Image.new("L", (512, 384), 0)
    mask.paste(255, (100, 100, 200, 200))
    image = ComfyUIClient(comfy.url, work_dir=tmp_path).generate(conditions(mask=mask))
    assert image.mode == "RGBA"
    assert image.getpixel((150, 150)) == (*comfy.output_color, 255)
    assert image.getpixel((10, 10))[3] == 0


def test_keeps_output_and_submitted_workflow_in_the_work_dir(comfy, tmp_path):
    ComfyUIClient(comfy.url, work_dir=tmp_path).generate(conditions())
    assert len(list((tmp_path / "generations").glob("*.png"))) == 1
    assert len(list((tmp_path / "generations").glob("*.json"))) == 1


def test_rejects_a_reference_count_the_workflow_cannot_take(comfy, tmp_path):
    with pytest.raises(ValueError, match="takes 1 reference image"):
        ComfyUIClient(comfy.url, work_dir=tmp_path).generate(conditions(references=()))


def test_reports_a_workflow_the_server_rejects(comfy, tmp_path):
    comfy.node_types.discard("IPAdapterAdvanced")
    with pytest.raises(GenerationError, match="rejected workflow"):
        ComfyUIClient(comfy.url, work_dir=tmp_path).generate(conditions())


def test_free_asks_the_server_to_drop_its_cache(comfy, tmp_path):
    ComfyUIClient(comfy.url, work_dir=tmp_path).free(unload_models=True)
    assert comfy.frees == [{"free_memory": True, "unload_models": True}]


def test_reports_a_failure_while_executing(comfy, tmp_path):
    comfy.execution_error = "CUDA out of memory"
    with pytest.raises(GenerationError, match="CUDA out of memory"):
        ComfyUIClient(comfy.url, work_dir=tmp_path).generate(conditions())
