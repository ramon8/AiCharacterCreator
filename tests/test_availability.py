import pytest

from make.comfyui.availability import ComfyUIUnavailable, check


def test_passes_when_everything_is_installed(comfy):
    check(comfy.url)


def test_reports_server_not_responding():
    with pytest.raises(ComfyUIUnavailable, match=r"not responding at http://127\.0\.0\.1:1"):
        check("http://127.0.0.1:1")


def test_names_a_missing_model(comfy):
    comfy.models["controlnet"].remove("controlnet-union-sdxl-promax.safetensors")
    with pytest.raises(ComfyUIUnavailable) as error:
        check(comfy.url)
    message = str(error.value)
    assert "missing model controlnet/controlnet-union-sdxl-promax.safetensors" in message
    assert "sd_xl_base_1.0" not in message


def test_names_the_custom_node_that_provides_a_missing_node_type(comfy):
    comfy.node_types -= {"IPAdapterAdvanced", "IPAdapterModelLoader"}
    with pytest.raises(ComfyUIUnavailable) as error:
        check(comfy.url)
    message = str(error.value)
    assert "missing node type IPAdapterAdvanced (custom node ComfyUI_IPAdapter_plus not installed)" in message
    assert "missing node type IPAdapterModelLoader" in message
    assert "missing model" not in message
