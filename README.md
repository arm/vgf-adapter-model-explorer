# VGF Adapter for Model Explorer

VGF Adapter for [google-ai-edge/model-explorer](https://github.com/google-ai-edge/model-explorer) that enables visualization of VGF files from the [Arm ML SDK for Vulkan®](https://github.com/arm/ai-ml-sdk-for-vulkan).

![](https://raw.githubusercontent.com/arm/vgf-adapter-model-explorer/main/screenshots/vgf-adapter-readme-screenshot.png)

## Requirements

- Python <3.13, >=3.10

## Supported Platforms

- Linux x86_64

## Installation

### pip + PyPI
    pip install vgf-adapter-model-explorer

### GitHub

    PYTHON_VERSION_TAG=311 &&
    gh release download \
    --repo arm/vgf-adapter-model-explorer \
    --pattern "*py${PYTHON_VERSION_TAG}*.whl" &&
    pip install *py${PYTHON_VERSION_TAG}*.whl

Or through the [GitHub Releases](https://github.com/arm/vgf-adapter-model-explorer/releases) UI.

## Usage

Install Model Explorer:

    pip install torch ai-edge-model-explorer

Launch Model Explorer with the VGF adapter enabled:

    model-explorer --extensions=vgf_adapter_model_explorer

See the [Model Explorer wiki](https://github.com/google-ai-edge/model-explorer/wiki) for more information.

## Trademark notice
Arm® is a registered trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Khronos® and Vulkan® are registered trademarks, and SPIR-V™ is a trademark of the Khronos Group Inc.

## Contributions

We are not accepting direct contributions at this time.
If you have any feedback or feature requests, please use the repository issues section.
