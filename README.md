# VLM maneuver decision demo

This repository contains a small Python demo that uses the Hugging Face
LLaVA 1.5 7B vision-language model (VLM) to choose a driving maneuver from an image.
The VLM interprets the frontal camera view, describes the road scene, and recommends whether to go or stop.

## Files

- `Dockerfile` builds a CUDA-enabled Python environment with pinned PyTorch,
  Transformers, and model dependencies.
- `decide.py` defines `ManeuverDecider`, which loads the model and exposes
  `decide()` for image-based inference. Its `main()` function is a runnable
  example with a sample image path.
- `llava_decision.png` is an example visualization of a model decision.

## Clone the repository

```bash
git clone https://github.com/giovannilucente/vlm-driver.git
cd vlm-driver
```

## Requirements

- Docker with NVIDIA GPU support configured (including NVIDIA Container
  Toolkit).
- An NVIDIA GPU with enough memory for LLaVA 1.5 7B in FP16.
- Network access the first time the model is loaded, so Hugging Face can
  download the model files.

The class places the model and input tensors on CUDA; CPU-only inference is not
configured.

## Container environment

The Dockerfile pins the main environment as follows:

- Base image: NVIDIA CUDA 11.4.3 with cuDNN 8 runtime on Ubuntu 20.04.
- Python and pip are installed from the Ubuntu image; pip, setuptools, and
  wheel are pinned.
- PyTorch 1.12.1, torchvision 0.13.1, and torchaudio 0.12.1 are installed
  using the PyTorch CUDA 11.3 (`cu113`) builds where provided.
- Transformers 4.45.2 and Accelerate 0.27.2 are installed, along with pinned
  model dependencies, including Jinja2 3.1.6 for chat-template processing.

It is a nice example of running LLaVA on a relatively old CUDA-era stack. More
precisely, the base image is CUDA 11.4.3, while the PyTorch packages target
CUDA 11.3; it is therefore more accurate to describe this as a CUDA 11.4
container running a `cu113` PyTorch build, rather than as PyTorch built for
CUDA 11.4. GPU access also depends on a compatible NVIDIA driver on the host
and NVIDIA container support being configured.

## Build the image

Run these commands from the repository directory:

```bash
docker build -t vlm .
```

## Run the container

Mount the repository so the script and any output files are available on the
host. Optionally mount a directory containing input images at `/data`:

```bash
docker run --rm -it \
  --gpus all \
  --shm-size=30g \
  -v "$PWD:/workspace" \
  -v "/path/to/your/images:/data:ro" \
  -w /workspace \
  vlm bash
```

Replace `/path/to/your/images` with the directory containing your image files.
The image directory is mounted read-only. You can omit that mount if you arrange
for images to be available in the repository or another mounted directory.

Inside the container, Python code can use `/data/<image-name>` for a mounted
image. The model is downloaded and cached on first use; the container above
does not persist that cache after it is removed.

## Use `ManeuverDecider`

Create one decider and reuse it for multiple images. Model loading happens when
the instance is created, so a ROS node should generally create the instance
once (for example, during node initialization) and call `decide()` for each
incoming image.

### Pass an image path

```python
from decide import ManeuverDecider

decider = ManeuverDecider()
decision, image = decider.decide("/data/front_left.png")
print("Decision:", decision)

decider.save_visualization(decision, image, "llava_decision.png")
```

`decide()` accepts a string path or a `pathlib.Path`. The returned value is a
pair: the decision text and the RGB PIL image used for inference.

### Pass a PIL image

```python
from PIL import Image

from decide import ManeuverDecider

decider = ManeuverDecider()

with Image.open("/data/front_left.png") as input_image:
    decision, image = decider.decide(input_image)

print("Decision:", decision)
decider.save_visualization(decision, image, "llava_decision.png")
```

The PIL image is converted to RGB inside `decide()`. This form is useful when
another component, such as a ROS image bridge, has already produced a PIL
image.

### Run the example entry point

```bash
python decide.py
```

The example path is defined in `main()` in `decide.py`; update it to the path
where your sample image is available inside the container before using this
entry point. For custom paths or images supplied by another application, use
the class examples above instead.
