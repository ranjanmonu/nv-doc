# Running pyAerial on a GPU Platform with VS Code + Docker
## Complete Step-by-Step Guide for Beginners

---

## Before You Start — What Is What?

| Term | Plain English |
|---|---|
| **Docker** | A system that packages software into a "container" — like a self-contained box with everything pre-installed |
| **NGC** | NVIDIA GPU Cloud — NVIDIA's registry where you download their pre-built containers |
| **pyAerial** | Python interface to NVIDIA's 5G PHY algorithms, runs on GPU |
| **cuBB** | The full SDK container that contains all source code |
| **VS Code** | Code editor from Microsoft |
| **Dev Containers** | VS Code extension that lets you write code inside a Docker container as if it were local |
| **JupyterLab** | A browser-based interface for running Python notebooks |
| **GH200 / A100** | NVIDIA GPU hardware that pyAerial targets |

---

## System Requirements

Before starting, confirm your machine has:

- Ubuntu 20.04 or 22.04 (Linux host)
- NVIDIA GPU: **A100, H100, or GH200** (Compute Capability 8.0 or 9.0)
- NVIDIA GPU Driver: **535 or newer**
- At least **32 GB RAM** and **100 GB free disk**
- Internet access to download from NGC

Check your GPU driver version:
```bash
nvidia-smi
```
You should see your GPU listed with a driver version ≥ 535.

---

## PHASE 1 — Install the Prerequisites on Your Host Machine

### Step 1.1 — Install Docker

```bash
# Remove any old docker versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker CE
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Allow running docker without sudo (log out and back in after this)
sudo usermod -aG docker $USER
```

Verify Docker works:
```bash
docker run hello-world
```

### Step 1.2 — Install NVIDIA Container Toolkit

This is what lets Docker containers actually use your GPU.

```bash
# Add NVIDIA's package repository
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

Verify GPU access inside Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```
You should see your GPU listed inside the container output.

### Step 1.3 — Install HPC Container Maker (HPCCM)

This tool is needed to build the pyAerial container.

```bash
pip install hpccm
```

Verify:
```bash
hpccm --version
```

### Step 1.4 — Install VS Code

Download from https://code.visualstudio.com and install it, or:
```bash
# On Ubuntu via snap
sudo snap install --classic code
```

---

## PHASE 2 — Set Up NGC Access and Pull the cuBB Container

### Step 2.1 — Create an NGC Account

1. Go to https://ngc.nvidia.com
2. Click **Sign Up** (free account)
3. After logging in, go to **Setup → Generate API Key**
4. Copy the API key — you will need it in the next step

### Step 2.2 — Log In to NGC from Docker

```bash
docker login nvcr.io
```

When prompted:
- **Username:** Enter `$oauthtoken` (literally this string, with the dollar sign)
- **Password:** Paste your NGC API key

### Step 2.3 — Pull the cuBB Container

```bash
# Pull the main cuBB container (this is ~20–30 GB, takes a while)
docker pull nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:26-1-cubb
```

You can see download progress. When it finishes, verify:
```bash
docker images | grep aerial
```

---

## PHASE 3 — Clone the Repository and Extract Source Code

### Step 3.1 — Clone the GitHub Repository

```bash
# Go to your working directory
cd ~
mkdir aerial && cd aerial

# Clone with all submodules
git clone https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git \
    --recurse-submodules

cd aerial-cuda-accelerated-ran
```

### Step 3.2 — Copy Source Code Out of the Container

The container has additional pre-built files that are NOT in the GitHub repo.
You need to copy them out:

```bash
# Start the container temporarily in background
docker run --rm -d --name cuBB \
    nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:26-1-cubb \
    sleep infinity

# Copy the full SDK from inside the container to your local machine
docker cp cuBB:/opt/nvidia/cuBB ./cuBB

# Stop the temporary container
docker stop cuBB
```

Now you have the full SDK at `~/aerial/aerial-cuda-accelerated-ran/cuBB/`.

```bash
# Go into the SDK directory — this is your main working directory
cd cuBB
export cuBB_SDK=$(pwd)
echo $cuBB_SDK   # should print the full path
```

---

## PHASE 4 — Build the pyAerial Container

### Step 4.1 — Build the Container

This creates a new Docker container specifically for pyAerial (with Python, TensorFlow, PyTorch, Sionna, etc.).

```bash
cd $cuBB_SDK

# This script reads the cuBB container and builds a pyAerial container on top
./pyaerial/container/build.sh
```

This takes **10–20 minutes**. Watch for errors. When done, verify:
```bash
docker images | grep pyaerial
```
You should see an image named something like `pyaerial:latest`.

### Step 4.2 — Start the pyAerial Container

```bash
# This script starts the container with GPU access and mounts the SDK directory
$cuBB_SDK/pyaerial/container/run.sh
```

You are now **inside the pyAerial container**. Your prompt will change. You should see something like:
```
root@abc123:/workspace#
```

Everything from here until Phase 5 happens **inside this container**.

---

## PHASE 5 — Build and Install pyAerial Inside the Container

### Step 5.1 — Set the SDK Path

```bash
# Inside the container
export cuBB_SDK=/workspace   # or wherever the SDK was mounted by run.sh
cd $cuBB_SDK
```

> **Note:** Check where `run.sh` mounted your SDK. It usually mounts it at `/workspace`. Run `ls /workspace` to confirm you see directories like `cuPHY`, `pyaerial`, etc.

### Step 5.2 — Build with CMake

**If your machine is x86 (standard PC/server):**
```bash
cmake -Bbuild -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/x86-64 \
    -DNVIPC_FMTLOG_ENABLE=OFF \
    -DASIM_CUPHY_SRS_OUTPUT_FP32=ON

cmake --build build -t _pycuphy pycuphycpp
```

**If your machine is ARM (Grace Hopper GH200):**
```bash
cmake -Bbuild -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/grace-cross \
    -DNVIPC_FMTLOG_ENABLE=OFF \
    -DASIM_CUPHY_SRS_OUTPUT_FP32=ON

cmake --build build -t _pycuphy pycuphycpp
```

**Optional — for non-A100/H100/GH200 GPUs (e.g., RTX 4090 = CC 8.9):**
```bash
cmake -Bbuild -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/x86-64 \
    -DNVIPC_FMTLOG_ENABLE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DASIM_CUPHY_SRS_OUTPUT_FP32=ON
```

> **What compute capability is my GPU?**
> A100 = 80, H100 = 90, GH200 = 90, RTX 4090 = 89, RTX 3090 = 86

### Step 5.3 — Install pyAerial as a Python Package

```bash
./pyaerial/scripts/install_dev_pkg.sh
```

### Step 5.4 — Verify the Installation

```bash
python3 -c "import aerial; print('pyAerial import OK')"
```

If you see `pyAerial import OK` — it worked. If you see an import error, check that the cmake build completed without errors in Step 5.2.

---

## PHASE 6 — Connect VS Code to the Running Container

This is the key step that gives you a full IDE experience (autocomplete, debugging, file browsing) while running inside the Docker container.

### Step 6.1 — Install VS Code Extensions on Your Host

Open VS Code and install these extensions (click the Extensions icon on the left sidebar, search each name):

1. **Remote - Containers** (by Microsoft) — `ms-vscode-remote.remote-containers`
2. **Python** (by Microsoft) — `ms-python.python`
3. **Jupyter** (by Microsoft) — `ms-toolsai.jupyter`

### Step 6.2 — Make Sure the pyAerial Container Is Running

On your host machine (not inside the container), open a new terminal:
```bash
# Check if the container is running
docker ps

# You should see the pyaerial container listed
# If not, start it again:
$cuBB_SDK/pyaerial/container/run.sh &
```

### Step 6.3 — Attach VS Code to the Running Container

1. Open VS Code
2. Press **F1** (or Ctrl+Shift+P) to open the Command Palette
3. Type: `Dev Containers: Attach to Running Container`
4. Press Enter
5. A list of running Docker containers appears — select the **pyaerial** container
6. VS Code will open a **new window** connected to the inside of the container

You are now editing files that live **inside** the container, with full GPU access.

### Step 6.4 — Open the SDK Folder in VS Code

Inside the new VS Code window (the one connected to the container):

1. Click **File → Open Folder**
2. Navigate to `/workspace` (or wherever the SDK is mounted inside the container)
3. Click **OK**

You will now see the full `cuBB` SDK directory tree in the VS Code file explorer.

### Step 6.5 — Set the Python Interpreter

1. Press **F1** → type `Python: Select Interpreter`
2. Choose the Python that has `aerial` installed — usually `/usr/bin/python3` or the conda environment shown

---

## PHASE 7 — Run pyAerial Notebooks in VS Code or Browser

### Option A — Run in Browser via JupyterLab (Recommended for First Time)

Inside the pyAerial container terminal:
```bash
cd $cuBB_SDK/pyaerial/notebooks

# Start JupyterLab server accessible from your host browser
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

You will see output like:
```
http://127.0.0.1:8888/lab?token=abc123xyz...
```

Open that URL in your **host machine's browser**. You will see JupyterLab with all the example notebooks listed.

**Start with this notebook first:**
`example_pusch_simulation.ipynb` — runs a full PUSCH link simulation on the GPU.

### Option B — Run Notebooks Directly in VS Code

1. In VS Code (connected to the container), navigate to `pyaerial/notebooks/`
2. Click on any `.ipynb` file to open it
3. VS Code will ask you to select a kernel — choose the Python interpreter from Step 6.5
4. Run cells with **Shift+Enter**

---

## PHASE 8 — Run Your First pyAerial Script

Create a new file in VS Code at `/workspace/my_test.py`:

```python
import numpy as np
import aerial

# Check GPU is visible
import cupy as cp
print(f"GPU device: {cp.cuda.Device().id}")
print(f"GPU name: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

# Simple channel estimation test
from aerial.phy5g.algorithms import ChannelEstimator

num_rx_ant = 4
num_prbs   = 52   # 10 MHz bandwidth
num_layers = 1

# Create estimator on GPU
estimator = ChannelEstimator(
    num_rx_ant=num_rx_ant,
    ch_est_algo=1,          # Multi-stage MMSE
    enable_per_prg_chest=0,
)

print("ChannelEstimator created successfully on GPU")
print("pyAerial is working correctly!")
```

Run it from the VS Code terminal (inside the container):
```bash
python3 /workspace/my_test.py
```

Expected output:
```
GPU device: 0
GPU name: NVIDIA A100-SXM4-80GB
ChannelEstimator created successfully on GPU
pyAerial is working correctly!
```

---

## PHASE 9 — Run the Unit Tests (Optional but Recommended)

Unit tests require test vectors. Generate them first:
```bash
# Inside the container
cd $cuBB_SDK
./testBenches/phase4_test_scripts/build_aerial_sdk.sh
```

Then run unit tests:
```bash
export TEST_VECTOR_DIR=$cuBB_SDK/testVectors
$cuBB_SDK/pyaerial/scripts/run_unit_tests.sh
```

---

## Troubleshooting

### Problem: `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`
**Fix:** The NVIDIA Container Toolkit is not installed or Docker was not restarted after installing it.
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Problem: `import aerial` gives `ImportError: libcuphy.so: cannot open shared object file`
**Fix:** The cmake build did not complete or the install script was not run.
```bash
cmake --build build -t _pycuphy pycuphycpp
./pyaerial/scripts/install_dev_pkg.sh
```

### Problem: VS Code says `Dev Containers: Attach to Running Container` shows no containers
**Fix:** The pyAerial container is not running. On your host:
```bash
$cuBB_SDK/pyaerial/container/run.sh
```

### Problem: JupyterLab port 8888 is not reachable from browser
**Fix:** The `run.sh` script may not forward port 8888. Run the container manually with port forwarding:
```bash
docker run --rm -it --gpus all \
    -p 8888:8888 \
    -v $cuBB_SDK:/workspace \
    pyaerial:latest \
    bash
```
Then inside, start Jupyter:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Problem: `nvidia-smi` works on host but not inside the container
**Fix:** You are running the container without `--gpus all`. The `run.sh` script should handle this automatically. If running manually, always include `--gpus all`.

### Problem: CMake fails with `CUDA_ARCHITECTURES not set`
**Fix:** Specify your GPU's compute capability explicitly:
```bash
cmake -Bbuild -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/x86-64 \
    -DNVIPC_FMTLOG_ENABLE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="80"   # change to match your GPU \
    -DASIM_CUPHY_SRS_OUTPUT_FP32=ON
```

---

## Quick Reference — Commands You'll Use Daily

```bash
# Start the pyAerial container
$cuBB_SDK/pyaerial/container/run.sh

# Start JupyterLab inside the container
cd $cuBB_SDK/pyaerial/notebooks && jupyter lab --ip=0.0.0.0 --no-browser --allow-root

# Check what containers are running
docker ps

# Attach VS Code to the running container
# (use Command Palette: Dev Containers: Attach to Running Container)

# Run a Python script inside the container
python3 /workspace/my_script.py

# Run unit tests
export TEST_VECTOR_DIR=$cuBB_SDK/testVectors
$cuBB_SDK/pyaerial/scripts/run_unit_tests.sh

# Check GPU usage while running
watch -n 1 nvidia-smi
```

---

## Available Example Notebooks

Once JupyterLab is running, you will find these ready-to-run notebooks under `pyaerial/notebooks/`:

| Notebook | What it demonstrates |
|---|---|
| `example_pusch_simulation.ipynb` | Full PUSCH link simulation on GPU |
| `example_ldpc_coding.ipynb` | LDPC encode + decode |
| `example_srs_tx_rx.ipynb` | SRS transmit and receive pipeline |
| `example_csi_rs_tx_rx.ipynb` | CSI-RS transmit and receive |
| `channel_estimation.ipynb` | DMRS-based channel estimation (MMSE, LS, NN) |
| `example_neural_receiver.ipynb` | Full neural network PUSCH receiver |
| `llrnet_dataset_generation.ipynb` | Generate training data for LLRNet |
| `llrnet_model_training.ipynb` | Train the LLRNet model |
| `example_simulated_dataset.ipynb` | Simulate and save datasets |
| `datalake_channel_estimation.ipynb` | Run channel estimation on Data Lake IQ data |
| `datalake_pusch_decoding.ipynb` | Decode PUSCH from real captured IQ data |

**Recommended learning order for a beginner:**
1. `example_pusch_simulation.ipynb` — understand the full pipeline
2. `channel_estimation.ipynb` — understand estimation in isolation
3. `example_neural_receiver.ipynb` — see how AI replaces classical algorithms
4. `example_srs_tx_rx.ipynb` — understand SRS

## Debugging steps

# 1. TF version and GPU list
python3 -c "
import tensorflow as tf
print('TF version:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
print('Built with CUDA:', tf.test.is_built_with_cuda())
"

# 2. CUDA libraries TF is actually finding
python3 -c "
from tensorflow.python.platform import build_info
print(build_info.build_info)
"

# 3. What CUDA is installed in the container
ldconfig -p | grep libcuda
ldconfig -p | grep libcudart
ls /usr/local/cuda/lib64/libcudart* 2>/dev/null || echo "No CUDA at /usr/local/cuda"

# 4. CUDA visible devices env var
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 5. nvidia-smi from inside container
nvidia-smi

---
# Pull NVIDIA's GH200-native TF container
docker pull nvcr.io/nvidia/tensorflow:24.12-tf2-py3

# Verify GPU visibility immediately
docker run --rm --gpus all \
    nvcr.io/nvidia/tensorflow:24.12-tf2-py3 \
    python3 -c "
import tensorflow as tf
print('TF:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('GPUs:', tf.config.list_physical_devices('GPU'))
"

# Inside cubb_rrk or pyaerial container
pip uninstall tensorflow tensorflow-cpu tensorflow-cpu-aws -y

# NVIDIA's aarch64 + CUDA TF wheel
pip install --upgrade \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/ \
    nvidia-tensorflow==1.15.5+nv22.12

# Install NGC CLI
wget -O ngccli.zip \
    https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.41.4/files/ngccli_linux.zip

unzip ngccli.zip
chmod +x ngc-cli/ngc
sudo mv ngc-cli/ngc /usr/local/bin/ngc

# Configure with your API key
ngc config set   # enter your API key when prompted

# Pull the container
ngc registry image pull \
    nvcr.io/nvidia/tensorflow:24.12-tf2-py3
    
*Based on NVIDIA Aerial CUDA-Accelerated RAN official documentation, release 26-1.*
*Last updated: May 2026*
