CONDA_ENV_NAME=syde577
PYTHON_VERSION=3.9

PYTORCH_VER="2.3.1"
TORCHVISION_VER="0.18.1"
TORCHAUDIO_VER="2.3.1"

# Data URLs
HAM_PART1_URL = "https://dataverse.harvard.edu/api/access/datafile/3172832"
HAM_PART2_URL = "https://dataverse.harvard.edu/api/access/datafile/3172838"
HAM_METADATA_URL = "https://dataverse.harvard.edu/api/access/datafile/3172836"

.PHONY: setup-data download-data prepare-data

setup_data: download_data prepare_data

download_data:
	@echo "📥 Downloading HAM10000 dataset..."
	@python scripts/download_data.py
	@echo "✅ Download complete!"

prepare_data:
	@echo "🔧 Preparing dataset..."
	@mkdir -p data/images
	@unzip -q -o data/raw/HAM10000_images_part1.zip -d data/images/ || true
	@unzip -q -o data/raw/HAM10000_images_part2.zip -d data/images/ || true
	@cp data/raw/HAM10000_metadata.csv data/metadata.csv
	@echo "✅ Data preparation complete!"

clean_data:
	@echo "🧹 Cleaning data directory..."
	@rm -rf data/raw
	@rm -rf data/images
	@rm -f data/metadata.csv
	@echo "✅ Data cleaned!"

generate_new_requirements:
	pip-compile requirements/requirements.in
	@echo "🎉 Generated new requirements.txt in requirements/ folder!"

setup_environment:
	conda create -y -c conda-forge --name ${CONDA_ENV_NAME} python=${PYTHON_VERSION} pip-tools

install_deps: install_torch
	pip install -r requirements/requirements.txt

install_torch:
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "Installing PyTorch for macOS"; \
		conda install pytorch==${PYTORCH_VER} torchvision==${TORCHVISION_VER} torchaudio==${TORCHAUDIO_VER} -c pytorch -y; \
	elif [ "$$(uname)" = "Linux" ]; then \
		if command -v nvidia-smi >/dev/null 2>&1; then \
			echo "Installing PyTorch with CUDA support for Linux"; \
			conda install pytorch==${PYTORCH_VER} torchvision==${TORCHVISION_VER} torchaudio==${TORCHAUDIO_VER} pytorch-cuda=11.8 -c pytorch -c nvidia -y; \
		else \
			echo "Installing PyTorch for Linux (CPU only)"; \
			conda install pytorch==${PYTORCH_VER} torchvision==${TORCHVISION_VER} torchaudio==${TORCHAUDIO_VER} cpuonly -c pytorch -y; \
		fi; \
	else \
		echo "Unsupported operating system"; \
		exit 1; \
	fi

reset_environment:
	conda remove --name ${CONDA_ENV_NAME} --all -y

format:
	black src

run_tests:
	python -m unittest discover tests