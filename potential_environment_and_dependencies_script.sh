#!/usr/bin/env bash

# Exit immediately on error
set -e

# Ensure goinfre user directory exists
GOINFRE_USER_DIR="/goinfre/${USER}"
mkdir -p "${GOINFRE_USER_DIR}"

# Define target venv location inside goinfre
PROJECT_NAME="$(basename "$(pwd)")"
GOINFRE_VENV="${GOINFRE_USER_DIR}/venvs/${PROJECT_NAME}_venv"

# Create target directory for venvs
mkdir -p "${GOINFRE_USER_DIR}/venvs"

# Set uv cache to goinfre as well to avoid filling up ~
export UV_CACHE_DIR="${GOINFRE_USER_DIR}/.cache/uv"

# Set HF cache to goinfre to avoid filling up ~ with LLM models
export HF_HOME="${GOINFRE_USER_DIR}/.cache/huggingface"

echo "==> Creating virtual environment in ${GOINFRE_VENV}..."
uv venv --clear "${GOINFRE_VENV}"

# Symlink local .venv to goinfre venv
if [ -L ".venv" ] || [ -e ".venv" ]; then
    rm -rf .venv
fi

ln -s "${GOINFRE_VENV}" .venv
echo "==> Linked .venv -> ${GOINFRE_VENV}"

echo "==> Syncing dependencies with uv..."
UV_PROJECT_ENVIRONMENT="${GOINFRE_VENV}" uv sync

echo "==> Done! Virtual environment is set up on goinfre."