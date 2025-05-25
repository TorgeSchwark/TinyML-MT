#!/usr/bin/env zsh
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate threestudio
export HF_HOME="./.cache"
cd threestudio
python launch.py "$@"
read -p "Press enter to exit"