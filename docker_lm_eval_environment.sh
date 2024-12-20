echo "setup environment in container"
# change directory and install local packages in editable mode and other dependencies
# install num_tab_qa package in editable mode
pip install -e .
cd lm-evaluation-harness
pip install -e .
bash install_missing_pip_packages.sh
pip install lm_eval[vllm]