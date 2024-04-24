# Install dependencies into the .venv directory
# No flash_attn in requirements.txt to avoid order related installation error
pip install -r requirements.txt

# Activate the virtual environment
source .venv/bin/activate

# Install flash_attn from PyPI
pip install flash-attn --no-build-isolation
# Other option: Install flash_attn from source
# mkdir -p .tmp
# cd .tmp
# git clone git@github.com:Dao-AILab/flash-attention.git
# cd flash-attention
# python setup.py install
