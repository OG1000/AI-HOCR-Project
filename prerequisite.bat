@echo off
Building Environment
...
pause
python -m venv .venv
@echo off 
Environment built successfully
...
...
Attempting to install libraries
...
pause

pip install torch
pip install transformers
pip install torchvision

@echo off
...
Libraries built successfully
...
...
Done
pause