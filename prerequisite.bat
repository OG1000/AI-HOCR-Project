@echo off
REM Prerequisite Running

REM Create virtual environment in 'venv' folder
python -m venv .venv

REM Activate the virtual environment
call .venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install required Python packages
pip install pillow numpy nltk symspellpy doctr[torch] transformers torch torchvision

REM Download NLTK data (words, stopwords, punkt, averaged_perceptron_tagger)
python -m nltk.downloader words stopwords punkt averaged_perceptron_tagger

echo.
echo Environment setup complete.
echo To activate the environment later, run:
echo     .venv\Scripts\activate
echo.
pause
