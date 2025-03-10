"""
# install packages
$ uv pip install surya-ocr streamlit pdftext


# run streamlit app

$ surya_gui

OR

$ python surya_gui.py

"""
from surya.scripts.run_streamlit_app import streamlit_app_cli

if __name__ == "__main__":
    streamlit_app_cli()
