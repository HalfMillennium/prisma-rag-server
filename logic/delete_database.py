from dotenv import load_dotenv
import os
import shutil
import logging
from logic.utils import CHROMA_PATH

def main():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logging.info(f"Cleared the database at {CHROMA_PATH}.")
