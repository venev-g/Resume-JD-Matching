# src/config.py

import os
import logging
from dotenv import load_dotenv

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Load .env Variables ---
logger.info("Attempting to load .env file...")
env_loaded = load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

if env_loaded:
    logger.info(".env file successfully loaded.")
else:
    logger.warning(".env file NOT found or failed to load.")

# --- Get Environment Variables ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TESSERACT_CMD = os.getenv("TESSERACT_CMD_PATH")

# --- Logging Retrieved Values ---
logger.info(f"GEMINI_API_KEY retrieved: {'SET' if GEMINI_API_KEY else 'NOT SET'}")
logger.info(f"TESSERACT_CMD_PATH retrieved: '{TESSERACT_CMD or 'Not Provided'}'")

# --- Basic Validation ---
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not defined in the .env file.")

logger.info("Finished loading configuration.")