import re
import logging

import google.generativeai as genai
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GeminiError(Exception):
    """Custom exception for Gemini API failures."""
    pass


def configure_gemini():
    """Configures the Gemini client."""
    if not config.GEMINI_API_KEY:
        raise GeminiError("Gemini API Key is not configured in .env file.")
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
    except Exception as e:
        raise GeminiError(f"Failed to configure Gemini: {e}") from e


def analyze_resume_jd(resume_text, jd_text):
    """
    Uses Gemini to compare resume and job description with increased consistency.

    Args:
        resume_text (str): Text content of the resume.
        jd_text (str): Text content of the job description.

    Returns:
        tuple[dict, str]: A tuple containing:
            - dict: The parsed analysis dictionary.
            - str: The raw text response received from the Gemini API.

    Raises:
        GeminiError: If the API call or parsing fails.
        ValueError: If input text is empty.
    """
    configure_gemini()

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")
    if not jd_text or not jd_text.strip():
        raise ValueError("Job Description text cannot be empty.")

    model = genai.GenerativeModel("gemini-2.5-flash")

    generation_config = genai.types.GenerationConfig(
        temperature=0.2,
        candidate_count=1
    )

    prompt = f"""
Analyze the following Resume and Job Description.

**Resume Text:**
--- START RESUME ---
{resume_text}
--- END RESUME ---

**Job Description Text:**
--- START JOB DESCRIPTION ---
{jd_text}
--- END JOB DESCRIPTION ---

**Analysis Task:**
1. Provide a brief summary of the candidate's profile based *only* on the resume text.
2. Provide a brief summary of the key requirements based *only* on the job description text.
3. List the key requirements *explicitly stated* in the job description that **ARE clearly met** by the candidate's resume. Be specific and quote evidence from the resume if possible. If none are met, state "None".
4. List the key requirements *explicitly stated* in the job description that **ARE potentially missing or NOT clearly met** based on the resume. Be specific. If all requirements seem met, state "None".
5. Based *only* on the comparison between the explicitly stated requirements (minimum and preferred qualifications if listed) in the Job Description and the evidence in the Resume, estimate the percentage match (0-100). Provide *only* the number, without any explanation or percentage sign.

**Output Format:**
Use the following exact markers for each section:

RESUME_SUMMARY:
[Your summary of the resume]

JD_SUMMARY:
[Your summary of the job description]

REQUIREMENTS_MET:
- [Requirement 1 met]
- [Requirement 2 met]
... or None

REQUIREMENTS_MISSING:
- [Requirement 1 missing]
- [Requirement 2 missing]
... or None

PERCENTAGE_MATCH:
[A single number between 0 and 100]
    """

    logger.info("Sending request to Gemini API with low temperature...")
    raw_text_response = ""  # Initialize response container

    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        if not response.candidates:
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", "Unknown")
            safety_ratings = getattr(feedback, "safety_ratings", [])
            logger.warning(
                f"Gemini response blocked or empty. Reason: {block_reason}. "
                f"Safety Ratings: {safety_ratings}"
            )
            raise GeminiError(f"Gemini response was blocked or empty. Reason: {block_reason}")

        raw_text_response = response.text

        analysis_result = {}
        text = raw_text_response

        summary_match = re.search(r"RESUME_SUMMARY:(.*?)JD_SUMMARY:", text, re.DOTALL | re.IGNORECASE)
        analysis_result["resume_summary"] = (
            summary_match.group(1).strip() if summary_match else "Could not parse Resume Summary."
        )

        jd_match = re.search(r"JD_SUMMARY:(.*?)REQUIREMENTS_MET:", text, re.DOTALL | re.IGNORECASE)
        analysis_result["jd_summary"] = (
            jd_match.group(1).strip() if jd_match else "Could not parse Job Description Summary."
        )

        met_match = re.search(r"REQUIREMENTS_MET:(.*?)REQUIREMENTS_MISSING:", text, re.DOTALL | re.IGNORECASE)
        analysis_result["matches"] = (
            met_match.group(1).strip() if met_match else "Could not parse Requirements Met."
        )

        missing_match = re.search(r"REQUIREMENTS_MISSING:(.*?)PERCENTAGE_MATCH:", text, re.DOTALL | re.IGNORECASE)
        analysis_result["misses"] = (
            missing_match.group(1).strip() if missing_match else "Could not parse Requirements Missing."
        )

        percentage_match = re.search(r"PERCENTAGE_MATCH:\s*(\d{1,3})", text, re.IGNORECASE)
        analysis_result["percentage"] = int(percentage_match.group(1)) if percentage_match else 0

        if not 0 <= analysis_result["percentage"] <= 100:
            logger.warning(
                f"Parsed percentage ({analysis_result['percentage']}) out of range. Setting to 0."
            )
            analysis_result["percentage"] = 0

        logger.info("Successfully parsed Gemini response.")
        return analysis_result, raw_text_response

    except Exception as e:
        if not isinstance(e, GeminiError):
            error_msg = (
                f"Failed during Gemini API call or parsing: {e}. "
                f"Raw response snippet: '{raw_text_response[:500]}...'"
            )
        else:
            error_msg = str(e)

        logger.error(error_msg)
        if "response" in locals() and hasattr(response, "prompt_feedback") and response.prompt_feedback:
            logger.info(f"Gemini Prompt Feedback: {response.prompt_feedback}")
        raise GeminiError(error_msg) from e