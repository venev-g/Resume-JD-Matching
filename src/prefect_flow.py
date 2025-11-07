import asyncio
import re
import os

from prefect import task, flow, get_run_logger, runtime
from prefect.client.orchestration import get_client
from prefect.client.schemas.objects import Artifact

import ocr_utils
import scraping_utils
import gemini_analyzer
import config


def sanitize_artifact_key(input_string: str) -> str:
    """
    Sanitizes a string to be used as an artifact key.

    The function converts the input string to lowercase and replaces disallowed characters
    (anything other than a-z, 0-9, or hyphen) with a hyphen. For URLs, it attempts to extract 
    a meaningful segment (for example, the last one or two path components). The result is 
    trimmed to a maximum of 50 characters and any leading/trailing hyphens are removed.

    Args:
        input_string (str): The input string to sanitize (e.g., a URL or filename).

    Returns:
        str: A sanitized string suitable for use as an artifact key.
    """
    sanitized = input_string.lower()
    if sanitized.startswith("http"):
        try:
            from urllib.parse import urlparse
            path_parts = [p for p in urlparse(sanitized).path.split("/") if p]
            if path_parts:
                sanitized = "-".join(path_parts[-2:]) if len(path_parts) > 1 else path_parts[-1]
            else:
                sanitized = "linkedin-url"
        except Exception:
            sanitized = "linkedin-url"
    sanitized = re.sub(r"[^a-z0-9-]+", "-", sanitized)
    sanitized = sanitized.strip("-")[:50]
    return sanitized if sanitized else "sanitized-key"


@task(name="Extract Text (OCR)", retries=1, retry_delay_seconds=5)
def ocr_task(file_path: str) -> str:
    """
    Prefect task to perform OCR on a given file.

    Args:
        file_path (str): Path to the input file (image or PDF).

    Returns:
        str: The OCR extracted text; returns an empty string if no text is extracted.
    """
    logger = get_run_logger()
    file_basename = os.path.basename(file_path)
    logger.info(f"Starting OCR for file: {file_basename}")
    try:
        text = ocr_utils.extract_text_from_file(file_path)
        if not text or not text.strip():
            logger.warning(f"OCR completed for {file_basename}, but no text was extracted.")
            return ""
        logger.info(f"OCR successful for: {file_basename} ({len(text)} characters extracted)")
        return text
    except (ocr_utils.OCRError, FileNotFoundError) as e:
        logger.error(f"OCR failed for {file_basename}: {e}")
        raise


@task(name="Scrape Job Description", retries=1, retry_delay_seconds=10)
def scrape_task(url: str) -> str:
    """
    Prefect task to scrape a job description from a URL.

    Args:
        url (str): The URL to a LinkedIn job posting.

    Returns:
        str: The scraped text from the job description; an empty string if no text is extracted.
    """
    logger = get_run_logger()
    logger.info(f"Starting scraping for URL: {url}")
    try:
        text = scraping_utils.scrape_linkedin_job_description(url)
        if not text or not text.strip():
            logger.warning(f"Scraping completed for {url}, but no text was extracted.")
            return ""
        logger.info(f"Scraping successful for: {url} ({len(text)} characters extracted)")
        return text
    except (scraping_utils.ScrapingError, ValueError) as e:
        logger.error(f"Scraping failed for {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected scraping error for {url}: {e}", exc_info=True)
        raise


@task(name="Analyze with Gemini", retries=1, retry_delay_seconds=10)
async def analyze_task(resume_text: str, jd_text: str) -> dict:
    """
    Prefect task to analyze the resume and job description texts using the Gemini API.

    The task calls the Gemini analyzer, creates an artifact for the raw API response, and 
    returns the parsed analysis result.

    Args:
        resume_text (str): Extracted text from the resume.
        jd_text (str): Extracted or scraped text from the job description.

    Returns:
        dict: The analysis result as a dictionary, or an error dictionary if analysis fails.
    """
    logger = get_run_logger()
    task_run_id = runtime.task_run.id if runtime.task_run else None
    logger.info("Starting Gemini analysis...")
    raw_response_text = "Gemini call did not complete successfully."
    analysis_dict = None
    try:
        analysis_dict, raw_response_text = gemini_analyzer.analyze_resume_jd(resume_text, jd_text)
        logger.info("Creating Gemini raw response artifact via client...")
        try:
            artifact_to_create = Artifact(
                key="gemini-raw-response",
                type="markdown",
                description="The raw text response received from the Gemini API before parsing.",
                data=f"```\n{raw_response_text}\n```",
                task_run_id=task_run_id,
            )
            async with get_client() as client:
                created_artifact_response = await client.create_artifact(artifact=artifact_to_create)
                logger.info(f"Created raw response artifact with ID: {created_artifact_response.id}")
        except Exception as artifact_err:
            logger.error(f"Failed to create gemini-raw-response artifact: {artifact_err}", exc_info=True)
        logger.info("Gemini analysis successful.")
        return analysis_dict
    except (gemini_analyzer.GeminiError, ValueError) as e:
        logger.error(f"Gemini analysis failed: {e}")
        try:
            error_artifact = Artifact(
                key="gemini-error-info",
                type="markdown",
                description="Details of the Gemini analysis failure.",
                data=f"**Gemini Analysis Failed:**\n```\n{e}\n```\n**Raw response snippet (if available):**\n```\n{raw_response_text[:1000]}...\n```",
                task_run_id=task_run_id,
            )
            async with get_client() as client:
                await client.create_artifact(artifact=error_artifact)
        except Exception as artifact_err:
            logger.error(f"Failed to create gemini-error-info artifact: {artifact_err}", exc_info=True)
        raise


@flow(name="Resume Analyzer Flow")
async def resume_analyzer_flow(resume_path: str, jd_input: str) -> dict:
    """
    Orchestrates OCR/Scraping and Gemini analysis for a resume and job description.

    This flow performs the following steps:
      1. Determines if the job description (JD) input is a URL or a file path.
      2. Submits an OCR task to extract text from the resume file.
      3. Submits either a scraping task (if the JD input is a URL) or an OCR task (if it's a file path) 
         to extract text from the job description.
      4. Creates artifacts for input sources.
      5. Validates that both resume and job description texts are not empty.
      6. Submits a Gemini analysis task to compare the texts.
      7. Creates artifacts to store analysis summaries and match percentages.

    Args:
        resume_path (str): File path to the resume document.
        jd_input (str): Either a file path to the job description or a URL (e.g., LinkedIn job posting).

    Returns:
        dict: The parsed Gemini analysis result, or an error message in a dictionary if any step fails.
    """
    logger = get_run_logger()
    flow_run_id = runtime.flow_run.id if runtime.flow_run else None
    resume_basename = os.path.basename(resume_path)

    # --- Determine JD Input Type ---
    is_jd_url = jd_input.strip().startswith("http://") or jd_input.strip().startswith("https://")
    if is_jd_url:
        jd_source_display = jd_input.strip()
        logger.info("Job description source is a URL.")
    else:
        if not os.path.exists(jd_input):
            logger.error(f"Job description input '{jd_input}' is neither a valid URL nor an existing file path.")
            return {"error": f"Invalid Job Description input: '{jd_input}' is not a valid URL or file path."}
        jd_source_display = os.path.basename(jd_input)
        logger.info("Job description source is a file path.")

    logger.info(f"Starting analysis for Resume: {resume_basename}, JD Source: {jd_source_display}")

    # --- Create Input Artifacts ---
    try:
        input_artifact = Artifact(
            key="input-sources",
            type="markdown",
            description="Input sources for the analysis.",
            data=f"- **Resume:** `{resume_basename}`\n- **Job Description Source:** `{jd_source_display}`",
            flow_run_id=flow_run_id,
        )
        async with get_client() as client:
            created_artifact_response = await client.create_artifact(artifact=input_artifact)
            logger.info(f"Created input sources artifact with ID: {created_artifact_response.id}")
    except Exception as e:
        logger.error(f"Failed to create input sources artifact: {e}", exc_info=True)

    # --- Process Resume (OCR) ---
    resume_text = None
    try:
        logger.info(f"Submitting OCR task for Resume: {resume_basename}")
        resume_text = ocr_task(resume_path)
        logger.info("Resume OCR task completed.")

        if resume_text and resume_text.strip():
            try:
                artifact_key = sanitize_artifact_key(f"ocr-resume-preview-{resume_basename}")
                artifact_description = f"Preview and character count for OCR result of {resume_basename}."
                artifact_data = f"**Characters:** {len(resume_text)}\n**Preview:**\n```\n{resume_text[:500].strip()}...\n```"
                ocr_resume_artifact = Artifact(
                    key=artifact_key,
                    type="markdown",
                    description=artifact_description,
                    data=artifact_data,
                    flow_run_id=flow_run_id,
                )
                async with get_client() as client:
                    created_artifact_response = await client.create_artifact(artifact=ocr_resume_artifact)
                    logger.info(f"Created resume OCR artifact with ID: {created_artifact_response.id}")
            except Exception as e:
                logger.error(f"Failed to create resume OCR artifact: {e}", exc_info=True)
        else:
            logger.warning("Resume OCR text is empty, skipping artifact creation.")
    except Exception as e:
        logger.error("Resume OCR task failed. Aborting analysis.")
        return {"error": f"Failed to process Resume '{resume_basename}'. Check logs."}

    # --- Process Job Description (OCR or Scrape) ---
    jd_text = None
    jd_artifact_key = ""
    try:
        if is_jd_url:
            logger.info(f"Submitting scraping task for JD URL: {jd_input}")
            jd_text = scrape_task(jd_input.strip())
            logger.info("JD scraping task completed.")
            jd_artifact_key = sanitize_artifact_key(f"scrape-jd-preview-{jd_input.strip()}")
        else:
            jd_basename = os.path.basename(jd_input)
            logger.info(f"Submitting OCR task for JD File: {jd_basename}")
            jd_text = ocr_task(jd_input)
            logger.info("JD OCR task completed.")
            jd_artifact_key = sanitize_artifact_key(f"ocr-jd-preview-{jd_basename}")

        if jd_text and jd_text.strip():
            try:
                artifact_description = f"Preview and character count for JD from: {jd_source_display}"
                artifact_data = f"**Characters:** {len(jd_text)}\n**Preview:**\n```\n{jd_text[:500].strip()}...\n```"
                jd_artifact = Artifact(
                    key=jd_artifact_key,
                    type="markdown",
                    description=artifact_description,
                    data=artifact_data,
                    flow_run_id=flow_run_id,
                )
                async with get_client() as client:
                    created_artifact_response = await client.create_artifact(artifact=jd_artifact)
                    logger.info(f"Created JD text artifact with ID: {created_artifact_response.id}")
            except Exception as e:
                logger.error(f"Failed to create JD text artifact: {e}", exc_info=True)
        elif jd_text is not None:
            logger.warning(f"JD processing yielded empty text for source: {jd_source_display}")
    except Exception as e:
        logger.error(f"Failed processing Job Description from '{jd_source_display}'. Error: {e}. Aborting analysis.")
        return {"error": f"Failed to process Job Description from '{jd_source_display}'. Check logs."}

    # --- Validate Extracted Text ---
    if resume_text is None or not resume_text.strip():
        logger.error("Resume text is empty after OCR. Cannot proceed.")
        return {"error": f"Resume '{resume_basename}' text is empty after OCR."}
    if jd_text is None or not jd_text.strip():
        logger.error(f"Job Description text is empty after processing '{jd_source_display}'. Cannot proceed.")
        return {"error": f"Job Description text from '{jd_source_display}' is empty after processing."}

    # --- Run Gemini Analysis ---
    analysis_result = None
    try:
        logger.info("Submitting Gemini Analysis task...")
        analysis_result = await analyze_task(resume_text, jd_text)
        logger.info("Gemini Analysis task completed.")

        if analysis_result and isinstance(analysis_result, dict) and "error" not in analysis_result:
            try:
                match_percentage = analysis_result.get("percentage", 0)
                async with get_client() as client:
                    analysis_summary_data = (
                        f"### Analysis Summary\n\n"
                        f"**Overall Match:** {match_percentage}%\n\n"
                        f"**Resume Summary Snippet:**\n```\n{analysis_result.get('resume_summary', 'N/A')[:200]}...\n```\n\n"
                        f"**JD Summary Snippet:**\n```\n{analysis_result.get('jd_summary', 'N/A')[:200]}...\n```\n\n"
                        f"**Requirements Met Snippet:**\n```\n{analysis_result.get('matches', 'N/A')[:200]}...\n```\n\n"
                        f"**Requirements Missing Snippet:**\n```\n{analysis_result.get('misses', 'N/A')[:200]}...\n```"
                    )
                    summary_artifact = Artifact(
                        key="analysis-summary",
                        type="markdown",
                        data=analysis_summary_data,
                        flow_run_id=flow_run_id,
                    )
                    created_artifact_response = await client.create_artifact(summary_artifact)
                    logger.info(f"Created analysis summary artifact with ID: {created_artifact_response.id}")

                    match_percentage_data = f"{match_percentage}%"
                    percentage_artifact = Artifact(
                        key="match-percentage",
                        type="markdown",
                        data=match_percentage_data,
                        flow_run_id=flow_run_id,
                    )
                    created_artifact_response = await client.create_artifact(percentage_artifact)
                    logger.info(f"Created match percentage artifact with ID: {created_artifact_response.id}")
            except Exception as e:
                logger.error(f"Failed to create analysis artifacts: {e}", exc_info=True)

        logger.info("Flow completed successfully.")
        return analysis_result

    except Exception as e:
        logger.error(f"Gemini Analysis task failed within the flow: {e}")
        return {"error": "Analysis task failed. Check logs and 'gemini-error-info' artifact."}


# To run the flow from CLI, uncomment the following lines:
# if __name__ == "__main__":
#     import asyncio
#     result = asyncio.run(resume_analyzer_flow("path/to/resume.pdf", "path/to/jd.pdf"))
#     print(result)