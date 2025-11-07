import asyncio
import os
from datetime import datetime

import streamlit as st
import config

try:
    from prefect_flow import resume_analyzer_flow
except ImportError as e:
    st.error(f"Failed to import backend components: {e}. Check terminal logs.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error importing backend components: {e}. Check terminal logs.")
    st.stop()

TEMP_DIR = "temp_files"


def save_uploaded_file(uploaded_file):
    """
    Saves an uploaded file to a temporary directory with a timestamped filename.

    Args:
        uploaded_file (UploadedFile): The file uploaded through Streamlit.

    Returns:
        str: Full path to the saved file.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_" for c in uploaded_file.name
    )
    file_path = os.path.join(TEMP_DIR, f"{timestamp}_{safe_filename}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def cleanup_temp_file(file_path):
    """
    Removes a temporary file from the filesystem if it exists.

    Args:
        file_path (str): Path to the temporary file to delete.
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        st.warning(f"Could not remove temporary file {file_path}: {e}")


st.set_page_config(page_title="Resume Analyzer AI", page_icon="🤖", layout="wide")

st.title("📄🤖 AI Resume Analyzer")
st.markdown(
    """
    Upload your Resume (PDF/Image) and provide **either** a Job Description file (PDF/Image)
    **or** a LinkedIn Job URL.
    """
)

api_key_valid = config.GEMINI_API_KEY and config.GEMINI_API_KEY != "mock"
if not api_key_valid:
    st.error(
        "🚨 **Error:** Gemini API Key not configured or is set to 'mock'. "
        "Please set `GEMINI_API_KEY` in your `.env` file."
    )
    st.stop()

resume_path_temporary = None
jd_path_temporary = None

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Upload Resume")
    resume_file = st.file_uploader(
        "Upload Resume File (PDF/Image)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
        key="resume_upload",
    )

with col2:
    st.subheader("2. Provide Job Description")
    jd_file = st.file_uploader(
        "Upload JD File (PDF/Image)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
        key="jd_upload",
    )
    st.markdown(
        "<p style='text-align: center; margin: 10px 0;'>OR</p>",
        unsafe_allow_html=True,
    )
    if "jd_url_value" not in st.session_state:
        st.session_state.jd_url_value = ""
    jd_url = st.text_input(
        "Enter LinkedIn Job URL",
        value=st.session_state.jd_url_value,
        key="jd_url_input",
        placeholder="e.g., https://www.linkedin.com/jobs/view/..."
    )

    if jd_file and jd_url and jd_url.strip().startswith("http"):
        jd_source_option = st.radio(
            "Both an uploaded file and a URL were provided. Please choose the source for analysis:",
            options=["Uploaded File", "Provided URL"],
            key="jd_source_option",
        )
        st.session_state.jd_source = jd_source_option
    elif jd_file:
        st.session_state.jd_source = "Uploaded File"
    elif jd_url and jd_url.strip().startswith("http"):
        st.session_state.jd_source = "Provided URL"
    else:
        st.session_state.jd_source = None

if st.session_state.jd_source == "Uploaded File":
    final_jd_input_flag = "file"
elif st.session_state.jd_source == "Provided URL":
    final_jd_input_flag = "url"
else:
    final_jd_input_flag = None

jd_input_provided = final_jd_input_flag is not None
analyze_ready = resume_file is not None and jd_input_provided

if not analyze_ready:
    if resume_file is None and (jd_file or (jd_url and jd_url.strip().startswith("http"))):
        st.warning("Please upload your resume.")
    if resume_file is not None and not jd_input_provided:
        st.warning("Please upload a Job Description file OR enter a valid LinkedIn Job URL.")

analyze_button = st.button("✨ Analyze Now", disabled=not analyze_ready)

if analyze_button:
    st.markdown("---")
    st.subheader("⚙️ Processing...")

    final_resume_path = None
    final_jd_input = None

    try:
        with st.spinner("Handling resume file..."):
            if resume_file:
                resume_path_temporary = save_uploaded_file(resume_file)
                final_resume_path = resume_path_temporary
                st.write(f"Using file '{os.path.basename(resume_path_temporary)}' for resume processing.")
            else:
                st.error("Resume file missing after clicking Analyze.")
                st.stop()

        with st.spinner("Handling job description source..."):
            if final_jd_input_flag == "file":
                jd_path_temporary = save_uploaded_file(jd_file)
                final_jd_input = jd_path_temporary
                st.write(f"Using file '{os.path.basename(jd_path_temporary)}' for job description processing.")
            elif final_jd_input_flag == "url":
                final_jd_input = jd_url.strip()
                st.write(f"Using URL '{final_jd_input}' for job description processing.")
            else:
                st.error("Job Description source missing.")
                st.stop()

        if final_resume_path and final_jd_input:
            with st.spinner("Performing OCR/Scraping and AI Analysis..."):
                analysis_result = asyncio.run(
                    resume_analyzer_flow(resume_path=final_resume_path, jd_input=final_jd_input)
                )
        else:
            st.error("Input processing failed before starting analysis.")
            st.stop()

        st.subheader("📊 Analysis Results")
        if analysis_result and isinstance(analysis_result, dict):
            if "error" in analysis_result:
                st.error(f"Analysis failed: {analysis_result['error']}")
            else:
                st.markdown("#### Summaries")
                with st.expander("Resume Summary"):
                    st.markdown(analysis_result.get("resume_summary", "Not available."))
                with st.expander("Job Description Summary"):
                    st.markdown(analysis_result.get("jd_summary", "Not available."))

                st.markdown("#### Requirements Match")
                col_match, col_miss = st.columns(2)
                with col_match:
                    st.success("**✅ Requirements Met:**")
                    st.markdown(analysis_result.get("matches", "Not available."))
                with col_miss:
                    st.warning("**⚠️ Requirements Missing/Not Clear:**")
                    st.markdown(analysis_result.get("misses", "Not available."))

                st.markdown("#### Estimated Match Score")
                percentage = analysis_result.get("percentage", 0)
                st.progress(percentage / 100.0)
                st.metric(label="Overall Match Percentage", value=f"{percentage}%")
                st.success("Analysis complete!")
        else:
            st.error("Analysis failed or returned unexpected results.")
            st.write(f"Received from flow: {analysis_result}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        if resume_path_temporary:
            cleanup_temp_file(resume_path_temporary)
        if jd_path_temporary:
            cleanup_temp_file(jd_path_temporary)

st.markdown("---")
st.caption("Powered by Streamlit, Prefect, Google Gemini, Tesseract OCR, and Selenium.")