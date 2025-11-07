import os
import re
import time
from urllib.parse import urlparse, parse_qs

from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    StaleElementReferenceException,
)
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from dotenv import load_dotenv

load_dotenv()


try:
    import html
except ImportError:
    html = None


class ScrapingError(Exception):
    """Custom exception for scraping failures."""
    pass


def _close_overlay(driver, locator, description, timeout=5):
    """
    Closes an overlay element identified by the given locator.

    Args:
        driver: Selenium WebDriver instance.
        locator: Tuple containing the locating strategy and locator.
        description: Description of the overlay element.
        timeout: Time to wait for the overlay to become clickable.

    Returns:
        True if the overlay was successfully closed, False otherwise.
    """
    try:
        close_button = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(locator)
        )
        close_button = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable(locator)
        )
        try:
            driver.execute_script("arguments[0].click();", close_button)
        except Exception:
            close_button.click()
        time.sleep(1)
        return True
    except TimeoutException:
        return False
    except Exception:
        return False


def _basic_html_cleanup(html_content: str) -> str:
    """
    Performs basic HTML cleanup on the provided HTML content.

    Args:
        html_content: Raw HTML content.

    Returns:
        Cleaned text string.
    """
    if not html_content:
        return ""
    text = re.sub(r'<br\s*/?>', '\n', html_content, flags=re.IGNORECASE)
    text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<li.*?>', '* ', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</h[1-6]>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<strong>', '**', text, flags=re.IGNORECASE)
    text = re.sub(r'</strong>', '**', text, flags=re.IGNORECASE)
    text = re.sub(
        r'<a.*?href="(.*?)".*?>(.*?)</a>', r'\2 (\1)', text, flags=re.IGNORECASE
    )
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    if html:
        text = html.unescape(text)
    return text.strip()


def scrape_linkedin_job_description(url: str, wait_time: int = 15) -> str:
    """
    Scrapes the job description text from a LinkedIn job posting URL.
    Handles search URLs by converting them to direct view URLs.
    Loads the ChromeDriver path from a .env file.

    Args:
        url: The URL of the LinkedIn job posting (can be search or view URL).
        wait_time: Max time (in seconds) to wait for elements.

    Returns:
        The extracted and partially cleaned job description text.

    Raises:
        ScrapingError: If scraping fails for any reason.
        ValueError: If the input URL format is invalid or missing necessary components.
    """
    target_url = url
    try:
        parsed_url = urlparse(url)
        if "linkedin.com" not in parsed_url.netloc:
            raise ValueError("URL does not appear to be a valid LinkedIn URL.")

        if parsed_url.path.startswith("/jobs/search/"):
            query_params = parse_qs(parsed_url.query)
            job_id_list = query_params.get("currentJobId", [])
            if job_id_list:
                job_id = job_id_list[0]
                target_url = f"https://www.linkedin.com/jobs/view/{job_id}/"
            else:
                raise ValueError("Search URL provided, but 'currentJobId' parameter is missing.")
        elif not parsed_url.path.startswith("/jobs/view/"):
            raise ValueError("URL does not appear to be a LinkedIn jobs/search or jobs/view URL.")
    except ValueError as e:
        raise ValueError(f"Invalid LinkedIn URL format: {e}") from e

    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )

    driver = None
    try:
        driver_path = os.getenv("CHROMEDRIVER_PATH")
        if not driver_path or not os.path.exists(driver_path):
            error_msg = (
                f"ChromeDriver path specified in .env (CHROMEDRIVER_PATH) is invalid or not found: {driver_path}. "
                "Ensure the path is correct."
            )
            raise ScrapingError(error_msg)

        service = ChromeService(executable_path=driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        driver.maximize_window()

        driver.get(target_url)
        time.sleep(3)

        _close_overlay(driver, (By.CSS_SELECTOR, 'button[action-type="DENY"]'), "Cookie Reject")
        modal_close_locator = (By.XPATH, "//button[.//path[starts-with(@d, 'M20,5.32L13.32,12')]]")
        _close_overlay(driver, modal_close_locator, "Modal Close Button (SVG Path)")
        time.sleep(1)

        try:
            base_container_locator = (By.CSS_SELECTOR, "div.description__text")
            WebDriverWait(driver, 5).until(EC.presence_of_element_located(base_container_locator))
            show_more_button_locator = (By.CSS_SELECTOR, "button.show-more-less-html__button--more")
            show_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(show_more_button_locator)
            )
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", show_more_button
            )
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", show_more_button)
            time.sleep(2)
        except (TimeoutException, NoSuchElementException, Exception):
            pass

        target_content_locator = (By.CSS_SELECTOR, "div.show-more-less-html__markup")
        wait = WebDriverWait(driver, wait_time)
        content_div = wait.until(EC.visibility_of_element_located(target_content_locator))

        description_html = ""
        for attempt in range(3):
            try:
                description_html = content_div.get_attribute("innerHTML")
                if description_html:
                    break
                time.sleep(1)
            except StaleElementReferenceException:
                time.sleep(1)
                content_div = wait.until(EC.visibility_of_element_located(target_content_locator))
            except Exception as e:
                raise ScrapingError(f"Unexpected error during innerHTML extraction: {e}") from e

        if not description_html:
            try:
                description_text_fallback = content_div.text
                if description_text_fallback:
                    return description_text_fallback.strip()
                raise ScrapingError("Found content div, but both innerHTML and text were empty.")
            except Exception as fallback_e:
                raise ScrapingError(f"Error during .text fallback: {fallback_e}") from fallback_e
        else:
            cleaned_text = _basic_html_cleanup(description_html)
            return cleaned_text

    except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
        error_msg = f"Failed to find or interact with required elements on {target_url}. Error: {e}"
        try:
            if driver:
                driver.save_screenshot("linkedin_scraping_timeout_error.png")
        except Exception:
            pass
        raise ScrapingError(error_msg) from e
    except Exception as e:
        error_msg = f"An unexpected error occurred during scraping of {target_url}: {e}"
        try:
            if driver:
                driver.save_screenshot("linkedin_unexpected_error.png")
        except Exception:
            pass
        raise ScrapingError(error_msg) from e
    finally:
        if driver:
            driver.quit()