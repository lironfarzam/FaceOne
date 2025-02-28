print("Start: Download images from Facebook profiles")
import os
import time
import requests
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from PIL import Image
from io import BytesIO

# List of profile URLs to download photos from
PROFILE_URLS = [
    "https://www.facebook.com/noguznow",
]

PHOTO_FOLDER = "downloaded_photos"

# Photo section URLs to visit
PHOTO_URLS = [
    "/photos_by",
    "/photos_of",
]

# Set to store all photo links to avoid duplicate processing
processed_links = set()


# Function to scroll down the page to load more images
def scroll_down(driver, scroll_pause_time=2):
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    # Number of scrolls (adjust as needed)
    scrolls = 3

    for i in range(scrolls):
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(scroll_pause_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        print(f"Scrolled down {i+1} times")


# Function to check if image is visible and of good size
def is_visual_image(img):
    """Check if an image element is likely a visual image (not icon/spacer)"""
    try:
        # Check if image is displayed
        if not img.is_displayed():
            return False

        # Get dimensions
        width = int(img.get_attribute("width") or img.size["width"] or 0)
        height = int(img.get_attribute("height") or img.size["height"] or 0)

        # Filter by size - adjust thresholds as needed
        # Images that are too small are likely icons
        if width > 0 and height > 0 and (width < 100 or height < 100):
            return False

        # Check natural dimensions with JavaScript
        natural_width = int(img.get_attribute("naturalWidth") or 0)
        natural_height = int(img.get_attribute("naturalHeight") or 0)

        # If natural dimensions available, use them as better indicator
        if natural_width > 0 and natural_height > 0:
            return natural_width >= 150 and natural_height >= 150

        # If we can't get dimensions but image is displayed, consider it
        return True
    except:
        return False  # If we can't analyze the element, skip it


# Function to extract visually displayed images
def extract_image_links(driver):
    """Extract links of visually displayed images from the page"""
    all_imgs = driver.find_elements(By.TAG_NAME, "img")

    # Filter images based on visual properties
    visual_imgs = [img for img in all_imgs if is_visual_image(img)]
    print(
        f"Found {len(visual_imgs)} visually displayed images out of {len(all_imgs)} total images"
    )

    # Get links from visual images
    links = []
    for img in visual_imgs:
        src = img.get_attribute("src")
        if src and src not in processed_links:
            links.append(src)

    # Log what we found
    if links:
        print(f"Sample image URLs: {links[:3]}")

    return links


# Function to verify image quality
def is_valid_photo(image_data):
    """Check if the downloaded image data is likely a photo (not an icon)"""
    try:
        # Minimum file size (15KB) - adjust as needed
        if len(image_data) < 15 * 1024:
            return False

        # Check dimensions - photos are typically larger
        img = Image.open(BytesIO(image_data))
        width, height = img.size

        # Minimum dimensions for a useful photo
        return width >= 150 and height >= 150
    except:
        return False  # If we can't analyze the image, consider it invalid


# Function to download a single image
def download_image(link, download_folder):
    """Download a single image from the given link"""
    try:
        response = requests.get(link)
        if response.status_code == 200:
            # Verify the image is a photo (not a small icon)
            if not is_valid_photo(response.content):
                print(f"Skipping non-photo image: {link}")
                return False

            # Use a more unique filename to avoid collisions
            filename = os.path.join(
                download_folder, f"{link.split('/')[-1].split('?')[0]}"
            )

            # Add a counter if filename already exists
            base_filename = filename
            counter = 1
            while os.path.exists(filename):
                name_parts = os.path.splitext(base_filename)
                filename = f"{name_parts[0]}_{counter}{name_parts[1]}"
                counter += 1

            with open(filename, "wb") as f:
                f.write(response.content)
            print("Photo downloaded: ", filename)
            return True
        else:
            print("Photo not downloaded (status code error): ", link)
            return False
    except Exception as e:
        print(f"Error downloading image {link}: {e}")
        return False


# Function to download images from a user's profile
def download_user_photos(profile_url, download_folder):
    global processed_links
    processed_links.clear()  # Reset processed links for each profile

    # Initialize download counter
    download_count = 0

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Set up Chrome options to use the user's existing profile
    chrome_options = Options()

    # Set up the Selenium WebDriver with the user's profile
    driver = webdriver.Chrome(
        options=chrome_options
    )  # Make sure to have the Chrome WebDriver installed

    try:
        # Log in to Facebook
        driver.get("https://www.facebook.com")  # Open Facebook
        print("Please log in to Facebook and then press Enter...")
        input()  # Wait for the user to log in manually

        # Visit each photo section
        for photo_url in PHOTO_URLS:
            section_url = profile_url + photo_url
            driver.get(section_url)
            time.sleep(5)  # Wait for the profile to load
            print("Profile section loaded: ", section_url)

            # Scroll down to load more images
            scroll_down(driver)

            # Get photos from this section that we haven't processed yet
            section_links = extract_image_links(driver)
            print(f"Found {len(section_links)} potential photos in section {photo_url}")

            # Download new images
            for link in section_links:
                print("Image source: ", link)
                if download_image(link, download_folder):
                    download_count += 1
                processed_links.add(link)  # Mark as processed

        print(f"Total photos downloaded: {download_count}")
        print(f"Total links processed: {len(processed_links)}")

    finally:
        driver.quit()


# Example usage for multiple profiles
for profile_url in PROFILE_URLS:
    download_user_photos(
        profile_url, PHOTO_FOLDER
    )  # Replace with the actual profile link
