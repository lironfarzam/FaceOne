"""
Download images from Facebook profiles
====================================

This module provides functionality to download images from Facebook profiles.

Usage:
    from Facebook_profile_handling.download_images import download_user_photos

    download_user_photos(profile_url="https://www.facebook.com/liron.farzam", download_folder="./photos")

The `download_user_photos` function takes a Facebook profile URL and a download folder path as input.

Authors: Liron Farzam
"""

import json
import os
import time
from typing import List
from xml.etree.ElementTree import SubElement
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
import sys
import hashlib

# Add the parent directory to sys.path to find the utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cool_utils import load_config
from cool_utils import print_red, print_green, print_blue

print_blue("Start: Download images from Facebook profiles")

config = load_config()

# List of profile URLs to download photos from
PROFILE_URLS = [config["profile_url"]]
DOWNLOAD_PHOTO_FOLDER = config["download_photo_folder"]

# Photo section URLs to visit
PHOTO_URLS = [
    "/photos_by",
    "/photos_of",
]

# Set to store all photo links to avoid duplicate processing
processed_links = set()

# Set to store image hashes to detect duplicates
downloaded_image_hashes = set()


def get_image_hash(image_data: bytes) -> str:
    """Generate a hash of the image data to detect duplicates

    Args:
        image_data (bytes): The raw image data

    Returns:
        str: Hash of the image data
    """
    return hashlib.md5(image_data).hexdigest()


def is_duplicate_image(image_data: bytes) -> bool:
    """Check if we've already downloaded this exact image

    Args:
        image_data (bytes): The raw image data

    Returns:
        bool: True if this image has already been downloaded
    """
    image_hash = get_image_hash(image_data)
    if image_hash in downloaded_image_hashes:
        return True
    downloaded_image_hashes.add(image_hash)
    return False


# Function to scroll down the page and load more images
def scroll_and_download(
    driver: webdriver.Chrome, download_folder: str, scroll_pause_time: int = 1
) -> int:
    """Scroll down the page, load more images, and download them incrementally

    Args:
        driver (webdriver.Chrome): The Chrome driver instance
        download_folder (str): The folder to save images to
        scroll_pause_time (int, optional): The time to pause between scrolls. Defaults to 1.

    Returns:
        int: Number of new images downloaded in this scroll session
    """
    new_downloads = 0

    # First download all initially visible images
    print("Downloading initially visible images...")
    initial_links = extract_image_links(driver)
    for link in initial_links:
        if link not in processed_links:
            if download_image(link, download_folder):
                new_downloads += 1
            processed_links.add(link)
    print(f"Downloaded {new_downloads} initial photos")

    # Then do 5 scrolls, downloading new images each time
    for scroll_num in range(5):
        print(f"\nScroll {scroll_num + 1}/5")

        # Scroll down one page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)  # Wait for content to load

        # Get photos from current view
        section_links = extract_image_links(driver)
        print(f"Found {len(section_links)} potential new photos in current view")

        # Download new images
        downloads_this_scroll = 0
        for link in section_links:
            if link not in processed_links:
                if download_image(link, download_folder):
                    new_downloads += 1
                    downloads_this_scroll += 1
                processed_links.add(link)

        print(
            f"Downloaded {downloads_this_scroll} new photos in scroll {scroll_num + 1}"
        )

    return new_downloads


# Function to check if image is visible and of good size
def is_visual_image(img: SubElement) -> bool:
    """Check if an image element is likely a visual image (not icon/spacer)

    Args:
        img (SubElement): The image element to check

    Returns:
        bool: True if the image is likely a visual image, False otherwise
    """
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
def extract_image_links(driver: webdriver.Chrome) -> List[str]:
    """Extract links of visually displayed images from the page

    Args:
        driver (webdriver.Chrome): The Chrome driver instance

    Returns:
        List[str]: A list of image links
    """
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
def is_valid_photo(image_data: bytes) -> bool:
    """Check if the downloaded image data is likely a photo (not an icon)

    Args:
        image_data (bytes): The image data to check

    Returns:
        bool: True if the image is likely a photo, False otherwise
    """
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
def download_image(link: str, download_folder: str) -> bool:
    """Download a single image from the given link

    Args:
        link (str): The link to the image to download
        download_folder (str): The folder to save the images to

    Returns:
        bool: True if the image was downloaded successfully, False otherwise
    """
    try:
        response = requests.get(link)
        if response.status_code == 200:
            # Only check for duplicates if the image is valid
            if not is_valid_photo(response.content):
                return False

            # Then verify the image is a photo (not a small icon)
            if not is_valid_photo(response.content):
                return False

            # Generate base name for the file
            base_name = link.split("/")[-1].split("?")[0]

            # Check if we already have this exact image hash
            image_hash = get_image_hash(response.content)
            if image_hash in downloaded_image_hashes:
                print_blue(f"Duplicate detected, but downloading anyway: {link}")
            downloaded_image_hashes.add(image_hash)

            # Use a more unique filename to avoid collisions
            filename = os.path.join(download_folder, base_name)

            # Add a counter if filename already exists
            base_filename = filename
            counter = 1
            while os.path.exists(filename):
                name_parts = os.path.splitext(base_filename)
                filename = f"{name_parts[0]}_{counter}{name_parts[1]}"
                counter += 1

            with open(filename, "wb") as f:
                f.write(response.content)
            print_green(f"Photo downloaded: {filename}")
            return True
        else:
            print_red(f"Photo not downloaded (status code error): {link}")
            return False
    except Exception as e:
        print_red(f"Error downloading image {link}: {e}")
        return False


# Function to download images from a user's profile
def download_user_photos(profile_url: str, download_folder: str) -> None:
    """Download images from a user's profile

    Args:
        profile_url (str): The URL of the user's profile
        download_folder (str): The folder to save the images to
    """
    global processed_links, downloaded_image_hashes
    processed_links.clear()  # Reset processed links for each profile
    downloaded_image_hashes.clear()  # Reset image hashes for each profile

    # Initialize download counter
    download_count = 0

    # delete the folder if it already exists
    if os.path.exists(download_folder):
        for file in os.listdir(download_folder):
            os.remove(os.path.join(download_folder, file))
        os.rmdir(download_folder)

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
            time.sleep(10)  # Wait for the profile to load
            print("Profile section loaded: ", section_url)

            # Scroll, load and download images incrementally
            download_count += scroll_and_download(driver, download_folder)

        print_green(f"Total photos downloaded trying to download: {download_count}")
        print_green(f"Total photos downloaded: {len(os.listdir(download_folder))}")
        print_green(f"Total links processed: {len(processed_links)}")
        print_green(
            f"Percentage of photos downloaded: {download_count/len(processed_links) * 100}%"
        )

    finally:
        driver.quit()


# Example usage for multiple profiles
for profile_url in PROFILE_URLS:
    download_user_photos(profile_url, DOWNLOAD_PHOTO_FOLDER)
