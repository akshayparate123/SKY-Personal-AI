import mss
import mss.tools
import time
import cv2
import pytesseract
from PIL import Image
from datetime import datetime
from googlesearch import search


def screenshot():
    with mss.mss() as sct:
        # Get screen dimensions
        monitor = sct.monitors[1]  # The primary monitor (1 is primary, 0 is all monitors)
        # Calculate the middle region of the screen
        screen_width = monitor['width']
        screen_height = monitor['height']

        # Base middle region dimensions
        middle_width = screen_width // 2
        middle_height = screen_height // 2

        # Define the increase size (you can adjust this value)
        increase_by = 100  # Increase the size by 100 pixels on all sides

        # Calculate new dimensions by increasing size on all sides
        new_width = middle_width + 2 * increase_by
        new_height = middle_height + 2 * increase_by

        # Ensure the new region does not exceed screen boundaries
        left = max(0, (screen_width - new_width) // 2)
        top = max(0, (screen_height - new_height) // 2)

        # Define the region of interest (ROI)
        region = {'left': left, 'top': top, 'width': new_width, 'height': new_height}
        # time.sleep(3)
        # Take a screenshot of the specified region
        screenshot = sct.grab(region)
        now = datetime.now()
        formatted_date_time = now.strftime("%d%m%y_%H%M%S")
        # Save the screenshot
        # mss.tools.to_png(screenshot.rgb, screenshot.size, output='{}_screenshot.png'.format(formatted_date_time))
        mss.tools.to_png(screenshot.rgb, screenshot.size, output='./Supporting_Docs/{}_screenshot.png'.format(formatted_date_time))
    print('Screenshot saved as {}_screenshot.png'.format(formatted_date_time))
    return '{}_screenshot.png'.format(formatted_date_time)
    
def read_text_from_image(image_path):
    # Path to the Tesseract executable (change this if needed)
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    image = cv2.imread("./Supporting_Docs/"+image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the colors
    inverted_image = cv2.bitwise_not(gray_image)
    zoom_factor = 2.0

    # Get the dimensions of the ROI


    final_text = ""
    now = datetime.now()
    formatted_date_time = now.strftime("%d%m%y_%H%M%S")
    for i in range(1,8):
        if i == 1:
            new_inv = inverted_image[(i-1)*100:i*100]
        else:
            new_inv = inverted_image[((i-1)*100)-20:i*100]
        height, width = new_inv.shape[:2]
        # Resize the image (zoom in)
        zoomed_image = cv2.resize(new_inv, (int(width * zoom_factor), int(height * zoom_factor)), interpolation=cv2.INTER_LINEAR)
        _, binary_image = cv2.threshold(zoomed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use pytesseract to do OCR on the processed image
        text = pytesseract.image_to_string(binary_image)
        
        # Print the extracted text
        # print("Extracted Text:")
        final_text = final_text+text
        # print(text)
        # Optional: Save the processed image for review
        cv2.imwrite('./Supporting_Docs/{}_processed_image_{}.png'.format(formatted_date_time,i), binary_image)
    # print(final_text)
    file_path = "./Supporting_Docs/{}_imageToText.txt".format(formatted_date_time)
    with open(file_path, "w") as file:
        file.write(final_text.replace("\n",""))
    return final_text.replace("\n","")
        
def fetch_from_internet(query):
    pass
# read_text_from_image(screenshot())