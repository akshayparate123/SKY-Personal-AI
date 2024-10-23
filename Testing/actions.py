import mss
import mss.tools
import time
import cv2
import pytesseract
from PIL import Image
from datetime import datetime
from googlesearch import search

import requests
from bs4 import BeautifulSoup
import time
from threading import Thread
import math
import random
import chromadb
import re
chroma_client = chromadb.PersistentClient(path="./")
collection = chroma_client.get_or_create_collection(name="my_collection")
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

def clean_latex(text):
    # Remove LaTeX commands like {\\displaystyle}, {\\text{}}, etc.
    cleaned_text = re.sub(r'\\displaystyle|\\text\{.*?\}', '', text)
    
    # Remove any LaTeX curly braces and unnecessary whitespaces
    cleaned_text = re.sub(r'\\[a-z]+|{|}', '', cleaned_text)
    
    # Replace LaTeX-specific representations like \\dots with their equivalent
    cleaned_text = re.sub(r'\\dots', '...', cleaned_text)
    
    # Remove multiple spaces introduced by LaTeX removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
   

def create_chunks(cleaned_data):
    ids_list = []
    final_chunks = []
    random_number = random.randint(0,10000000000)
    loop = math.ceil(len(cleaned_data)/2000)
    for i in range(0,loop):
        if i ==0:
            final_chunks.append(cleaned_data[(i*2000):(i+1)*2000])
        else:
            final_chunks.append(cleaned_data[(i*2000)-500:(i+1)*2000])
        ids_list.append(str(random_number+(i/50)))
    return (final_chunks,ids_list)
    # return final_chunks
   
def data_mining(website):
    chunks = ""
    filtered_content = ""
    if (".gov" not in website) and ("linkedin.com" not in website) and ("reddit.com" not in website):
        URL = website
        r = requests.get(URL) 
        soup = BeautifulSoup(r.content, 'html5lib')
        for tag in soup(['nav', 'header', 'footer', 'script', 'style', 'aside']):
            tag.decompose()
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li','strong']):
            filtered_content = filtered_content+tag.get_text()
        remove_latex = clean_latex(filtered_content)
        chunks = create_chunks(remove_latex)
    return chunks  
     
def get_google_search_links(query):
    return [link for link in search(query)]


def fetch_from_internet(user_query):
    start_time = time.time()
    website_links = get_google_search_links(user_query)
    threads_list = [ThreadWithReturnValue(target=data_mining, args=(website,)) for website in website_links[:5]]
    [thread.start() for thread in threads_list]
    fetched_data = [thread.join() for thread in threads_list]
    # print("Query : {}".format(q))
    final_chunks = []
    final_ids = []
    for f in fetched_data:
        if len(f) == 0:
            continue
        for i in f[0]:
            final_chunks.append(i)
        for i in f[1]:
            final_ids.append(i)

    collection.add(
        documents=final_chunks,
        ids=final_ids
    )
    results = collection.query(
        query_texts=user_query, # Chroma will embed this for you
        n_results=2 # how many results to return
    )
    print(results['distances'])
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")    
    
    return results

