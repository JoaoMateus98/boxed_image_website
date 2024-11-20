import io
import os
import logging
from google.cloud import vision, storage
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np

# Get the bucket name from the environment variable set in app.yaml
BUCKET_NAME = 'comp-399-vision'

### ONLY FOR LOCAL TESTING
# Set the path to your Google Cloud service account credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Programming\Dev\secrets\vision-text-detector-2024-0632f72c7099.json'

# Configure logging
logging.basicConfig(level=logging.INFO)

def initialize_clients():
    """Initialize Google Vision API and Cloud Storage clients."""
    vision_client = vision.ImageAnnotatorClient()
    storage_client = storage.Client()
    return vision_client, storage_client

"""calls image boxing function and returns the boxed image"""
def get_boxed_image(image, title):
    logging.info('Processing image {}'.format(title))

    # Convert image to binary data for Vision API
    image.stream.seek(0) # Reset stream pointer to the beginning
    image_content = image.read()

    vision_image = vision.Image(content=image_content)

    vision_client, _ = initialize_clients()

    response = vision_client.document_text_detection(image=vision_image)

    # Check for any errors in the response
    if response.error.message:
        logging.error('Vision API error: %s', response.error.message)
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(response.error.message)
        )

    texts = response.text_annotations

    if not texts:
        logging.info('No text detected on first attempt, preprocessing image and trying again.')

        # Preprocess the image and try text detection again
        preprocessed_image = preprocess_image_for_ocr(image_content)

        # Convert preprocessed image to bytes for Vision API
        with io.BytesIO() as output:
            preprocessed_image.save(output, format="PNG")
            preprocessed_content = output.getvalue()

        preprocessed_image_vision = vision.Image(content=preprocessed_content)
        response = vision_client.document_text_detection(image=preprocessed_image_vision)
        texts = response.text_annotations

    if texts:
        logging.info('Detected text: "%s"', texts[0].description)

        # Draw bounding boxes around detected text
        img_with_boxes = draw_bounding_boxes(image_content, texts)

        # Upload the image with bounding boxes
        upload_image(img_with_boxes, title)

        return img_with_boxes
    else:
        logging.warning('No text found in the image after preprocessing.')
        return None

# image enhancer
def preprocess_image_for_ocr(image_content):
    # Open the original image
    image_stream = io.BytesIO(image_content)
    with Image.open(image_stream) as img:
        # Convert to grayscale
        gray_img = img.convert('L')

        # Apply Gaussian blur to reduce noise
        blurred_img = gray_img.filter(ImageFilter.GaussianBlur(radius=1))

        # Increase contrast
        enhancer = ImageEnhance.Contrast(blurred_img)
        contrasted_img = enhancer.enhance(2.0)

        # Adaptive thresholding using numpy
        img_array = np.array(contrasted_img)
        mean = np.mean(img_array)
        binary_img = np.where(img_array > mean, 255, 0).astype(np.uint8)

        # Convert back to PIL image
        processed_img = Image.fromarray(binary_img)

        # Optionally sharpen the image
        sharpener = ImageEnhance.Sharpness(processed_img)
        sharpened_img = sharpener.enhance(2.0)

        return sharpened_img

def draw_bounding_boxes(image_content, texts):
    """Draw bounding boxes on the image for the detected text."""
    image_stream = io.BytesIO(image_content)
    with Image.open(image_stream) as img:
        draw = ImageDraw.Draw(img)

        # Draw bounding boxes around detected text
        for text in texts[1:]:  # Skip the first element which is the entire text block
            vertices = text.bounding_poly.vertices
            box = [(vertex.x, vertex.y) for vertex in vertices]
            draw.line(box + [box[0]], width=2, fill="red")

    # Return the modified image with bounding boxes
    output_image_stream = io.BytesIO()
    img.save(output_image_stream, format='PNG')
    output_image_stream.seek(0)

    return output_image_stream

def upload_image(image_stream, title):
    _, storage_client = initialize_clients()

    bucket = storage_client.bucket(BUCKET_NAME)

    blob_name = f'{title}__boxed.png'
    new_blob = bucket.blob(blob_name)

    new_blob.upload_from_file(image_stream, content_type='image/png')
    logging.info("Saved image with bounding boxes to %s in bucket %s", blob_name, BUCKET_NAME)

    # sets the pointer to the beginning of the stream
    image_stream.seek(0)

