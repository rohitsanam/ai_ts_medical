import pydicom
import base64
import cv2
from io import BytesIO
from PIL import Image

def process_dicom_image(file):
    """Extracts image from a DICOM file and returns metadata + Base64-encoded image."""
    dicom_data = pydicom.dcmread(file)
    
    # Extract image data and resize
    image_array = dicom_data.pixel_array
    image_resized = cv2.resize(image_array, (512, 512), interpolation=cv2.INTER_AREA)

    # Convert to PIL Image
    image_pil = Image.fromarray(image_resized)
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")

    # Convert image to Base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "patient_id": dicom_data.PatientID,
        "diagnosis": "TB Positive",
        "image": image_base64
    }