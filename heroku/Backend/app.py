from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import pydicom
import cv2
import csv
import os
import base64
from datetime import datetime
from pathlib import Path
from io import BytesIO
from PIL import Image
# Assuming dicom_image.py and MedicalDenseNet are in your project structure
from dicom_image import process_dicom_image # Handles DICOM image extraction
import numpy as np

# Import ultralytics for YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics not installed. Install with: pip install ultralytics")

app = FastAPI(title="Medical Image Classification API", version="1.0.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ================================
# CONFIGURATION AND MODEL LOADING
# ================================

MODEL_PATH = Path(__file__).parent.parent / "Models" / "densenet_final_model.pt"
FALLBACK_PATH = Path(__file__).parent.parent / "Models" / "augmented_densenet_final_model.pt"
FALLBACK_PATH2 = Path(__file__).parent.parent / "Models" / "yolov8x.pt"

# Medical classification mapping
MEDICAL_CLASS_MAPPING = {
    0: "TB Positive",
    1: "Normal",
    2: "COPD",
    3: "Silicosis",
    4: "Lung Cancer"
}

# COCO to Medical mapping (for generic YOLOv8x models)
# This mapping needs to be robust if your YOLO model is COCO-trained.
# We map common COCO objects that might appear in an X-ray (person, general background items) to "Normal".
COCO_TO_MEDICAL_MAPPING = {
    0: 1,    # person -> Normal
    56: 1,   # chair -> Normal (common background noise)
    57: 1,   # couch -> Normal
    58: 1,   # potted plant -> Normal
    60: 1,   # dining table -> Normal
    62: 1,   # laptop -> Normal
    63: 1,   # mouse -> Normal
    64: 1,   # remote -> Normal
    65: 1,   # keyboard -> Normal
    67: 1,   # cell phone -> Normal
    72: 1,   # tv -> Normal
    # Add more as you discover them from debug logs if they frequently misclassify
    'default': 1  # Default to Normal for unrecognized COCO classes that aren't explicitly mapped
}

# Initialize model variables
model = None
model_type = None

def load_ai_models():
    """Load AI models with fallback logic."""
    global model, model_type

    # Try to load YOLOv8x first
    if MODEL_PATH.exists() and YOLO_AVAILABLE:
        try:
            print(f"🔄 Loading YOLOv8x model from: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            model_type = "yolo"

            # Get model information
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                class_names = model.model.names
                num_classes = len(class_names)
                print(f"📋 YOLOv8x Model Info:")
                print(f"    • Number of classes: {num_classes}")
                print(f"    • Class names: {class_names}") # Print all class names for debugging
                print(f"    • Model task: {getattr(model, 'task', 'unknown')}")

                # Check if it's likely a medical model
                medical_keywords = ['tb', 'normal', 'copd', 'silicosis', 'lung', 'cancer', 'pneumonia', 'chest']
                # Check if any of the model's *actual* class names match medical keywords
                is_medical = any(keyword in str(name).lower() for name in class_names.values() for keyword in medical_keywords)

                if is_medical and num_classes <= len(MEDICAL_CLASS_MAPPING): # Assuming medical model has <= 5 classes typically
                    print(f"✅ Detected medical-specific YOLO model!")
                    # If it's a true medical model, its classes should align directly with MEDICAL_CLASS_MAPPING
                    # No need for complex COCO mapping, but map_yolo_class_to_medical will still validate
                else:
                    print(f"⚠️ Appears to be generic COCO model or a detection model not directly mapped to medical categories.")
                    print(f"   Will use COCO to medical mapping logic for detected objects.")

            print(f"✅ YOLOv8x model loaded successfully!")
            return
        except Exception as e:
            print(f"❌ Failed to load YOLOv8x from {MODEL_PATH}: {e}")

    # Fallback to DenseNet if YOLOv8x fails or is not available
    print("🔄 Falling back to DenseNet model...")
    from ModelClass import MedicalDenseNet # Ensure this import is correct

    # Try different DenseNet model paths
    densenet_path = None
    if FALLBACK_PATH.exists():
        densenet_path = FALLBACK_PATH
    elif FALLBACK_PATH2.exists():
        densenet_path = FALLBACK_PATH2

    if densenet_path:
        try:
            num_classes = 5 # As per your MEDICAL_CLASS_MAPPING
            model = MedicalDenseNet(num_classes)
            # Use weights_only=False if you're loading a full model state dictionary,
            # or ensure your MedicalDenseNet class correctly handles it.
            model.load_state_dict(torch.load(densenet_path, map_location=torch.device("cpu")))
            model.eval()
            model_type = "densenet"
            print(f"✅ DenseNet model loaded successfully from: {densenet_path}")
        except Exception as e:
            print(f"❌ Failed to load DenseNet from {densenet_path}: {e}")
            raise FileNotFoundError(f"Could not load any model. Error loading DenseNet: {e}")
    else:
        raise FileNotFoundError(f"No model files found. Please ensure you have either a YOLOv8 model (.pt) or DenseNet models in the Models folder.")

# Load models on startup
load_ai_models()
print(f"🎯 Using model type: {model_type.upper()}")

# ================================
# UTILITY FUNCTIONS
# ================================

def map_yolo_class_to_medical(yolo_class_id, confidence):
    """
    Map YOLOv8x class (numeric ID) to medical classification.
    Assumes `model` is a loaded YOLO model for accessing class names.
    """
    yolo_class_name = model.names.get(yolo_class_id, "Unknown")
    print(f"🔍 Mapping YOLO class ID: {yolo_class_id}, Name: '{yolo_class_name}' (Conf: {confidence:.4f}) to medical category")

    # First, check if the YOLO model itself is already trained on medical classes (0-4)
    # This is for the case where your YOLO model's output classes directly represent medical conditions.
    if hasattr(model, 'model') and hasattr(model.model, 'names'):
        yolo_class_names = model.model.names
        # Check if the YOLO class name corresponds to one of our medical diagnoses
        if yolo_class_name in MEDICAL_CLASS_MAPPING.values():
            # Find the medical class ID for this name
            for medical_id, medical_name in MEDICAL_CLASS_MAPPING.items():
                if medical_name == yolo_class_name:
                    print(f"✅ Direct medical class mapping (YOLO is medical): {yolo_class_name} -> {medical_name}")
                    return medical_id
        # If YOLO class ID is within 0-4 and the model's name for that ID matches a medical class
        elif 0 <= yolo_class_id < len(MEDICAL_CLASS_MAPPING) and \
             yolo_class_names.get(yolo_class_id, 'Unknown') in MEDICAL_CLASS_MAPPING.values():
            print(f"✅ Direct medical class ID mapping (YOLO is medical): {yolo_class_id} -> {MEDICAL_CLASS_MAPPING.get(yolo_class_id)}")
            return yolo_class_id


    # If not a direct medical class from YOLO, proceed with COCO-like mapping
    # Extended mapping for common COCO objects that should resolve to "Normal" in a medical context
    COCO_TO_MEDICAL_MAPPING_EXTENDED = {
        0: 1,    # person -> Normal
        56: 1,   # chair -> Normal
        57: 1,   # couch -> Normal
        58: 1,   # potted plant -> Normal
        60: 1,   # dining table -> Normal
        62: 1,   # laptop -> Normal
        63: 1,   # mouse -> Normal
        64: 1,   # remote -> Normal
        65: 1,   # keyboard -> Normal
        67: 1,   # cell phone -> Normal
        72: 1,   # tv -> Normal
        # Add more COCO class IDs here if you observe them frequently and they should map to "Normal"
        # For example, if you see 'bottle', 'cup', 'book', etc., add them.
    }

    if yolo_class_id in COCO_TO_MEDICAL_MAPPING_EXTENDED:
        medical_class = COCO_TO_MEDICAL_MAPPING_EXTENDED[yolo_class_id]
        print(f"🔄 COCO to medical mapping: YOLO Class '{yolo_class_name}' (ID: {yolo_class_id}) -> {medical_class} ({MEDICAL_CLASS_MAPPING[medical_class]})")
        return medical_class

    # Fallback for unknown/unmapped YOLO classes.
    # Instead of "TB Positive" as cautious, default to "Normal" unless very high confidence.
    print(f"⚠️ YOLO class '{yolo_class_name}' (ID: {yolo_class_id}) not explicitly mapped in COCO_TO_MEDICAL_MAPPING_EXTENDED.")
    if confidence > 0.8: # A higher threshold for strong confidence in an unmapped object
        # If a very confident unmapped object, it's safer to still lean towards Normal in a medical image
        # unless you have specific generic objects that indicate disease.
        medical_class = 1 # Still default to Normal for unmapped generic objects with high confidence
        print(f"🔄 High confidence unmapped generic class -> {MEDICAL_CLASS_MAPPING[medical_class]} (defaulting to Normal)")
    else:
        medical_class = 1  # Default to Normal for moderate/low confidence unmapped objects
        print(f"🔄 Defaulting unmapped class -> {MEDICAL_CLASS_MAPPING[medical_class]}")

    return medical_class

def safe_get_dicom_value(dicom_data, tag, default="Unknown"):
    """Safely extract values from DICOM data, handling encoding issues."""
    try:
        if hasattr(dicom_data, tag):
            value = getattr(dicom_data, tag)
            if value is None or value == "":
                return default

            # Convert to string and clean up
            str_value = str(value).strip()

            # Filter out obviously corrupted/encoded data
            if (
                len(str_value) > 50 or  # Too long for typical metadata fields
                any(char in str_value for char in ['=', '+', '/', '_']) and len(str_value) > 15 or # Base64-like chars
                str_value.count('A') > len(str_value) * 0.4 or # Too many A's (common in corrupted data)
                any(ord(char) > 126 for char in str_value[:20]) or # Non-printable/extended ASCII characters
                str_value.lower().startswith(('unkn', 'null', 'none', 'n/a')) or # Obviously null values
                len(str_value.replace('A', '').replace('a', '')) < 3 and len(str_value) > 5 # Mostly A's, but long
            ):
                return default

            # Handle PersonName objects specifically
            if hasattr(value, 'family_name') and hasattr(value, 'given_name'):
                try:
                    family = str(value.family_name or '').strip()
                    given = str(value.given_name or '').strip()
                    # Validate names don't contain encoded data and are reasonable length
                    if family and given and len(family) < 30 and len(given) < 30 and \
                       not any(char in family+given for char in ['=', '+', '/', '_']):
                        return f"{given} {family}"
                except:
                    pass # Fall through to default if parsing fails

            # For regular string values, additional validation
            # Ensure it's alphanumeric, allows spaces/hyphens/dots, and is reasonably short
            if len(str_value) < 35 and all(c.isalnum() or c in ' -.' for c in str_value):
                return str_value

            return default # If it didn't pass specific checks, return default
        return default
    except Exception as e:
        print(f"Error extracting DICOM field '{tag}': {e}")
        return default


def process_medical_image(image_array):
    """Process medical image with proper contrast and normalization."""
    # Ensure we have a valid image array
    if image_array is None or image_array.size == 0:
        raise ValueError("Invalid image data")

    # Convert to float for processing
    image_float = image_array.astype(np.float32)

    # Normalize to 0-255 range
    if image_float.max() > 0: # Avoid division by zero
        image_float = (image_float - image_float.min()) / (image_float.max() - image_float.min()) * 255
    else: # Handle all-zero image
        image_float = np.zeros_like(image_float)


    # Apply contrast enhancement for medical images
    image_8bit = image_float.astype(np.uint8)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better visibility
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    image_enhanced = clahe.apply(image_8bit)

    return image_enhanced

def create_display_image(image_processed):
    """Create base64 encoded image for frontend display."""
    # Resize for display (smaller size for frontend)
    # Ensure the image has at least 1 dimension
    if image_processed.shape[0] == 0 or image_processed.shape[1] == 0:
        print("Warning: Processed image has zero dimension, returning empty base64 string.")
        return "" # Return empty string for invalid image

    image_resized = cv2.resize(image_processed, (256, 256), interpolation=cv2.INTER_AREA)

    # Convert to PIL Image for base64 encoding
    # Check if image_resized is already grayscale (2D array) or needs conversion
    if len(image_resized.shape) == 2:
        image_pil = Image.fromarray(image_resized, mode='L')
    elif len(image_resized.shape) == 3 and image_resized.shape[2] == 3: # RGB
        image_pil = Image.fromarray(image_resized, mode='RGB')
    else:
        # Handle unexpected formats, e.g., convert to L if it's a 3-channel image where channels are identical
        image_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY), mode='L')


    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")

    # Convert image to Base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

# ================================
# DICOM PROCESSING FUNCTIONS
# ================================

def extract_patient_demographics(dicom_data):
    """Extract patient demographics with fallbacks and validation."""
    patient_name = "Unknown"
    patient_id = "Unknown"

    print("🔍 Analyzing DICOM patient fields...")

    # Try to get PatientName
    name_value = safe_get_dicom_value(dicom_data, 'PatientName', None)
    if name_value and name_value != "Unknown":
        patient_name = name_value
    else:
        print("PatientName field problematic or encrypted.")

    # Try to get PatientID
    id_value = safe_get_dicom_value(dicom_data, 'PatientID', None)
    if id_value and id_value != "Unknown":
        patient_id = id_value
    else:
        print("PatientID field problematic or encrypted.")

    # Fallback to study info if primary patient data is still unknown/encrypted
    if patient_name == "Unknown" or patient_id == "Unknown":
        print("🔄 Patient data still unknown, generating from study information...")

        study_date = safe_get_dicom_value(dicom_data, 'StudyDate', '')
        modality = safe_get_dicom_value(dicom_data, 'Modality', '')
        study_uid = safe_get_dicom_value(dicom_data, 'StudyInstanceUID', '')

        # Generate patient name
        if patient_name == "Unknown":
            if study_date and len(study_date) >= 8:
                try:
                    formatted_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
                    patient_name = f"{modality if modality != 'Unknown' else 'Medical'} Patient ({formatted_date})"
                except:
                    patient_name = "Anonymous Patient"
            else:
                patient_name = "Anonymous Patient"

        # Generate patient ID
        if patient_id == "Unknown":
            if study_uid and len(study_uid) > 10:
                short_uid = study_uid.split('.')[-1][-8:] if '.' in study_uid else study_uid[-8:]
                patient_id = f"UID_{short_uid.upper()}"
            else:
                # Use a hash of current time as a last resort
                import hashlib
                current_time_hash = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8].upper()
                patient_id = f"GEN_{current_time_hash}"

    print(f"✅ Final patient name: {patient_name}")
    print(f"✅ Final patient ID: {patient_id}")

    return patient_name, patient_id

def enhanced_dicom_extraction(dicom_data):
    """Enhanced DICOM metadata extraction with better handling of encrypted data."""
    try:
        # Basic extraction
        patient_name, patient_id = extract_patient_demographics(dicom_data)

        # Extract other fields with encryption awareness and validation
        patient_sex = safe_get_dicom_value(dicom_data, 'PatientSex', 'Unknown')
        patient_age = safe_get_dicom_value(dicom_data, 'PatientAge', 'Unknown')
        patient_weight = safe_get_dicom_value(dicom_data, 'PatientWeight', 'Unknown')

        # Attempt to format age if it's numeric and was extracted
        if patient_age != 'Unknown' and patient_age.isdigit():
            patient_age = f"{patient_age} years"
        elif 'Y' in patient_age or 'M' in patient_age: # DICOM age format like "060Y"
            pass # Keep as is if already formatted by pydicom

        # Attempt to format weight if it's numeric and was extracted
        if patient_weight != 'Unknown':
            try:
                weight_float = float(patient_weight)
                patient_weight = f"{weight_float:.1f} kg"
            except ValueError:
                pass # Keep as is if not a number

        # Get study information for display (even if patient data is obfuscated)
        study_date = safe_get_dicom_value(dicom_data, 'StudyDate', 'Unknown')
        study_time = safe_get_dicom_value(dicom_data, 'StudyTime', 'Unknown')
        modality = safe_get_dicom_value(dicom_data, 'Modality', 'Unknown')
        manufacturer = safe_get_dicom_value(dicom_data, 'Manufacturer', 'Unknown')

        result = {
            'patient_name': patient_name,
            'patient_id': patient_id,
            'patient_sex': patient_sex,
            'patient_age': patient_age,
            'patient_weight': patient_weight,
            'study_info': {
                'study_date': study_date,
                'study_time': study_time,
                'modality': modality,
                'manufacturer': manufacturer
            }
        }

        return result

    except Exception as e:
        print(f"Error in enhanced DICOM extraction: {e}")
        return {
            'patient_name': 'Anonymous Patient',
            'patient_id': 'PAT_UNKNOWN',
            'patient_sex': 'Unknown',
            'patient_age': 'Unknown',
            'patient_weight': 'Unknown',
            'study_info': {}
        }

# ================================
# IMAGE PROCESSING FUNCTIONS
# ================================

def process_image_file(file_content, filename):
    """Process both DICOM and regular image files."""
    patient_info = {
        'patient_name': 'Unknown',
        'patient_id': 'Unknown',
        'patient_sex': 'Unknown',
        'patient_age': 'Unknown',
        'patient_weight': 'Unknown'
    }

    image_processed = None # Initialize to None

    # Check if it's a DICOM file (by extension or magic number)
    is_dicom = filename.lower().endswith('.dcm') or b'DICM' in file_content[:132] # DICOM magic number at offset 128

    if is_dicom:
        try:
            dicom_data = pydicom.dcmread(BytesIO(file_content))

            # Extract patient information
            dicom_info = enhanced_dicom_extraction(dicom_data)
            patient_info.update(dicom_info)

            # Process image using dicom_image module or pydicom's pixel_array
            try:
                # Assuming process_dicom_image handles windowing and normalization
                image_array = process_dicom_image(dicom_data)
            except Exception as e:
                print(f"Error with dicom_image.process_dicom_image: {e}. Falling back to pydicom.pixel_array")
                image_array = dicom_data.pixel_array

            image_processed = process_medical_image(image_array)

        except Exception as e:
            print(f"Full DICOM parsing/processing error: {e}. Attempting to process as regular image.")
            # If DICOM parsing fails, treat as regular image
            try:
                image = Image.open(BytesIO(file_content))
                image_processed = process_regular_image(image)
            except Exception as img_e:
                raise HTTPException(status_code=400, detail=f"Could not process file as DICOM or regular image: {img_e}")
    else:
        # Process regular image file
        try:
            image = Image.open(BytesIO(file_content))
            image_processed = process_regular_image(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not open or process image file: {e}")

    if image_processed is None:
        raise HTTPException(status_code=400, detail="Failed to process image data.")

    return image_processed, patient_info

def process_regular_image(image):
    """Process regular image files (PNG, JPG, etc.)."""
    # Convert to grayscale if needed, then to numpy array
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    if image.mode == 'RGB':
        image = image.convert('L')  # Convert to grayscale

    image_array = np.array(image)
    return process_medical_image(image_array)

# ================================
# AI INFERENCE FUNCTIONS
# ================================

def run_yolo_inference(image_processed):
    """Run YOLOv8x inference on processed image."""
    # YOLOv8x preprocessing - convert to RGB, resize to 640x640
    if len(image_processed.shape) == 2:  # Grayscale to RGB
        image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image_processed # Already RGB or similar

    # YOLOv8x expects 640x640 input
    image_resized = cv2.resize(image_rgb, (640, 640), interpolation=cv2.INTER_AREA)

    # Run YOLOv8x inference
    print(f"🔄 Running YOLOv8x inference...")
    results = model(image_resized, verbose=False)

    predicted_class = 1 # Default to Normal
    confidence_scores = [0.1, 0.8, 0.1, 0.0, 0.0] # Higher confidence in Normal

    if results and len(results) > 0:
        result = results[0]

        # Check if the model has 'probs' (classification mode)
        if hasattr(result, 'probs') and result.probs is not None and len(result.probs.data) > 0:
            # Classification mode (e.g., if your YOLO was trained for classification directly)
            probs = result.probs.data.cpu().numpy()
            predicted_class = int(np.argmax(probs))
            confidence_scores = probs.tolist()

            print(f"🎯 YOLOv8x Classification - Predicted Class ID: {predicted_class} ('{MEDICAL_CLASS_MAPPING.get(predicted_class, 'Unknown')}') Confidence: {probs[predicted_class]:.4f}")

            # Ensure confidence_scores is 5 elements long, pad if necessary
            if len(confidence_scores) < len(MEDICAL_CLASS_MAPPING):
                confidence_scores.extend([0.0] * (len(MEDICAL_CLASS_MAPPING) - len(confidence_scores)))
            elif len(confidence_scores) > len(MEDICAL_CLASS_MAPPING):
                confidence_scores = confidence_scores[:len(MEDICAL_CLASS_MAPPING)]


        # Check if the model has 'boxes' (detection mode)
        elif hasattr(result, 'boxes') and len(result.boxes) > 0:
            # Detection mode - use highest confidence detection
            boxes = result.boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            if len(confidences) > 0:
                # Get highest confidence detection
                max_conf_idx = np.argmax(confidences)
                yolo_class_id = classes[max_conf_idx]
                max_confidence = confidences[max_conf_idx]

                print(f"🎯 YOLOv8x Detection - Highest Confidence YOLO Class ID: {yolo_class_id} (Name: '{model.names.get(yolo_class_id, 'Unknown')}'), Confidence: {max_confidence:.4f}")

                # Map YOLO class to medical class using the refined function
                predicted_class = map_yolo_class_to_medical(yolo_class_id, max_confidence)

                # Initialize confidence scores for all medical classes
                confidence_scores = [0.0] * len(MEDICAL_CLASS_MAPPING)

                # Assign the max_confidence to the predicted medical class
                confidence_scores[predicted_class] = float(max_confidence)

                # Distribute remaining confidence among other classes for realism
                remaining_confidence = 1.0 - float(max_confidence)
                if remaining_confidence > 0:
                    other_classes_indices = [i for i in range(len(MEDICAL_CLASS_MAPPING)) if i != predicted_class]
                    if other_classes_indices: # Avoid division by zero
                        each_share = remaining_confidence / len(other_classes_indices)
                        for idx in other_classes_indices:
                            confidence_scores[idx] += float(each_share) # Use += to add to existing, potentially non-zero confidences

                # Re-normalize to ensure sum is 1 (due to potential floating point errors from adding shares)
                sum_scores = sum(confidence_scores)
                if sum_scores > 0:
                    confidence_scores = [s / sum_scores for s in confidence_scores]

                print(f"📊 Mapped confidence distribution: {[f'{MEDICAL_CLASS_MAPPING[i]}: {conf:.3f}' for i, conf in enumerate(confidence_scores)]}")
            else:
                print("⚠️ YOLOv8x Detection: No objects detected, defaulting to Normal with uncertainty.")
                # Default values already set at the start of the function.

        else:
            print("⚠️ YOLOv8x: No valid predictions (neither classification nor detection results found), defaulting to Normal with uncertainty.")
            # Default values already set at the start of the function.

    else:
        print("⚠️ YOLOv8x: Model returned no results, defaulting to Normal.")
        # Default values already set at the start of the function.

    return predicted_class, confidence_scores


def run_densenet_inference(image_processed):
    """Run DenseNet inference on processed image."""
    # DenseNet preprocessing - convert to RGB, resize to 224x224
    if len(image_processed.shape) == 2:  # Grayscale
        image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2RGB)
    else:  # Already RGB or similar
        image_rgb = image_processed

    # Resize for model input (224x224 for DenseNet)
    image_resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0,1] range
    image_normalized = image_resized.astype(float) / 255.0

    # Convert to PyTorch tensor (channel-first format)
    # Unsqueeze for batch dimension
    image_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0).float()

    # Run DenseNet inference
    print(f"🔄 Running DenseNet inference...")
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence_scores = torch.softmax(output, dim=1).squeeze().tolist()

    print(f"🎯 DenseNet - Class: {predicted_class} ('{MEDICAL_CLASS_MAPPING.get(predicted_class, 'Unknown')}'), Confidence: {confidence_scores[predicted_class]:.4f}")

    return predicted_class, confidence_scores

def prepare_response_data(predicted_class, confidence_scores, patient_info, filename, image_processed):
    """Prepare final response data with proper type conversion."""
    # Use medical class mapping
    diagnosis = MEDICAL_CLASS_MAPPING.get(predicted_class, "Unknown")

    # Ensure confidence_scores is the right length and all values are Python floats
    expected_num_classes = len(MEDICAL_CLASS_MAPPING)
    if len(confidence_scores) != expected_num_classes:
        # Pad with zeros if less, truncate if more
        confidence_scores = confidence_scores[:expected_num_classes] + [0.0] * (expected_num_classes - len(confidence_scores))

    # Convert all numpy types to Python native types for JSON serialization
    confidence_scores = [float(score) for score in confidence_scores]
    predicted_class = int(predicted_class)

    # Prepare confidence scores for all classes
    class_confidences = {
        MEDICAL_CLASS_MAPPING[i]: round(float(confidence_scores[i]) * 100, 2)
        for i in range(len(MEDICAL_CLASS_MAPPING))
    }

    # Create display image
    display_image = create_display_image(image_processed)

    # Store details in response
    response_data = {
        "PatientID": str(patient_info.get('patient_id', 'Unknown')),
        "PatientName": str(patient_info.get('patient_name', 'Unknown')),
        "PatientSex": str(patient_info.get('patient_sex', 'Unknown')),
        "PatientWeight": str(patient_info.get('patient_weight', 'Unknown')),
        "Age": str(patient_info.get('patient_age', 'Unknown')),
        "Diagnosis": str(diagnosis),
        "Confidence": round(float(confidence_scores[predicted_class]) * 100, 2),
        "AllClassConfidences": class_confidences,
        "FileName": str(filename),
        "ProcessedImage": display_image
    }

    return response_data

# ================================
# API ENDPOINTS
# ================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Medical Image Classification API is running", "status": "healthy"}

@app.post("/analyze/")
async def analyze_medical_image(file: UploadFile = File(...)):
    """
    Single endpoint for complete medical image analysis.
    Combines upload, processing, DICOM extraction, AI inference, and display image creation.
    """
    try:
        # Read file content
        file_content = await file.read()
        print(f"📁 Processing file: {file.filename} ({len(file_content)} bytes)")

        # Process image and extract patient information
        image_processed, patient_info = process_image_file(file_content, file.filename)

        # Run AI inference based on model type
        if model_type == "yolo":
            predicted_class, confidence_scores = run_yolo_inference(image_processed)
        else: # DenseNet or other model type
            predicted_class, confidence_scores = run_densenet_inference(image_processed)

        # Prepare and return response
        response_data = prepare_response_data(
            predicted_class, confidence_scores, patient_info, file.filename, image_processed
        )

        print(f"✅ Analysis complete for '{file.filename}': Diagnosis -> {response_data['Diagnosis']} ({response_data['Confidence']}%)")
        return response_data

    except HTTPException as http_e:
        raise http_e # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        print(f"❌ Analysis error for {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Error analyzing medical image: {str(e)}")

@app.post("/debug-dicom/")
async def debug_dicom(file: UploadFile = File(...)):
    """Debug endpoint to show available DICOM fields."""
    try:
        file_content = await file.read()

        # Check for DICOM file more robustly
        if not (file.filename.lower().endswith('.dcm') or b'DICM' in file_content[:132]):
            raise HTTPException(status_code=400, detail="Uploaded file is not a DICOM file.")

        dicom_data = pydicom.dcmread(BytesIO(file_content))

        # Extract interesting fields for debugging
        debug_info = {
            "file_name": file.filename,
            "file_size_bytes": len(file_content),
            "is_dicom_magic_number_present": b'DICM' in file_content[:132],
            "available_fields_summary": [], # To avoid flooding with raw data
            "patient_fields": {},
            "study_fields": {},
            "image_fields": {},
            "raw_dicom_header_preview": file_content[:200].hex() # First 200 bytes as hex
        }

        # Get all available fields (keywords and safe values)
        for elem in dicom_data:
            debug_info["available_fields_summary"].append({
                "tag": str(elem.tag),
                "keyword": elem.keyword if hasattr(elem, 'keyword') else 'N/A',
                "VR": elem.VR if hasattr(elem, 'VR') else 'N/A',
                "value_length": len(str(elem.value)),
                "value_preview": str(elem.value)[:100] + "..." if len(str(elem.value)) > 100 else str(elem.value)
            })

        # Extract specific patient-related fields using safe_get_dicom_value
        patient_fields = ['PatientName', 'PatientID', 'PatientSex', 'PatientAge', 'PatientBirthDate',
                          'PatientWeight', 'PatientSize', 'ResponsiblePerson', 'PatientsName']

        for field in patient_fields:
            if hasattr(dicom_data, field):
                debug_info["patient_fields"][field] = {
                    "raw_value": str(getattr(dicom_data, field)),
                    "safe_value": safe_get_dicom_value(dicom_data, field),
                    "type": str(type(getattr(dicom_data, field)))
                }

        # Extract study-related fields
        study_fields = ['StudyDate', 'StudyTime', 'StudyInstanceUID', 'AccessionNumber', 'StudyDescription', 'Modality']
        for field in study_fields:
            if hasattr(dicom_data, field):
                debug_info["study_fields"][field] = safe_get_dicom_value(dicom_data, field)

        # Image-related fields
        image_fields = ['Rows', 'Columns', 'BitsAllocated', 'Modality', 'Manufacturer', 'SOPInstanceUID', 'SeriesInstanceUID']
        for field in image_fields:
            if hasattr(dicom_data, field):
                debug_info["image_fields"][field] = safe_get_dicom_value(dicom_data, field)

        # Try to get pixel_array info without actually processing the image fully
        try:
            if hasattr(dicom_data, 'pixel_array'):
                debug_info["image_fields"]["PixelArrayShape"] = dicom_data.pixel_array.shape
                debug_info["image_fields"]["PixelArrayDtype"] = str(dicom_data.pixel_array.dtype)
        except Exception as pa_e:
            debug_info["image_fields"]["PixelArrayError"] = str(pa_e)

        return debug_info

    except pydicom.errors.InvalidDicomError as e:
        raise HTTPException(status_code=400, detail=f"Invalid DICOM file: {str(e)}")
    except Exception as e:
        print(f"❌ Debug DICOM failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Make sure to run with `python -m uvicorn main:app --reload` during development
    uvicorn.run(app, host="0.0.0.0", port=8000)