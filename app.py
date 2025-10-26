import google.generativeai as genai
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# --- 1. Load API Key & Chat Model ---
load_dotenv()

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("FATAL: API Key not found. Check .env file.")
    exit()

krishi_mitra_prompt = """
You are Krishi Mitra (Farmer's Friend), a helpful, patient, and friendly AI assistant.
Your single most important goal is to provide clear, simple, and actionable advice to farmers based on general farming knowledge. You must be respectful and understanding of their needs.

**Your Core Capabilities:**
1.  **Crop Management:** Answer general questions about planting, watering, fertilizer, and harvesting.
2.  **Pests and Diseases:** Help identify common pests and diseases based on the farmer's description.

**NEW: Handling Location Data:**
* The user's message may include their physical location (latitude and longitude) as context.
* If this location is provided, you MUST use it to make your answer more specific.
* For example: "What crop should I plant?" with a location in Maharashtra, India, should get an answer specific to that region's soil and climate (e.g., "In your area, crops like cotton or sugarcane are common...").
* **Crucially: Do NOT mention the latitude and longitude numbers back to the user.** Just use the location to inform your answer.

**Your Guiding Rules (Most Important):
* **Simple Language:** ALWAYS use simple, everyday language.
* **Be Patient and Respectful:** Always be polite. Start conversations with "Namaste!"
* **Handle Out-of-Scope Questions:** You do not have access to live data.
    * If asked for **today's weather** or **current market prices**, you must politely explain: "Namaste. I am an AI assistant and I don't have a connection to live weather or market prices. However, based on your location (if provided), I can give you general advice about crops for your season."
"""

try:
    chat_model = genai.GenerativeModel(
        model_name="models/gemini-flash-latest",
        system_instruction=krishi_mitra_prompt
    )
    chat = chat_model.start_chat(history=[])
    print("Chat model and chat initialized successfully.")
except Exception as e:
    print(f"FATAL: Error initializing chat model: {e}")
    exit()

# --- 2. Load Disease Classification Model ---

MODEL_PATH = 'MobileNet.keras'

# (NEW) Inserted your 42 class names, sorted alphabetically
# I've also fixed the "thirps on  cotton" typo
CLASS_NAMES = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm',
    'Becterial Blight in Rice', 'Brownspot', 'Common_Rust', 'Cotton Aphid',
    'Flag Smut', 'Gray_Leaf_Spot', 'Healthy Maize', 'Healthy Wheat',
    'Healthy cotton', 'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane',
    'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast', 'Sugarcane Healthy',
    'Tungro', 'Wheat Brown leaf Rust', 'Wheat Stem fly', 'Wheat aphid',
    'Wheat black rust', 'Wheat leaf blight', 'Wheat mite',
    'Wheat powdery mildew', 'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt',
    'Yellow Rust Sugarcane', 'bacterial_blight in Cotton',
    'bollrot on Cotton', 'bollworm on Cotton', 'cotton mealy bug',
    'cotton whitefly', 'maize ear rot', 'maize fall armyworm',
    'maize stem borer', 'pink bollworm in cotton', 'red cotton bug',
    'thirps on cotton'  # <-- Corrected typo here
]

try:
    classify_model = tf.keras.models.load_model(MODEL_PATH)
    class_names = CLASS_NAMES
    print(f"Loaded classification model. Found {len(class_names)} classes.")
except Exception as e:
    print(f"FATAL: Could not load classification model: {e}")
    classify_model = None
    class_names = []


def preprocess_image_data(image_data):
    """Loads and preprocesses an in-memory image for the model."""
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((224, 224))
    img_array = np.array(img)

    if img_array.shape[2] == 4:  # Handle RGBA
        img_array = img_array[:, :, :3]

    img_batch = np.expand_dims(img_array, axis=0)

    # --- THIS IS THE FIX ---
    # We must scale the pixels to the 0-1 range, just like in training
    img_batch = img_batch / 255.0
    # ---------------------

    return img_batch


# --- 3. Flask Server Setup ---

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Namaste! Krishi Mitra API is running."


@app.route("/chat", methods=["POST"])
def chat_api():
    try:
        data = request.json
        user_message = data.get("message")
        location = data.get("location")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        augmented_message = user_message
        if location and location.get("lat") and location.get("lon"):
            augmented_message = f"""
            User's question: "{user_message}"
            (My current location context: latitude {location['lat']}, longitude {location['lon']})
            """

        response = chat.send_message(augmented_message)
        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Error during chat: {e}")
        return jsonify({"error": "An internal error occurred"}), 500


@app.route("/classify", methods=["POST"])
def classify_api():
    if not classify_model or len(class_names) == 0:
        return jsonify({"error": "Classification model not loaded or classes not configured"}), 500

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        image_data = file.read()

        # Use the FIXED preprocessing function
        processed_image = preprocess_image_data(image_data)

        predictions = classify_model.predict(processed_image)

        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = np.max(predictions[0]) * 100

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({"error": "An internal error occurred during classification"}), 500


# --- 5. Run the Server ---
if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)