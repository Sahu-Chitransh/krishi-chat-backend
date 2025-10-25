import google.generativeai as genai
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Load API Key & Model Setup ---
load_dotenv()

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("FATAL: API Key not found. Check .env file.")
    exit()

# (CHANGED) Updated the system prompt to handle location
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

**Your Guiding Rules (Most Important):**
* **Simple Language:** ALWAYS use simple, everyday language.
* **Be Patient and Respectful:** Always be polite. Start conversations with "Namaste!"
* **Handle Out-of-Scope Questions:** You do not have access to live data.
    * If asked for **today's weather** or **current market prices**, you must politely explain: "Namaste. I am an AI assistant and I don't have a connection to live weather or market prices. However, based on your location (if provided), I can give you general advice about crops for your season."
"""

try:
    model = genai.GenerativeModel(
        model_name="models/gemini-flash-latest",
        system_instruction=krishi_mitra_prompt
    )
    chat = model.start_chat(history=[])
    print("Model and chat initialized successfully.")

except Exception as e:
    print(f"FATAL: Error initializing model: {e}")
    exit()

# --- Flask Server Setup ---

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Namaste! Krishi Mitra API is running."


@app.route("/chat", methods=["POST"])
def chat_api():
    """
    This is the main API endpoint.
    It receives JSON with {"message": "..."} and optionally {"location": {"lat": ..., "lon": ...}}
    """
    try:
        # Get data from the request
        data = request.json
        user_message = data.get("message")
        location = data.get("location")  # (NEW) Get the location object

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # (CHANGED) Augment the user's message with location if it exists
        augmented_message = user_message

        if location and location.get("lat") and location.get("lon"):
            # This is the new prompt we send to the AI, combining the user's
            # question with the location context.
            augmented_message = f"""
            User's question: "{user_message}"

            (My current location context: latitude {location['lat']}, longitude {location['lon']})
            """
            print(f"Augmenting prompt with location: {location}")

        # Send the (potentially augmented) message to the model
        response = chat.send_message(augmented_message)

        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Error during chat: {e}")
        return jsonify({"error": "An internal error occurred"}), 500


# --- Run the Server ---
if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)