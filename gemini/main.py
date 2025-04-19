import cv2
import os
import google.generativeai as genai
import base64
from PIL import Image
import io
import mimetypes
import argparse
import tqdm  # For progress bar
from dotenv import load_dotenv # Import dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Configure API key from environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
    print("Please create a .env file with GOOGLE_API_KEY='YOUR_API_KEY_HERE' or set the environment variable.")
    exit(1)
genai.configure(api_key=API_KEY)

# --- Constants ---
FRAME_EXTRACTION_INTERVAL_SECONDS = 1  # Extract one frame per second
MAX_FRAMES_TO_PROCESS = 100  # Limit the number of frames sent to the API
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- Helper Functions ---

def get_video_duration(video_path):
    """Gets the duration of the video in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

def extract_frames(video_path, interval_seconds=1):
    """Extracts frames from a video file at a specified interval."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Could not determine video FPS.")
        cap.release()
        return None, None

    frame_interval = int(fps * interval_seconds)
    frames = []
    frame_count = 0
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting frames from '{os.path.basename(video_path)}' (approx. {total_frames_in_video // fps:.1f}s)...")

    pbar = tqdm.tqdm(total=total_frames_in_video, unit="frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert frame to PIL Image (RGB format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Encode image to JPEG bytes
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            frames.append(buffer.getvalue())

            if len(frames) >= MAX_FRAMES_TO_PROCESS:
                print(f"\nReached maximum frame limit ({MAX_FRAMES_TO_PROCESS}). Processing these frames.")
                pbar.update(total_frames_in_video - pbar.n) # Fill progress bar
                break

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames, "video/mp4" # Assuming mp4, adjust if needed or detect dynamically

def frames_to_parts(frames, mime_type="image/jpeg"):
    """Converts list of frame bytes to Gemini API Part format."""
    return [
        {"mime_type": mime_type, "data": frame}
        for frame in frames
    ]

def get_reward_for_prompt(video_path, eval_prompt):
    """Analyzes video frames against a prompt and returns a numerical reward (+1, -1, or 0)."""

    frames, video_mime_type = extract_frames(video_path, FRAME_EXTRACTION_INTERVAL_SECONDS)
    if not frames:
        print("Error: Could not extract frames from the video.")
        return 0.0 # Return neutral reward on frame extraction error

    image_parts = frames_to_parts(frames)

    # Construct the prompt for Gemini, asking for a simple POSITIVE/NEGATIVE response
    gemini_prompt = (
        f"Analyze the following video frames based *only* on the visual content. "
        f"Answer the question: \"{eval_prompt}\". "
        f"Respond with only the single word 'POSITIVE' if the condition is met, or 'NEGATIVE' if it is not met. "
        f"Do not provide any explanation or other text."
    )

    request_content = [gemini_prompt] + image_parts

    model = genai.GenerativeModel('gemini-1.5-flash')

    print(f"\nSending {len(image_parts)} frames to Gemini API for reward evaluation...")
    print(f"Evaluation Prompt: {eval_prompt}")
    try:
        response = model.generate_content(request_content, stream=False, safety_settings=SAFETY_SETTINGS)
        response.resolve()
        response_text = response.text.strip().upper()
        print(f"Received response from Gemini: '{response_text}'")

        if "POSITIVE" in response_text:
            return 1.0
        elif "NEGATIVE" in response_text:
            return -1.0
        else:
            print("Warning: Gemini response was not clearly POSITIVE or NEGATIVE. Returning neutral reward (0.0).")
            # Optionally inspect response.prompt_feedback here
            # print(f"Prompt Feedback: {response.prompt_feedback}")
            return 0.0

    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        # Optionally inspect response.prompt_feedback if available on error
        # try:
        #     print(f"Prompt Feedback on error: {response.prompt_feedback}")
        # except AttributeError:
        #     pass # No feedback available
        return 0.0 # Return neutral reward on API error

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a video against a prompt using Gemini API and return a numerical reward.")
    parser.add_argument("video_file", help="Path to the video file.")
    # Removed the generic prompt, added a required eval_prompt
    parser.add_argument("--eval_prompt", required=True, help="The evaluation question/prompt for Gemini (e.g., 'Is the text properly displayed?').")
    parser.add_argument("-i", "--interval", type=int, default=FRAME_EXTRACTION_INTERVAL_SECONDS, help="Interval in seconds between frame extractions.")
    parser.add_argument("-m", "--max_frames", type=int, default=MAX_FRAMES_TO_PROCESS, help="Maximum number of frames to send to the API.")

    args = parser.parse_args()

    # Update globals from args
    FRAME_EXTRACTION_INTERVAL_SECONDS = args.interval
    MAX_FRAMES_TO_PROCESS = args.max_frames

    # Call the reward function
    reward = get_reward_for_prompt(args.video_file, args.eval_prompt)

    print("\n--- Reward --- ")
    print(f"Numerical Reward: {reward}")
    print("--------------")
