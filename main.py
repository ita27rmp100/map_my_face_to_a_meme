import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import os
import random
import subprocess
import imageio.v3 as iio  

class MemeMatcher:
    
    EMOTION_KEYWORDS = {
        "smile": ["smile", "happy", "lol", "laugh", "grin", "joy", "fun"],
        "sad": ["sad", "cry", "tear", "upset", "depressed", "frown"],
        "angry": ["angry", "mad", "rage", "grumpy", "serious", "hate"],
        "surprise": ["wow", "shock", "surprise", "omg", "pog", "open", "amazed", "scary"],
        "neutral": ["neutral", "stare", "waiting", "bored"],
        "Tongue_Out": ["tongue", "bleh", "silly", "crazy", "mlem", "lick"] 
    }
    
    GESTURE_KEYWORDS = {
        "Thumb_Up": ["thumbs_up", "like", "good", "approve", "cool"],
        "Thumb_Down": ["thumbs_down", "dislike", "bad", "boo"],
        "Victory": ["peace", "victory", "cool", "vibes"],
        "Open_Palm": ["stop", "halt", "wait", "high_five"],
        "Pointing_Up": ["point", "look", "up", "idea", "nerd"],
        "Closed_Fist": ["fist", "punch", "strength", "power", "rock", "fight"],
        "ILoveYou": ["love", "heart", "rock_on", "metal", "spider"],
        "Raised_Hand": ["raised", "hand", "raise", "رفع", "يد", "wave", "hello", "hi"]
    }

    def __init__(self, assets_folder="assets"):
        self.assets_folder = assets_folder
        self.frame_count = 0
        
        self.hand_model_path = "gesture_recognizer.task"
        self.face_model_path = "face_landmarker.task"
        self._download_models()

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Initialize Hand Gesture Recognizer
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureOptions = mp.tasks.vision.GestureRecognizerOptions
        self.gesture_recognizer = GestureRecognizer.create_from_options(
            GestureOptions(
                base_options=BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2
            )
        )

        # Initialize Face Landmarker with Blendshapes (for tongue)
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.face_landmarker = FaceLandmarker.create_from_options(
            FaceOptions(
                base_options=BaseOptions(model_asset_path=self.face_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True
            )
        )

        self.meme_db = self._load_meme_db()
        self.current_meme_frames = [] # List of frames for GIF
        self.current_meme_index = 0
        self.current_state = "Waiting..."
        self.last_tag = None

    def _download_models(self):
        tasks = [
            (self.hand_model_path, "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"),
            (self.face_model_path, "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        ]
        for path, url in tasks:
            if not os.path.exists(path):
                print(f"Downloading {path}...")
                subprocess.run(['curl', '-L', url, '-o', path], shell=True)
                print("Done.")

    def _load_meme_db(self):
        db = []
        path = Path(self.assets_folder)
        # Added .gif to the search list
        files = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg")) + list(path.glob("*.gif"))
        
        print(f"\nScanning {len(files)} images/gifs for keywords...")
        for f in files:
            name = f.stem.lower()
            tags = []
            for state, keywords in self.EMOTION_KEYWORDS.items():
                if any(k in name for k in keywords) and len(tags) == 0:
                    tags.append(state)
            for state, keywords in self.GESTURE_KEYWORDS.items():
                if any(k in name for k in keywords) and len(tags) == 0:
                    tags.append(state)

            if tags:
                db.append({"path": str(f), "tags": tags, "name": f.stem})
                print(f"  [MATCH] {f.name} -> {tags}")
            else:
                db.append({"path": str(f), "tags": ["neutral"], "name": f.stem})
                print(f"  [NEUTRAL] {f.name}")
        
        print(f"Database loaded. Total recognized memes: {len(db)}\n")
        return db

    def get_random_meme_by_tag(self, tag):
        candidates = [m for m in self.meme_db if tag in m['tags']]
        if candidates:
            return random.choice(candidates)
        return None

    def load_image_or_gif(self, path):
        """Loads an image or GIF into a list of frames."""
        frames = []
        if path.lower().endswith(".gif"):
            try:
                # Load all frames from GIF using imageio
                gif_frames = iio.imread(path, index=None)
                for frame in gif_frames:
                    # Convert RGB to BGR for OpenCV
                    if frame.shape[-1] == 4: # Handle transparency
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames.append(frame)
            except Exception as e:
                print(f"Error loading GIF {path}: {e}")
        else:
            # Load standard image
            img = cv2.imread(path)
            if img is not None:
                frames.append(img)
        return frames

    def detect_facial_expression(self, landmarks, blendshapes):
        # 1. Check for Tongue Out 
        if blendshapes:
            for category in blendshapes:
                if category.category_name == 'tongueOut':
                    if category.score > 0.4: 
                        return "Tongue_Out"

        # 2. Geometric Checks
        pts = np.array([[l.x, l.y] for l in landmarks])
        
        UPPER_LIP = 13
        LOWER_LIP = 14
        LEFT_LIP = 61
        RIGHT_LIP = 291
        LEFT_BROW = 105
        RIGHT_BROW = 334
        
        mouth_open = np.linalg.norm(pts[UPPER_LIP] - pts[LOWER_LIP])
        mouth_width = np.linalg.norm(pts[LEFT_LIP] - pts[RIGHT_LIP])
        ratio = mouth_open / (mouth_width + 1e-6)
        
        if ratio > 0.5: return "surprise"

        lip_center_y = (pts[UPPER_LIP][1] + pts[LOWER_LIP][1]) / 2
        left_corner_y = pts[LEFT_LIP][1]
        right_corner_y = pts[RIGHT_LIP][1]
        
        if left_corner_y < lip_center_y - 0.01 and right_corner_y < lip_center_y - 0.01:
             return "smile"

        if left_corner_y > lip_center_y + 0.01 and right_corner_y > lip_center_y + 0.01:
            return "sad"

        brow_dist = np.linalg.norm(pts[LEFT_BROW] - pts[RIGHT_BROW])
        face_width = np.linalg.norm(pts[127] - pts[356])
        if (brow_dist / face_width) < 0.25: return "angry"

        return "neutral"

    def detect_raised_hand(self, hand_result):
        """
        Detect if hand is raised (palm open with hand positioned high in frame).
        This checks for Open_Palm gesture combined with hand position.
        """
        if not hand_result.gestures or not hand_result.hand_landmarks:
            return False
        
        # Check if gesture is Open_Palm (which MediaPipe recognizes)
        gesture_name = hand_result.gestures[0][0].category_name
        if gesture_name != "Open_Palm":
            return False
        
        # Check if hand is raised (wrist is higher than average finger position)
        landmarks = hand_result.hand_landmarks[0]
        wrist_y = landmarks[0].y
        
        # Average y position of fingertips
        fingertips_y = (landmarks[4].y + landmarks[8].y + landmarks[12].y + 
                       landmarks[16].y + landmarks[20].y) / 5
        
        # If wrist is significantly lower than fingertips, hand is raised
        if wrist_y > fingertips_y + 0.15:
            return True
        
        return False

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        
        if not cap.isOpened():
            print("Error: Camera not found.")
            return

        print("--- Meme Matcher Started ---")
        print("Press 'CTRL + C' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.frame_count += 1
            
            # Run AI Models 
            hand_result = self.gesture_recognizer.recognize_for_video(mp_image, self.frame_count)
            face_result = self.face_landmarker.detect_for_video(mp_image, self.frame_count)
            
            detected_tag = None
            
            # Check for Raised Hand first (priority gesture)
            if self.detect_raised_hand(hand_result):
                detected_tag = "Raised_Hand"
                self.current_state = "Gesture: Raised Hand"
            # Check Other Gestures 
            elif hand_result.gestures:
                gesture_name = hand_result.gestures[0][0].category_name
                if gesture_name in self.GESTURE_KEYWORDS:
                    detected_tag = gesture_name
                    self.current_state = f"Gesture: {gesture_name}"

            # Check Face 
            if not detected_tag and face_result.face_landmarks:
                blendshapes = face_result.face_blendshapes[0] if face_result.face_blendshapes else None
                emotion_name = self.detect_facial_expression(face_result.face_landmarks[0], blendshapes)
                if emotion_name != "neutral":
                    detected_tag = emotion_name
                    self.current_state = f"Face: {emotion_name}"
                else:
                    self.current_state = "Neutral"
            
            if not detected_tag:
                 detected_tag = "neutral"
                 self.current_state = "Neutral"

            # Update Meme / GIF 
            if detected_tag != self.last_tag:
                 self.last_tag = detected_tag
                 match = self.get_random_meme_by_tag(detected_tag)
                 
                 if match:
                     # Load Frames (Supports GIF or static image)
                     self.current_meme_frames = self.load_image_or_gif(match['path'])
                     self.current_meme_index = 0
                 else:
                     self.current_meme_frames = []

            # Then Render 
            h, w = frame.shape[:2]
            
            if self.current_meme_frames:
                # Cycle through frames for animation
                # Slow down animation by updating only every 3rd frame of video
                if self.frame_count % 3 == 0:
                    self.current_meme_index = (self.current_meme_index + 1) % len(self.current_meme_frames)
                
                current_img = self.current_meme_frames[self.current_meme_index]
                
                # Resize and Attach
                scale = h / current_img.shape[0]
                resized_meme = cv2.resize(current_img, (int(current_img.shape[1]*scale), h))
                
                mw = resized_meme.shape[1]
                combined = np.zeros((h, w + mw, 3), dtype=np.uint8)
                combined[:, :w] = frame
                combined[:, w:] = resized_meme
                
                cv2.putText(combined, self.current_state, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Meme Matcher", combined)
            else:
                cv2.putText(frame, "No image found!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Meme Matcher", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MemeMatcher(assets_folder="assets")
    app.run()