# Meme Matcher – Real-time Gesture & Expression to Meme

A real-time app that uses your webcam to detect **hand gestures** and **facial expressions**, then shows a meme that matches what you’re doing. Built with MediaPipe (gesture + face models) and OpenCV.

---

## What It Does

Run the script, show your face and hands to the camera, and make different gestures or expressions. The app picks a meme from the `assets` folder whose **filename** matches the detected gesture or expression and displays it next to your camera feed. Supports static images (JPG, PNG) and **GIFs** (animated memes).

---

## Installation

### Prerequisites

- **Python 3.8+**
- **Webcam**
- **curl** (used to download models on first run; on Windows you may need Git Bash or WSL, or download the `.task` files manually)

### Setup (with a virtual environment)

Using a virtual environment keeps this project’s dependencies separate from your system Python.

1. Clone or download the project and go to its folder:
   ```bash
   cd make_me_a_meme
   ```

2. Create and activate a virtual environment:

   **Windows (PowerShell):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   **Windows (Command Prompt):**
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   Your prompt should show `(venv)` when the environment is active.

3. Install dependencies from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   python main.py
   ```

When you’re done, type `deactivate` to leave the virtual environment.

On first run, the script downloads two MediaPipe model files (~few MB) into the project folder: `gesture_recognizer.task` and `face_landmarker.task`.

---

## How to Use

1. Run: `python main.py`
2. Allow webcam access when prompted.
3. Look at the camera and try:
   - **Hand gestures**: thumbs up, thumbs down, peace sign, open palm, pointing up, fist, “I love you” sign.
   - **Facial expressions**: smile, sad, angry, surprised (open mouth), neutral, tongue out.
4. When the app detects a gesture or expression, it shows a matching meme (if one exists for that tag) next to the video. The current state (e.g. `Gesture: Thumb_Up` or `Face: smile`) is shown on screen.
5. Press **`q`** in the window to quit.

**Note:** Gestures are checked first. If no gesture is detected, face expression is used. So a clear hand gesture will override your face.

---

## The Model’s Limited Gestures and Face Expressions

The script uses **two** MediaPipe models. Their outputs are fixed; you can only match memes to what these models actually detect.

### Hand gestures (Gesture Recognizer)

The **gesture recognizer** only outputs a fixed set of categories. The script uses these (same names as in code):

| Gesture tag    | Typical meaning / pose                    |
|----------------|-------------------------------------------|
| `Thumb_Up`     | Thumbs up                                 |
| `Thumb_Down`   | Thumbs down                               |
| `Victory`      | Peace / V sign                            |
| `Open_Palm`    | Open palm (e.g. stop, high five)          |
| `Pointing_Up`  | Index finger pointing up                  |
| `Closed_Fist`  | Fist (punch, power, rock)                 |
| `ILoveYou`     | “I love you” sign (thumb + index + pinky) |

You **cannot** add new gesture types without changing the model; you can only map these tags to memes via the **file naming convention** (see below).

### Face expressions (Face Landmarker + custom logic)

Face detection uses the **Face Landmarker** (478 landmarks) plus **blendshapes** (for tongue). The script then uses simple geometric rules and one blendshape to map to these **expression tags**:

| Expression tag | How it’s detected |
|----------------|--------------------|
| `Tongue_Out`   | Face blendshape `tongueOut` above a score threshold (~0.4). |
| `surprise`     | Mouth open vs width ratio (wide open mouth). |
| `smile`        | Lip corners higher than lip center (upturned mouth). |
| `sad`          | Lip corners lower than lip center (downturned mouth). |
| `angry`        | Eyebrows closer together (brow distance vs face width). |
| `neutral`      | Default when none of the above conditions are met. |

So the model does **not** recognize arbitrary expressions—only these six cases. Lighting, angle, and face shape can affect how often each is triggered.

---

## File Naming Convention

Memes are matched **only by the filename** (the stem: name without extension). The script scans the `assets` folder for `.jpg`, `.jpeg`, `.png`, and `.gif` and assigns **one tag** per file:

1. It checks the filename (lowercased) against **emotion keywords**; if any keyword is found, the file gets that **emotion tag**.
2. If no emotion keyword matches, it checks **gesture keywords**; if any match, the file gets that **gesture tag**.
3. If nothing matches, the file gets the tag **`neutral`**.

So: **one tag per meme**, and the **first** matching keyword type (emotion then gesture) wins.

### Emotion keywords (→ expression tag)

Use these **words in the filename** to map to a face expression tag (e.g. `happy_cat.png` → `smile`):

| Tag         | Example keywords in filename |
|-------------|------------------------------|
| smile       | smile, happy, lol, laugh, grin, joy, fun |
| sad         | sad, cry, tear, upset, depressed, frown |
| angry       | angry, mad, rage, grumpy, serious, hate |
| surprise    | wow, shock, surprise, omg, pog, open, amazed, scary |
| neutral     | neutral, stare, waiting, bored |
| Tongue_Out  | tongue, bleh, silly, crazy, mlem, lick |

### Gesture keywords (→ gesture tag)

Use these **words in the filename** to map to a gesture tag (e.g. `thumbs_up.png` → `Thumb_Up`):

| Tag          | Example keywords in filename |
|--------------|------------------------------|
| Thumb_Up     | thumbs_up, like, good, approve, cool |
| Thumb_Down   | thumbs_down, dislike, bad, boo |
| Victory      | peace, victory, cool, vibes |
| Open_Palm    | stop, halt, wait, high_five |
| Pointing_Up  | point, look, up, idea, nerd |
| Closed_Fist  | fist, punch, strength, power, rock, fight |
| ILoveYou     | love, heart, rock_on, metal, spider |

### Examples

- `thumbs_up.png` → gesture **Thumb_Up**
- `happy_shrek.png` → emotion **smile** (keyword “happy”)
- `bored_cat.png` → **neutral** (keyword “bored”)
- `tongue_out_cat.gif` → **Tongue_Out** (keyword “tongue”)
- `scary_gf.jpg` → **surprise** (keyword “scary”)
- `nerd_dog.png` → **Pointing_Up** (keyword “nerd” in gesture list)

If you add a new meme, name it so it contains at least one keyword from the table for the tag you want; otherwise it will be treated as **neutral**.

---

## Customization

### 1. Assets folder

Change which folder is scanned for memes:

```python
app = MemeMatcher(assets_folder="my_memes")
app.run()
```

### 2. Adding or changing keyword → tag mapping

Edit `main.py` and update the two dictionaries in `MemeMatcher`:

- **`EMOTION_KEYWORDS`**  
  Keys are the expression tags (e.g. `"smile"`, `"Tongue_Out"`). Values are lists of words; if any word appears in the **filename** (lowercased), that asset gets that emotion tag.

- **`GESTURE_KEYWORDS`**  
  Keys are the gesture tags (e.g. `"Thumb_Up"`, `"Victory"`). Same idea: filename must contain one of the listed words to get that gesture tag.

Adding a new keyword to an existing tag (e.g. `"yeah"` under `Thumb_Up`) will make files like `yeah_meme.png` match that gesture. You cannot invent new gesture or expression **types**—only map new **keywords** to the existing tags.

### 3. Adding new memes

1. Put the image or GIF in the `assets` folder (or your custom folder).
2. Name the file so it includes a keyword from `EMOTION_KEYWORDS` or `GESTURE_KEYWORDS` for the tag you want (see tables above).  
   No match → tag is **neutral**.

### 4. Face expression sensitivity

In `detect_facial_expression()` you can tweak:

- **Tongue:** `if category.score > 0.4` — lower value = more sensitive to tongue out.
- **Surprise:** `if ratio > 0.5` — mouth open/width ratio; lower = easier to trigger “surprise”.
- **Angry:** `if (brow_dist / face_width) < 0.25` — threshold for brow distance.

Adjust these if your camera or face often mis-triggers or misses an expression.

### 5. GIF animation speed

Meme frames are advanced every 3 video frames. To change speed, edit:

```python
if self.frame_count % 3 == 0:
```

Use a larger number (e.g. `5`) for slower GIF playback, or `1` for faster.

---

## Summary

- **Use:** Run `python main.py`, use webcam, press `CRTL + C` to quit.
- **Customize:** Change `assets_folder`, add/change keywords in `EMOTION_KEYWORDS` and `GESTURE_KEYWORDS`, add memes with the right names, tune expression thresholds and GIF speed.
- **Model limits:** Only the listed hand gestures and the six face expression tags are supported; matching is by filename keywords only.
- **File naming:** Include one keyword from the emotion or gesture tables in the filename so the meme gets the correct tag; otherwise it stays **neutral**.

---

## Credits

- **Original project:** [kristelTech/make_me_a_meme](https://github.com/kristelTech/make_me_a_meme) – real-time facial expression and gesture matching to memes.
- **Modified by:** [Anes-Zaidi](https://github.com/Anes-Zaidi) – this repo is a modified version of the above.
- **MediaPipe** – Gesture Recognizer and Face Landmarker models (Google).
- **OpenCV** – Video capture and display.
- **imageio** – GIF frame loading.
