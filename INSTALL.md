# Installing RedlineCheck on another computer (Windows)

RedlineCheck runs entirely on the local machine — nothing is uploaded. These steps
get it onto a computer that has **nothing installed** (no Python, no anything).

---

## A. Put it on the other computer

1. Zip this whole folder (the one containing `app.py`, `requirements.txt`, the
   `redlinecheck` and `installer` folders). Right‑click it → **Send to → Compressed
   (zipped) folder**.
2. Send the zip (email, USB stick, OneDrive…) to the other computer.
3. On the other computer, **right‑click the zip → Extract All**, and extract it to a
   permanent spot — e.g. `Documents\RedlineCheck`.
   **Don't move the folder after you install** (the Desktop icons point to it).

## B. Install (one time, click‑through)

4. Open the `installer` folder and **double‑click `Install RedlineCheck`**.
   - A black window appears and sets everything up automatically (it installs Python,
     the OCR text‑reader, and the program's parts). This can take a few minutes.
   - If Windows shows a blue "Windows protected your PC" box, click **More info →
     Run anyway** (it's just an unsigned script).
   - If it says *"Python was just installed — run me again"*, close the window and
     double‑click **Install RedlineCheck** one more time.
5. When it says **"All done!"**, you'll have two new icons on the Desktop:
   **RedlineCheck** and **Stop RedlineCheck**.

## C. Use it

6. **Double‑click `RedlineCheck`** on the Desktop. A black window opens and, a few
   seconds later, the app opens in your web browser.
7. Drag in your **redlined** drawing(s) on the left and the architect's **revised**
   drawing(s) on the right (a single pair, or whole folders), then click **Compare**.
8. **To stop the app:** close the black window, **or** double‑click
   **Stop RedlineCheck** on the Desktop.

That's it. To run it again another day, just double‑click **RedlineCheck** again.

---

## D. Optional — the AI reader (only on a powerful computer)

RedlineCheck works great with just OCR (installed above). There is also an optional
local **AI vision model** that can confirm a few extra changes OCR can't read. It is
**off by default** and is only worth turning on if the computer has a **good graphics
card (GPU)** — on an ordinary laptop it is very slow.

To enable it later:
1. Install Ollama from <https://ollama.com/download> (or run `winget install Ollama.Ollama`).
2. Open a Command Prompt and run: `ollama pull moondream`
3. In the app's sidebar, tick **"Local VLM (Ollama vision model)"** before clicking Compare.

---

## Troubleshooting

- **"winget is not recognized"** (older Windows): install Python yourself from
  <https://www.python.org/downloads/> — during setup **tick "Add python.exe to PATH"** —
  then run **Install RedlineCheck** again. (Tesseract OCR, optional, is at
  <https://github.com/UB-Mannheim/tesseract/wiki>.)
- **The browser didn't open:** open your browser and go to <http://localhost:8501>.
- **It won't start / "address in use":** double‑click **Stop RedlineCheck**, then
  **RedlineCheck** again.
- **Nothing happens when I drop a folder:** RedlineCheck only reads PDFs; other files
  are ignored automatically. Make sure each side has at least one PDF drawing.

## For a polished installer (optional, advanced)

`installer/RedlineCheck.iss` is an [Inno Setup](https://jrsoftware.org/isdl.php)
script. Installing Inno Setup and opening this file lets you **Build** a single
`RedlineCheckSetup.exe` that recipients run like any normal program. The `.bat`
installer above needs no extra tools and is the simplest path.
