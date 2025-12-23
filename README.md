# Project Title



## Demo Video

[![Watch the video](https://img.youtube.com/vi/ufUAakjEDVi6hAG1/0.jpg)](https://youtu.be/XrNSpenP5vQ?si=ufUAakjEDVi6hAG1)

## Setup

Follow these steps to get the project up and running on your local machine.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-project-name.git
cd your-project-name
```

### 2. Create a virtual environment (Python 3.11 recommended)

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Create a file named `.env` in the root directory of the project and add your API key(s). For example, if you're using OpenAI:

```
OPENAI_API_KEY="your_openai_api_key_here"
# Add other API keys as needed (e.g., ANTHROPIC_API_KEY, GOOGLE_API_KEY)
```

### 5. Start the application

```bash
python app.py
```

### 6. Start a MIDI listener (to see output values)

If your application generates MIDI output, you'll need a MIDI listener to observe the values. There are various tools available depending on your operating system:

- **Windows:** LoopMIDI, MIDI-OX
- **macOS:** Audio MIDI Setup (built-in), MIDI Monitor
- **Linux:** `aconnect`, `qjackctl`, `amidi`

For example, on macOS, you can open "Audio MIDI Setup", go to "Window" -> "Show MIDI Studio", and then use a tool like "MIDI Monitor" to see incoming MIDI messages.

On Linux, you might use `aconnect -o` to list output ports and then `amidi -p <port_number> -d` to dump MIDI data from a specific port.
