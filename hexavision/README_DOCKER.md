# Wirebond Inspection Pipeline — Docker Guide

## Project Structure

```
wirebond-inspection/
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── output/          ← auto-created; PDF reports & crops land here
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Docker | 24.x + |
| Docker Compose | v2 (bundled with Docker Desktop) |

---

## Step-by-Step Setup

### 1 — Clone / copy your project files

Make sure `main.py`, `requirements.txt`, `Dockerfile`, and
`docker-compose.yml` are all in the **same folder**.

---

### 2 — Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and paste your Roboflow API key:

```
ROBOFLOW_API_KEY=rf_xxxxxxxxxxxxxxxx
```

---

### 3 — Allow Docker to connect to your X11 display (Linux)

The app renders a PyQt5 GUI. Docker needs permission to draw on
your host screen:

```bash
xhost +local:docker
```

> **macOS users:** Install [XQuartz](https://www.xquartz.org/),
> then run `xhost + 127.0.0.1` and set `DISPLAY=host.docker.internal:0`.

---

### 4 — Build the image

```bash
docker compose build
```

This step:
- Pulls `python:3.11-slim`
- Installs all X11 / OpenCV / Qt system libraries
- Installs Python deps from `requirements.txt`

First build takes ~3–5 min. Subsequent builds use the **layer cache**
and are instant unless `requirements.txt` changes.

---

### 5 — Run the app

```bash
docker compose up
```

The PyQt5 window opens on your host desktop. Use it normally —
load a reference image, load a test image, pick an output folder
(inside the container, use `/app/output`), then click **Run All**.

PDF reports and cropped step-hole images will appear in `./output/`
on your host machine.

---

### 6 — Stop the container

Close the GUI window, or press `Ctrl+C` in the terminal where
`docker compose up` is running.

---

## Useful Commands

```bash
# Rebuild from scratch (no cache)
docker compose build --no-cache

# Run in the background
docker compose up -d

# Open a shell inside the running container
docker exec -it wirebond_app bash

# View logs
docker compose logs -f

# Remove the container (keeps the image)
docker compose down

# Remove the image too
docker compose down --rmi all
```

---

## How the Dockerfile Works (line-by-line)

```
FROM python:3.11-slim          # Minimal Debian base with Python 3.11
RUN apt-get install ...        # System libs: OpenCV (libGL), PyQt5 (X11/xcb)
COPY requirements.txt .        # Copy deps list first — cached unless it changes
RUN pip install -r ...         # Install Python packages (heavy, cached layer)
COPY main.py .                 # App source — only this layer rebuilds on code edits
ENV QT_X11_NO_MITSHM=1         # Prevents Qt shared-memory crash inside Docker
CMD ["python", "main.py"]      # Default command when container starts
```

**Why copy `requirements.txt` before `main.py`?**
Docker rebuilds only the layers that changed and everything after them.
Putting pip install before the source copy means editing `main.py`
does **not** re-run the slow pip install step.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cannot connect to X server :0` | Run `xhost +local:docker` on host |
| `qt.qpa.xcb: could not connect to display` | Check `DISPLAY` env var matches host |
| `ROBOFLOW_API_KEY not set` | Verify `.env` file exists and key is correct |
| `libGL error` on startup | Already handled — `libgl1` is in the Dockerfile |
| GUI very slow | Normal for X11 forwarding over a VM; use a native Linux host |
