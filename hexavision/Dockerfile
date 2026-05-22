FROM python:3.11-slim AS base

# ── 1. System dependencies
#  libGL / libglib2.0  → OpenCV
#  libxcb* / libx11*  → PyQt5 (X11 windowing)
#  libdbus-1-3        → Qt internal IPC
#  libfontconfig1     → Qt font rendering
#  libxkbcommon-x11   → Qt keyboard handling
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libxcb1 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xfixes0 \
        libxkbcommon-x11-0 \
        libdbus-1-3 \
        libfontconfig1 \
        libxrender1 \
        libxi6 \
        libxtst6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Python deps (cached layer) 
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 3. App source 
COPY main.py .

# 4. Runtime environment 
#  QT_X11_NO_MITSHM  → fixes shared-memory crash inside Docker
#  DISPLAY           → default X11 display; override at runtime
ENV QT_X11_NO_MITSHM=1 \
    DISPLAY=:0

# output folder inside the container (mount a host dir over this)
RUN mkdir -p /app/output

# 5. Entrypoint 
CMD ["python", "main.py"]
