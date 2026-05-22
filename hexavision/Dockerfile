FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxcb-cursor0 \
        libxcb-glx0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-util1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libxkbcommon0 \
        libdbus-1-3 \
        libfontconfig1 \
        libxrender1 \
        libxi6 \
        libxtst6 \
        libxext6 \
        libegl1 \
        libopengl0 \
        libsm6 \
        libice6 \
        mesa-utils \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

RUN rm -rf /usr/local/lib/python3.11/site-packages/cv2/qt/plugins

COPY main.py .

ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_XCB_GL_INTEGRATION=none

RUN mkdir -p /app/output
CMD ["python", "main.py"]