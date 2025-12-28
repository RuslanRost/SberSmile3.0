import json
import os
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from time import time, sleep

import cv2 as cv
import dlib
import numpy as np

from imutils import face_utils
from numpy.linalg import norm


DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_url": "",
    "camera_capture_resolution": [1280, 720],
    "processing_scale": 1.0,
    "lip_jaw_ratio_thresh": 0.44,
    "mouth_opening_ratio_thresh": 1.05,
    "face_center_weight": 1.0,
    "face_area_weight": 1.0,
    "detect_every_n": 1,
    "tcp_host": "127.0.0.1",
    "tcp_port": 5005,
    "smile_hold_seconds": 0.5,
    "smile_on_debounce_seconds": 0.15,
    "smile_off_debounce_seconds": 0.2,
    "stream_http_enabled": True,
    "stream_http_host": "127.0.0.1",
    "stream_http_port": 8080,
    "stream_http_resolution": [0, 0],
    "stream_http_jpeg_quality": 80,
    "rtsp_tcp": True,
    "rtsp_backend": "auto",
    "warmup_frames": 30,
    "log_fps": True,
    "fps_interval_sec": 5.0,
}


def load_config(path: Path):
    cfg = DEFAULT_CONFIG.copy()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in data.items() if k in cfg})
        except Exception as exc:
            print(f"Failed to read config {path}, using defaults. Error: {exc}")
    cfg["camera_index"] = int(cfg["camera_index"])
    cfg["camera_url"] = str(cfg.get("camera_url", "") or "")
    cfg["camera_capture_resolution"] = tuple(int(x) for x in cfg["camera_capture_resolution"])
    cfg["processing_scale"] = float(cfg["processing_scale"])
    cfg["lip_jaw_ratio_thresh"] = float(cfg["lip_jaw_ratio_thresh"])
    cfg["mouth_opening_ratio_thresh"] = float(cfg["mouth_opening_ratio_thresh"])
    cfg["face_center_weight"] = float(cfg["face_center_weight"])
    cfg["face_area_weight"] = float(cfg["face_area_weight"])
    cfg["detect_every_n"] = max(1, int(cfg["detect_every_n"]))
    cfg["tcp_host"] = str(cfg["tcp_host"])
    cfg["tcp_port"] = int(cfg["tcp_port"])
    cfg["smile_hold_seconds"] = float(cfg["smile_hold_seconds"])
    cfg["smile_on_debounce_seconds"] = float(cfg["smile_on_debounce_seconds"])
    cfg["smile_off_debounce_seconds"] = float(cfg["smile_off_debounce_seconds"])
    cfg["stream_http_enabled"] = bool(cfg["stream_http_enabled"])
    cfg["stream_http_host"] = str(cfg["stream_http_host"])
    cfg["stream_http_port"] = int(cfg["stream_http_port"])
    cfg["stream_http_resolution"] = tuple(int(x) for x in cfg["stream_http_resolution"])
    cfg["stream_http_jpeg_quality"] = int(cfg["stream_http_jpeg_quality"])
    cfg["rtsp_tcp"] = bool(cfg["rtsp_tcp"])
    cfg["rtsp_backend"] = str(cfg.get("rtsp_backend", "auto")).lower()
    cfg["warmup_frames"] = int(cfg["warmup_frames"])
    cfg["log_fps"] = bool(cfg["log_fps"])
    cfg["fps_interval_sec"] = float(cfg["fps_interval_sec"])
    return cfg


def setup_camera_props(cam, resolution):
    try:
        w, h = resolution
        if w > 0 and h > 0:
            cam.set(cv.CAP_PROP_FRAME_WIDTH, w)
            cam.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    except Exception:
        pass


def open_video_source(source, use_tcp, backend):
    if isinstance(source, str) and source.lower().startswith("rtsp://"):
        candidates = []
        if use_tcp and "rtsp_transport=tcp" not in source:
            sep = "&" if "?" in source else "?"
            candidates.append(f"{source}{sep}rtsp_transport=tcp")
        candidates.append(source)
        if source.endswith("/"):
            candidates.append(source.rstrip("/"))
        else:
            candidates.append(source + "/")
        try_ffmpeg = backend == "ffmpeg"
        try_auto = backend == "auto"
        if try_auto:
            cap = cv.VideoCapture(source)
            if cap.isOpened():
                print(f"RTSP opened (auto): {source}")
                return cap
        if try_ffmpeg or not try_auto:
            for url in candidates:
                cap = cv.VideoCapture(url, cv.CAP_FFMPEG)
                if cap.isOpened():
                    print(f"RTSP opened (FFMPEG): {url}")
                    return cap
        print("Failed RTSP URLs:")
        for url in candidates:
            print(f"  {url}")
        return cap
    return cv.VideoCapture(source)


def choose_best_face(faces, gray_shape, center_weight, area_weight):
    if not faces:
        return None
    center = np.array([gray_shape[1] / 2.0, gray_shape[0] / 2.0])
    best = None
    for face in faces:
        fx, fy = face.left(), face.top()
        fw, fh = face.right() - fx, face.bottom() - fy
        area = max(0, fw) * max(0, fh)
        face_center = np.array([fx + fw / 2.0, fy + fh / 2.0])
        dist = np.linalg.norm(face_center - center)
        dist_norm = dist / max(1.0, center[0])
        area_norm = area / max(1.0, gray_shape[0] * gray_shape[1])
        score = center_weight * dist_norm - area_weight * area_norm
        if best is None or score < best[0]:
            best = (score, face)
    return best[1]


def main():
    cfg_path = Path("config.json").resolve()
    cfg = load_config(cfg_path)
    print(f"Working dir: {os.getcwd()}")
    print(f"Config path: {cfg_path}")
    lip_jaw_ratio_thresh = cfg["lip_jaw_ratio_thresh"]
    mouth_opening_ratio_thresh = cfg["mouth_opening_ratio_thresh"]
    face_center_weight = cfg["face_center_weight"]
    face_area_weight = cfg["face_area_weight"]
    smile_hold_seconds = cfg["smile_hold_seconds"]
    smile_on_debounce_seconds = cfg["smile_on_debounce_seconds"]
    smile_off_debounce_seconds = cfg["smile_off_debounce_seconds"]
    stream_http_enabled = cfg["stream_http_enabled"]
    stream_http_host = cfg["stream_http_host"]
    stream_http_port = cfg["stream_http_port"]
    stream_http_resolution = cfg["stream_http_resolution"]
    stream_http_jpeg_quality = cfg["stream_http_jpeg_quality"]
    warmup_frames = max(0, cfg["warmup_frames"])
    log_fps = cfg["log_fps"]
    fps_interval_sec = max(1.0, cfg["fps_interval_sec"])
    print(f"camera_url: {cfg['camera_url']}")
    print(f"log_fps: {log_fps}, fps_interval_sec: {fps_interval_sec}, warmup_frames: {warmup_frames}")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cam_source = cfg["camera_url"] if cfg["camera_url"] else cfg["camera_index"]
    cap = open_video_source(cam_source, cfg["rtsp_tcp"], cfg["rtsp_backend"])
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv.CAP_PROP_CONVERT_RGB, 1)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    setup_camera_props(cap, cfg["camera_capture_resolution"])

    host = cfg["tcp_host"]
    port = cfg["tcp_port"]
    detect_every_n = cfg["detect_every_n"]

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"Detector listening on {host}:{port}")

    latest_frame = None
    latest_idx = -1
    frame_lock = threading.Lock()
    stop_event = threading.Event()
    fps_start = time()
    fps_frames = 0

    def stream_server():
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/stream":
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while True:
                    with frame_lock:
                        frame = None if latest_frame is None else latest_frame.copy()
                    if frame is None:
                        sleep(0.02)
                        continue
                    w, h = stream_http_resolution
                    if w > 0 and h > 0:
                        frame = cv.resize(frame, (w, h))
                    ok, jpg = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, stream_http_jpeg_quality])
                    if not ok:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpg.tobytes())
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break

            def log_message(self, format, *args):
                return

        httpd = HTTPServer((stream_http_host, stream_http_port), Handler)
        print(f"HTTP stream on http://{stream_http_host}:{stream_http_port}/stream")
        httpd.serve_forever()

    if stream_http_enabled:
        threading.Thread(target=stream_server, daemon=True).start()

    conn, addr = server.accept()
    print(f"UI connected from {addr}")

    def send_msg(data):
        payload = json.dumps(data).encode("utf-8")
        header = struct.pack("!I", len(payload))
        try:
            conn.sendall(header + payload)
            return True
        except OSError:
            return False

    last_smile = False
    debounced_smiling = False
    raw_smile_started_at = None
    raw_not_smile_started_at = None
    smile_started_at = None
    sent_active = False

    def capture_loop():
        nonlocal latest_frame, latest_idx, fps_start, fps_frames
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                sleep(0.002)
                continue
            if frame.dtype != np.uint8:
                frame = cv.convertScaleAbs(frame)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            with frame_lock:
                latest_frame = frame
                latest_idx += 1

            if log_fps:
                fps_frames += 1
                now_fps = time()
                if now_fps - fps_start >= fps_interval_sec:
                    fps = fps_frames / max(1e-6, now_fps - fps_start)
                    print(f"Camera FPS: {fps:.1f}", flush=True)
                    fps_start = now_fps
                    fps_frames = 0

    threading.Thread(target=capture_loop, daemon=True).start()

    last_processed_idx = -1
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
            frame_idx = latest_idx
        if frame is None or frame_idx == last_processed_idx:
            sleep(0.002)
            continue
        last_processed_idx = frame_idx

        if warmup_frames > 0:
            warmup_frames -= 1
            continue

        if frame_idx % detect_every_n == 0:
            h, w = frame.shape[:2]
            side = min(h, w)
            crop_x = (w - side) // 2
            crop_y = (h - side) // 2
            cropped = frame[crop_y:crop_y + side, crop_x:crop_x + side]
            local_scale = cfg["processing_scale"]
            if local_scale <= 0 or local_scale > 1:
                local_scale = 1.0
            if local_scale != 1.0:
                scaled_side = max(1, int(side * local_scale))
                cropped = cv.resize(cropped, (scaled_side, scaled_side))
            gray_image = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
            gray_image = np.ascontiguousarray(gray_image, dtype=np.uint8)

            faces = detector(gray_image, 1)
            face = choose_best_face(faces, gray_image.shape, face_center_weight, face_area_weight)
            result = {"frame_idx": frame_idx, "smile": False}
            if face is not None:
                x_face, y_face = face.left(), face.top()
                w_face, h_face = face.right() - x_face, face.bottom() - y_face
                landmarks = predictor(gray_image, face)
                landmarks = face_utils.shape_to_np(landmarks)
                scale_inv = 1.0 / local_scale if local_scale != 0 else 1.0
                landmarks = landmarks.astype(np.float32) * scale_inv
                landmarks[:, 0] += crop_x
                landmarks[:, 1] += crop_y
                x_face = int(x_face * scale_inv + crop_x)
                y_face = int(y_face * scale_inv + crop_y)
                w_face = int(w_face * scale_inv)
                h_face = int(h_face * scale_inv)

                mouth_landmarks = landmarks[48:60]
                mouth_poly = np.array(mouth_landmarks, np.int32).reshape((-1, 1, 2))

                jaw_width = norm(landmarks[2] - landmarks[14])
                lips_width = norm(landmarks[54] - landmarks[48])
                lip_jaw_ratio = lips_width / jaw_width

                mouth_opening = norm(landmarks[57] - landmarks[51])
                mouth_nose = norm(landmarks[33] - landmarks[51])

                mouth_to_cheeks = norm(landmarks[48] - landmarks[3]) + norm(landmarks[54] - landmarks[13])
                mouth_to_jaw = norm(landmarks[48] - landmarks[5]) + norm(landmarks[54] - landmarks[11])

                smile = False
                if lip_jaw_ratio > lip_jaw_ratio_thresh:
                    if mouth_opening / mouth_nose >= mouth_opening_ratio_thresh:
                        smile = True

                result = {
                    "frame_idx": frame_idx,
                    "smile": smile,
                    "face": (x_face, y_face, w_face, h_face),
                    "mouth_poly": mouth_poly.tolist(),
                    "frame_size": (frame.shape[1], frame.shape[0]),
                }
            else:
                result = {
                    "frame_idx": frame_idx,
                    "smile": False,
                    "face": None,
                    "mouth_poly": None,
                    "frame_size": (frame.shape[1], frame.shape[0]),
                }
            last_smile = result["smile"]
            if not send_msg({"cmd": "detect", "data": result}):
                stop_event.set()
                break

        now = time()
        if last_smile:
            raw_smile_started_at = raw_smile_started_at or now
            raw_not_smile_started_at = None
            if not debounced_smiling and (now - raw_smile_started_at) >= smile_on_debounce_seconds:
                debounced_smiling = True
        else:
            raw_not_smile_started_at = raw_not_smile_started_at or now
            raw_smile_started_at = None
            if debounced_smiling and (now - raw_not_smile_started_at) >= smile_off_debounce_seconds:
                debounced_smiling = False

        if debounced_smiling:
            if smile_started_at is None:
                smile_started_at = now
            elif not sent_active and (now - smile_started_at) >= smile_hold_seconds:
                if not send_msg({"cmd": "trigger", "ts": now}):
                    stop_event.set()
                    break
                sent_active = True
        else:
            smile_started_at = None
            sent_active = False

    stop_event.set()
    conn.close()
    server.close()
    cap.release()


if __name__ == "__main__":
    main()
