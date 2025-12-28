import json
import queue
import socket
import struct
import threading
from pathlib import Path
from time import monotonic, sleep

import cv2 as cv
import numpy as np


DEFAULT_CONFIG = {
    "output_resolution": [0, 0],
    "ui_camera_enabled": False,
    "ui_camera_index": 0,
    "ui_camera_url": "",
    "ui_camera_resolution": [640, 480],
    "video_size": [336, 672],
    "trigger_sync_frames": [0, 150, 283, 395, 480, 649, 741, 817, 892, 1025, 1138, 1238],
    "tcp_host": "127.0.0.1",
    "tcp_port": 5005,
    "rtsp_tcp": True,
}


def load_config(path: Path):
    cfg = DEFAULT_CONFIG.copy()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in data.items() if k in cfg})
        except Exception as exc:
            print(f"Failed to read config {path}, using defaults. Error: {exc}")
    cfg["output_resolution"] = tuple(int(x) for x in cfg["output_resolution"])
    cfg["ui_camera_enabled"] = bool(cfg["ui_camera_enabled"])
    cfg["ui_camera_index"] = int(cfg["ui_camera_index"])
    cfg["ui_camera_url"] = str(cfg.get("ui_camera_url", "") or "")
    cfg["ui_camera_resolution"] = tuple(int(x) for x in cfg["ui_camera_resolution"])
    cfg["video_size"] = tuple(int(x) for x in cfg["video_size"])
    cfg["trigger_sync_frames"] = [int(x) for x in cfg["trigger_sync_frames"]]
    cfg["tcp_host"] = str(cfg["tcp_host"])
    cfg["tcp_port"] = int(cfg["tcp_port"])
    cfg["rtsp_tcp"] = bool(cfg["rtsp_tcp"])
    return cfg


def get_screen_size():
    import ctypes
    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def setup_camera_props(cam, resolution):
    try:
        w, h = resolution
        if w > 0 and h > 0:
            cam.set(cv.CAP_PROP_FRAME_WIDTH, w)
            cam.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    except Exception:
        pass


class VideoLooper:
    def __init__(self, path: Path):
        self.path = path
        self.cap = None
        self.frame_idx = -1
        self.fps = 30.0
        self.loop_started_at = monotonic()
        self._open()

    def _open(self):
        if self.cap:
            self.cap.release()
        self.cap = cv.VideoCapture(str(self.path))
        self.frame_idx = -1
        fps = self.cap.get(cv.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        self.loop_started_at = monotonic()

    def frame(self):
        if not self.cap or not self.cap.isOpened():
            self._open()
        looped = False
        now = monotonic()
        expected_idx = int((now - self.loop_started_at) * self.fps)
        if self.frame_idx < 0:
            expected_idx = max(expected_idx, 0)
        max_skip = 60
        while self.frame_idx < expected_idx and max_skip > 0:
            ret, _ = self.cap.read()
            if not ret:
                self._open()
                looped = True
                return None, -1, looped
            self.frame_idx = 0 if self.frame_idx < 0 else self.frame_idx + 1
            max_skip -= 1
        ret, frame = self.cap.read()
        if not ret:
            self._open()
            looped = True
            return None, -1, looped
        self.frame_idx = 0 if self.frame_idx < 0 else self.frame_idx + 1
        return frame, self.frame_idx, looped

    def release(self):
        if self.cap:
            self.cap.release()


class VideoPlayerThread:
    def __init__(self, loop: bool):
        self.loop = loop
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.finished = True
        self.frame_idx = -1
        self.fps = 30.0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def set_source(self, path: Path | None):
        with self.lock:
            if self.cap:
                self.cap.release()
            self.cap = cv.VideoCapture(str(path)) if path else None
            self.latest_frame = None
            self.finished = False if path else True
            self.frame_idx = -1
            self.fps = 30.0
            if self.cap:
                fps = self.cap.get(cv.CAP_PROP_FPS)
                if fps and fps > 0:
                    self.fps = fps

    def _run(self):
        while not self.stop_event.is_set():
            with self.lock:
                cap = self.cap
                loop = self.loop
            if cap is None:
                sleep(0.02)
                continue
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    with self.lock:
                        self.frame_idx = -1
                    continue
                with self.lock:
                    self.finished = True
                    self.cap.release()
                    self.cap = None
                continue
            with self.lock:
                self.latest_frame = frame
                self.frame_idx = 0 if self.frame_idx < 0 else self.frame_idx + 1
            frame_time = 1.0 / max(1.0, self.fps)
            sleep(frame_time)

    def get_frame(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            finished = self.finished
            frame_idx = self.frame_idx
        return frame, finished, frame_idx

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap:
                self.cap.release()


class CameraThread:
    def __init__(self, source, resolution, rtsp_tcp):
        self.source = source
        self.resolution = resolution
        self.rtsp_tcp = rtsp_tcp
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        if isinstance(self.source, int):
            self.cap = cv.VideoCapture(self.source, cv.CAP_DSHOW)
        else:
            source = self.source
            if isinstance(source, str) and source.lower().startswith("rtsp://"):
                if self.rtsp_tcp and "rtsp_transport=tcp" not in source:
                    sep = "&" if "?" in source else "?"
                    source = f"{source}{sep}rtsp_transport=tcp"
                self.cap = cv.VideoCapture(source, cv.CAP_FFMPEG)
            else:
                self.cap = cv.VideoCapture(source)
        if not self.cap.isOpened():
            return
        setup_camera_props(self.cap, self.resolution)
        self.cap.set(cv.CAP_PROP_CONVERT_RGB, 1)
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


def find_idle_video() -> Path | None:
    idle_dir = Path("idle")
    if not idle_dir.exists():
        return None
    for candidate in sorted(idle_dir.iterdir()):
        if candidate.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            return candidate
    return None


def list_active_videos():
    active_dir = Path("active")
    if not active_dir.exists():
        return []
    return [p for p in sorted(active_dir.iterdir()) if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm")]


def play_video_once(path: Path):
    cap = cv.VideoCapture(str(path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def next_sync_frame(current_idx: int, frames: list[int]):
    for f in frames:
        if f >= current_idx:
            return f
    return frames[0] if frames else 0


def overlay_top_right(base: np.ndarray, overlay: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = base.shape[:2]
    ow, oh = size
    x1, y1 = max(w - ow, 0), 0
    x2, y2 = min(w, x1 + ow), min(h, oh)
    resized = cv.resize(overlay, (x2 - x1, y2 - y1))
    base[y1:y2, x1:x2] = resized
    return base


def place_camera_bottom_left(base: np.ndarray, cam_frame: np.ndarray, scale: float):
    if cam_frame is None:
        return base, None
    h, w = base.shape[:2]
    ch, cw = cam_frame.shape[:2]
    target_h = max(1, int(h * scale))
    target_w = max(1, int(cw * (target_h / ch)))
    if target_w > w:
        target_w = w
        target_h = int(ch * (target_w / cw))
    resized = cv.resize(cam_frame, (target_w, target_h))
    margin = 20
    x1 = margin
    y1 = max(0, h - target_h - margin)
    base[y1:y1 + target_h, x1:x1 + target_w] = resized
    return base, (x1, y1, target_w, target_h)


def place_frame_bottom_left(base: np.ndarray, frame: np.ndarray, size: tuple[int, int]):
    if frame is None:
        return base
    h, w = base.shape[:2]
    target_w, target_h = size
    if target_w <= 0 or target_h <= 0:
        return base
    target_w = min(target_w, w)
    target_h = min(target_h, h)
    resized = cv.resize(frame, (target_w, target_h))
    margin = 20
    x1 = margin
    y1 = max(0, h - target_h - margin)
    base[y1:y1 + target_h, x1:x1 + target_w] = resized
    return base


def recvall(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def main():
    cfg = load_config(Path("config.json"))
    host = cfg["tcp_host"]
    port = cfg["tcp_port"]

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    for _ in range(30):
        try:
            sock.connect((host, port))
            connected = True
            break
        except OSError:
            sleep(0.2)
    if not connected:
        raise RuntimeError(f"Cannot connect to detector at {host}:{port}")

    window_name = "Smile UI"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    screen_w, screen_h = get_screen_size()
    out_w, out_h = cfg["output_resolution"]
    if out_w > 0 and out_h > 0:
        screen_w, screen_h = out_w, out_h

    idle_path = find_idle_video()
    idle_player = VideoPlayerThread(loop=True)
    idle_player.start()
    if idle_path:
        idle_player.set_source(idle_path)
    active_player = VideoPlayerThread(loop=False)
    active_player.start()
    active_videos = list_active_videos()
    active_playing = False
    smile_started_at = None
    pending_active = False

    cam_thread = None
    if cfg["ui_camera_enabled"]:
        cam_source = cfg["ui_camera_url"] if cfg["ui_camera_url"] else cfg["ui_camera_index"]
        cam_thread = CameraThread(cam_source, cfg["ui_camera_resolution"], cfg["rtsp_tcp"])
        cam_thread.start()

    cmd_queue = queue.Queue()
    last_detect = None

    def receiver():
        while True:
            try:
                header = recvall(sock, 4)
            except OSError:
                break
            if not header:
                break
            meta_len = struct.unpack("!I", header)[0]
            meta = recvall(sock, meta_len)
            if not meta:
                break
            msg = json.loads(meta.decode("utf-8"))
            cmd_queue.put(msg)

    thread = threading.Thread(target=receiver, daemon=True)
    thread.start()

    while True:
        try:
            msg = cmd_queue.get_nowait()
            if msg.get("cmd") == "trigger":
                pending_active = True
            elif msg.get("cmd") == "detect":
                last_detect = msg.get("data")
        except queue.Empty:
            pass

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        if active_playing:
            active_frame, finished, _ = active_player.get_frame()
            if finished:
                active_playing = False
                if idle_path:
                    idle_player.set_source(idle_path)
                    idle_frame, _, _ = idle_player.get_frame()
                    if idle_frame is not None:
                        canvas = overlay_top_right(canvas, idle_frame, cfg["video_size"])
            elif active_frame is not None:
                canvas = overlay_top_right(canvas, active_frame, cfg["video_size"])
        if not active_playing:
            idle_frame, _, _ = idle_player.get_frame()
            if idle_frame is not None:
                canvas = overlay_top_right(canvas, idle_frame, cfg["video_size"])

        if cam_thread:
            cam_frame = cam_thread.get_frame()
            if cam_frame is not None:
                if last_detect and last_detect.get("frame_size"):
                    fw, fh = last_detect["frame_size"]
                    scale_x = cam_frame.shape[1] / max(1, fw)
                    scale_y = cam_frame.shape[0] / max(1, fh)
                    if last_detect.get("face"):
                        x, y, w, h = last_detect["face"]
                        x1 = int(x * scale_x)
                        y1 = int(y * scale_y)
                        x2 = int((x + w) * scale_x)
                        y2 = int((y + h) * scale_y)
                        cv.rectangle(cam_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if last_detect.get("mouth_poly"):
                        poly = np.array(last_detect["mouth_poly"], dtype=np.float32)
                        poly[:, 0, 0] *= scale_x
                        poly[:, 0, 1] *= scale_y
                        cv.polylines(cam_frame, [poly.astype(np.int32)], True, (0, 255, 0), 2)
                canvas = place_frame_bottom_left(canvas, cam_frame, cfg["ui_camera_resolution"])

        cv.imshow(window_name, canvas)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        if pending_active and not active_playing and active_videos:
            _, _, idle_idx = idle_player.get_frame()
            target = next_sync_frame(idle_idx, cfg["trigger_sync_frames"])
            if idle_idx >= target:
                chosen = np.random.choice(active_videos)
                active_player.set_source(chosen)
                active_playing = True
                pending_active = False

    sock.close()
    idle_player.stop()
    active_player.stop()
    if cam_thread:
        cam_thread.stop()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
