import ctypes
import json
import queue
import random
import multiprocessing as mp
from pathlib import Path
from time import time, monotonic

import cv2 as cv
import dlib
import numpy as np

from imutils import face_utils
from numpy.linalg import norm


DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_url": "",
    "camera_capture_resolution": [1280, 720],
    "output_resolution": [0, 0],
    "camera_view_scale": 0.5,
    "camera_display_scale": 1.0,
    "video_size": [336, 672],
    "idle_smile_delay": 2.0,
    "smile_hold_seconds": 0.5,
    "smile_on_debounce_seconds": 0.15,
    "smile_off_debounce_seconds": 0.2,
    "trigger_sync_frames": [0, 150, 283, 395, 480, 649, 741, 817, 892, 1025, 1138, 1238],
    "lip_jaw_ratio_thresh": 0.44,
    "mouth_opening_ratio_thresh": 1.05,
    "processing_scale": 1.0,
    "face_center_weight": 1.0,
    "face_area_weight": 1.0,
    "detect_every_n": 1,
    "video_frame_skip": 0,
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
    cfg["output_resolution"] = tuple(int(x) for x in cfg["output_resolution"])
    cfg["camera_view_scale"] = float(cfg["camera_view_scale"])
    cfg["camera_display_scale"] = float(cfg["camera_display_scale"])
    cfg["video_size"] = tuple(int(x) for x in cfg["video_size"])
    cfg["idle_smile_delay"] = float(cfg["idle_smile_delay"])
    cfg["smile_hold_seconds"] = float(cfg["smile_hold_seconds"])
    cfg["smile_on_debounce_seconds"] = float(cfg["smile_on_debounce_seconds"])
    cfg["smile_off_debounce_seconds"] = float(cfg["smile_off_debounce_seconds"])
    cfg["trigger_sync_frames"] = [int(x) for x in cfg["trigger_sync_frames"]]
    cfg["lip_jaw_ratio_thresh"] = float(cfg["lip_jaw_ratio_thresh"])
    cfg["mouth_opening_ratio_thresh"] = float(cfg["mouth_opening_ratio_thresh"])
    cfg["processing_scale"] = float(cfg["processing_scale"])
    cfg["face_center_weight"] = float(cfg["face_center_weight"])
    cfg["face_area_weight"] = float(cfg["face_area_weight"])
    cfg["detect_every_n"] = int(cfg["detect_every_n"])
    cfg["video_frame_skip"] = int(cfg["video_frame_skip"])
    return cfg


def get_screen_size():
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

    def frame(self, skip=0):
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
        for _ in range(max(0, skip)):
            ret, _ = self.cap.read()
            if not ret:
                self._open()
                looped = True
                return None, -1, looped
            self.frame_idx = 0 if self.frame_idx < 0 else self.frame_idx + 1
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


def next_active_frame(active_frames, skip=0):
    for _ in range(max(0, skip)):
        try:
            next(active_frames)
        except StopIteration:
            return None
    try:
        return next(active_frames)
    except StopIteration:
        return None


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


def detector_process(input_q, output_q, cfg):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    lip_jaw_ratio_thresh = cfg["lip_jaw_ratio_thresh"]
    mouth_opening_ratio_thresh = cfg["mouth_opening_ratio_thresh"]
    face_center_weight = cfg["face_center_weight"]
    face_area_weight = cfg["face_area_weight"]

    while True:
        try:
            item = input_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break

        gray_image = item["gray"]
        crop_x = item["crop_x"]
        crop_y = item["crop_y"]
        local_scale = item["scale"]
        frame_idx = item["frame_idx"]

        faces = detector(gray_image, 1)
        best_face = None
        center = np.array([gray_image.shape[1] / 2.0, gray_image.shape[0] / 2.0])
        for face in faces:
            fx, fy = face.left(), face.top()
            fw, fh = face.right() - fx, face.bottom() - fy
            area = max(0, fw) * max(0, fh)
            face_center = np.array([fx + fw / 2.0, fy + fh / 2.0])
            dist = np.linalg.norm(face_center - center)
            dist_norm = dist / max(1.0, center[0])
            area_norm = area / max(1.0, gray_image.shape[0] * gray_image.shape[1])
            score = face_center_weight * dist_norm - face_area_weight * area_norm
            if best_face is None or score < best_face[0]:
                best_face = (score, face)
        faces = [best_face[1]] if best_face else []

        result = {"frame_idx": frame_idx, "smile": False, "face": None, "mouth_poly": None, "metrics": []}
        for face in faces:
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

            metrics = [
                f"lip/jaw: {lip_jaw_ratio:.3f} (>{lip_jaw_ratio_thresh})",
                f"mouth/nose: {mouth_opening / mouth_nose:.3f} (>={mouth_opening_ratio_thresh})",
                f"mouth/cheeks: {mouth_to_cheeks:.1f}",
                f"mouth/jaw: {mouth_to_jaw:.1f}",
            ]
            result = {
                "frame_idx": frame_idx,
                "smile": smile,
                "face": (x_face, y_face, w_face, h_face),
                "mouth_poly": mouth_poly,
                "metrics": metrics,
            }
            break

        output_q.put(result)


def main():
    cfg = load_config(Path("config.json"))
    # Variables for saving the images when a person on the video is smiling
    start = time()
    img_counter = 0  # Counter for the image names

    # Thresholds for smile detection
    lip_jaw_ratio_thresh = cfg["lip_jaw_ratio_thresh"]
    mouth_opening_ratio_thresh = cfg["mouth_opening_ratio_thresh"]

    # Video overlay settings
    camera_view_scale = cfg["camera_view_scale"]
    camera_display_scale = cfg["camera_display_scale"]
    video_size = cfg["video_size"]
    idle_smile_delay = cfg["idle_smile_delay"]
    smile_hold_seconds = cfg["smile_hold_seconds"]
    smile_on_debounce_seconds = cfg["smile_on_debounce_seconds"]
    smile_off_debounce_seconds = cfg["smile_off_debounce_seconds"]
    trigger_sync_frames = cfg["trigger_sync_frames"]
    processing_scale = cfg["processing_scale"]
    face_center_weight = cfg["face_center_weight"]
    face_area_weight = cfg["face_area_weight"]
    detect_every_n = max(1, cfg["detect_every_n"])
    video_frame_skip = max(0, cfg["video_frame_skip"])

    cam_source = cfg["camera_url"] if cfg["camera_url"] else cfg["camera_index"]
    cap = cv.VideoCapture(cam_source)
    # Ask camera for 8-bit frames where supported.
    cap.set(cv.CAP_PROP_CONVERT_RGB, 1)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    setup_camera_props(cap, cfg["camera_capture_resolution"])

    window_name = "Smile Detection"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    screen_w, screen_h = get_screen_size()
    out_w, out_h = cfg["output_resolution"]
    if out_w > 0 and out_h > 0:
        screen_w, screen_h = out_w, out_h
    idle_path = find_idle_video()
    idle = VideoLooper(idle_path) if idle_path else None
    active_videos = list_active_videos()
    active_frames = iter(())
    active_playing = False
    smile_started_at = None
    idle_start_at = monotonic()
    pending_active = False

    debounced_smiling = False
    raw_smile_started_at = None
    raw_not_smile_started_at = None

    ctx = mp.get_context("spawn")
    input_q = ctx.Queue(maxsize=1)
    output_q = ctx.Queue(maxsize=1)
    detector_cfg = {
        "lip_jaw_ratio_thresh": lip_jaw_ratio_thresh,
        "mouth_opening_ratio_thresh": mouth_opening_ratio_thresh,
        "face_center_weight": face_center_weight,
        "face_area_weight": face_area_weight,
    }
    worker = ctx.Process(target=detector_process, args=(input_q, output_q, detector_cfg), daemon=True)
    worker.start()

    last_result = {"smile": False, "face": None, "mouth_poly": None, "metrics": []}

    frame_idx = 0
    while True:
        ret, image = cap.read()
        if not ret or image is None:
            continue
        # Force 8-bit frame and use grayscale for dlib to avoid dtype/format issues.
        if image.dtype != np.uint8:
            image = cv.convertScaleAbs(image)
        image = np.ascontiguousarray(image, dtype=np.uint8)

        h, w = image.shape[:2]
        side = min(h, w)
        crop_x = (w - side) // 2
        crop_y = (h - side) // 2
        cropped = image[crop_y:crop_y + side, crop_x:crop_x + side]
        local_scale = processing_scale
        if local_scale <= 0 or local_scale > 1:
            local_scale = 1.0
        if local_scale != 1.0:
            scaled_side = max(1, int(side * local_scale))
            cropped = cv.resize(cropped, (scaled_side, scaled_side))
        gray_image = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        gray_image = np.ascontiguousarray(gray_image, dtype=np.uint8)

        if frame_idx % detect_every_n == 0:
            payload = {
                "gray": gray_image,
                "crop_x": crop_x,
                "crop_y": crop_y,
                "scale": local_scale,
                "frame_idx": frame_idx,
            }
            try:
                input_q.put_nowait(payload)
            except queue.Full:
                pass

        while True:
            try:
                result = output_q.get_nowait()
                last_result = result
            except queue.Empty:
                break
        result = last_result

        smile = result["smile"]
        if result["face"] and result["mouth_poly"] is not None:
            x_face, y_face, w_face, h_face = result["face"]
            cv.polylines(image, [result["mouth_poly"]], True, (0, 255, 0), 2)
            status_text = "Smiling!" if smile else "Not smiling"
            cv.putText(image, status_text, (x_face, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            for i, line in enumerate(result["metrics"]):
                y = 80 + i * 20
                cv.putText(image, line, (x_face, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        now = monotonic()
        if smile:
            raw_smile_started_at = raw_smile_started_at or now
            raw_not_smile_started_at = None
            if not debounced_smiling and (now - raw_smile_started_at) >= smile_on_debounce_seconds:
                debounced_smiling = True
                if time() - start > 3:
                    image_copy = image.copy()
                    cv.imwrite(f'smiling_images\\smile{img_counter}.jpg', image_copy)
                    start = time()
                    img_counter += 1
        else:
            raw_not_smile_started_at = raw_not_smile_started_at or now
            raw_smile_started_at = None
            if debounced_smiling and (now - raw_not_smile_started_at) >= smile_off_debounce_seconds:
                debounced_smiling = False

        display_frame = image
        if camera_display_scale < 1.0:
            ds = max(0.1, min(1.0, camera_display_scale))
            display_frame = cv.resize(
                display_frame, (int(display_frame.shape[1] * ds), int(display_frame.shape[0] * ds))
            )
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        canvas, _ = place_camera_bottom_left(canvas, display_frame, camera_view_scale)
        if active_playing:
            active_frame = next_active_frame(active_frames, video_frame_skip)
            if active_frame is None:
                active_playing = False
                idle_start_at = monotonic()
            else:
                canvas = overlay_top_right(canvas, active_frame, video_size)
        if not active_playing and idle:
            idle_frame, _, _ = idle.frame(video_frame_skip)
            if idle_frame is not None:
                canvas = overlay_top_right(canvas, idle_frame, video_size)

        indicator_color = (0, 255, 0) if debounced_smiling else (0, 0, 255)
        cv.rectangle(canvas, (20, 20), (80, 80), indicator_color, -1)
        cv.imshow(window_name, canvas)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if not active_playing and active_videos and idle and (now - idle_start_at) >= idle_smile_delay:
            if debounced_smiling:
                if smile_started_at is None:
                    smile_started_at = now
                elif now - smile_started_at >= smile_hold_seconds:
                    pending_active = True
                    smile_started_at = None
            else:
                smile_started_at = None
        else:
            smile_started_at = None

        if pending_active and not active_playing and idle:
            target = next_sync_frame(idle.frame_idx, trigger_sync_frames)
            if idle.frame_idx >= target:
                chosen = random.choice(active_videos)
                active_frames = play_video_once(chosen)
                active_playing = True
                pending_active = False

        frame_idx += 1

    cap.release()
    if idle:
        idle.release()
    try:
        input_q.put_nowait(None)
    except queue.Full:
        pass
    worker.join(timeout=1.0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
