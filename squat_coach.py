#!/usr/bin/env python3
"""
Squat Coach — оценка техники приседаний по камере в реальном времени.

Установка:   pip install mediapipe opencv-python numpy pillow
Запуск:      python squat_coach.py                  (веб-камера 0)
             python squat_coach.py --source 1       (другая камера)
             python squat_coach.py --source squat.mp4 --save-video out.mp4

Как поставить камеру
  * Вид СБОКУ (камера на уровне пояса, 2–3 м): глубина, наклон корпуса, пятки.
  * Вид СПЕРЕДИ: глубина, колени (заваливание внутрь), перекос таза.
  * Человек виден целиком, от головы до стоп, хорошее освещение, облегающая одежда.

Клавиши:  q / Esc — выход,  r — сбросить счётчик,  m — зеркалить картинку.

ВАЖНО: программа оценивает технику по 2D-позе с одной камеры. Это подсказка,
а не врач и не тренер: округление спины и другие нюансы она не видит.
При боли в коленях/спине прекратите упражнение и обратитесь к специалисту.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------- точки MediaPipe Pose
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32
SIDE_PTS = {"L": (L_SH, L_HIP, L_KNEE, L_ANK, L_HEEL, L_TOE),
            "R": (R_SH, R_HIP, R_KNEE, R_ANK, R_HEEL, R_TOE)}

Landmarks = Dict[int, Tuple[float, float, float]]      # индекс -> (x_px, y_px, visibility)


# =====================================================================
#                      НАСТРОЙКИ И ТЕКСТЫ
# =====================================================================
@dataclass
class Config:
    vis_min: float = 0.5          # минимальная уверенность в точке
    stand_depth: float = -0.70    # бёдра заметно выше колен -> стоим
    start_depth: float = -0.55    # прошли это значение вниз -> начался спуск
    min_rep_depth: float = -0.45  # меньше — это просто наклон, не приседание
    good_depth: float = -0.05     # бёдра примерно на уровне коленей (параллель)
    ok_depth: float = -0.25       # «почти параллель»
    rise_margin: float = 0.12     # на сколько надо подняться, чтобы считать подъём
    zone_depth: float = -0.45     # ниже этого считаем, что идёт приседание (для замечаний)
    lean_max: float = 60.0        # наклон корпуса от вертикали, градусы
    valgus_max: float = 0.15      # колено внутрь от линии бедро–стопа, в длинах голени
    tilt_max: float = 0.12        # перекос таза, в длинах голени
    heel_max: float = 0.12        # подъём пятки, в длинах голени
    min_descent_s: float = 1.0    # быстрее — «слишком быстрый спуск»
    flag_time_s: float = 0.25     # сколько секунд ошибка должна длиться, чтобы попасть в итог
    hold_s: float = 0.6           # сколько держать подсказку на экране
    front_on: float = 0.35        # отношение ширина_таза/длина_корпуса: вид спереди
    front_off: float = 0.25       # ...и обратно к виду сбоку
    ema: float = 0.5              # сглаживание
    timeout_s: float = 25.0       # слишком долгое «повторение» — сброс


MSG = {
    "ru": {
        "shallow": "Недостаточная глубина — опускай бёдра до уровня коленей",
        "almost": "Почти параллель — присядь чуть глубже",
        "lean": "Корпус сильно наклонён — грудь вверх, смотри вперёд",
        "valgus": "Колени заваливаются внутрь — направляй их по линии носков",
        "heels": "Пятки отрываются от пола — держи вес на всей стопе",
        "tilt": "Перекос таза — приседай симметрично",
        "fast": "Слишком быстрый спуск — опускайся плавнее (около 2 с)",
        "perfect": "Отличная техника!",
        "no_pose": "Встаньте в кадр целиком (от головы до стоп)",
        "reps": "Повторения", "phase": "Фаза", "knee": "Колено", "torso": "Корпус",
        "stand": "стоя", "down": "вниз ↓", "up": "вверх ↑",
        "view_side": "вид сбоку", "view_front": "вид спереди",
        "rep": "Повтор", "avg": "Средняя оценка", "parallel": "параллель",
        "grade": ["Отлично", "Хорошо", "Есть над чем поработать", "Нужно исправить технику"],
    },
    "en": {
        "shallow": "Not deep enough - lower your hips to knee level",
        "almost": "Almost parallel - go a little deeper",
        "lean": "Torso leans too far - chest up, look ahead",
        "valgus": "Knees cave in - push them out over your toes",
        "heels": "Heels are lifting - keep your whole foot on the floor",
        "tilt": "Pelvis is tilted - squat symmetrically",
        "fast": "Descent is too fast - lower yourself in control (~2 s)",
        "perfect": "Great form!",
        "no_pose": "Step back so your whole body is in frame",
        "reps": "Reps", "phase": "Phase", "knee": "Knee", "torso": "Torso",
        "stand": "standing", "down": "down", "up": "up",
        "view_side": "side view", "view_front": "front view",
        "rep": "Rep", "avg": "Average score", "parallel": "parallel",
        "grade": ["Excellent", "Good", "Needs work", "Fix your form"],
    },
}
PENALTY = {"lean": 15, "valgus": 20, "heels": 10, "tilt": 10, "fast": 10}
ISSUE_ORDER = ["shallow", "almost", "valgus", "lean", "heels", "tilt", "fast"]


# =====================================================================
#                      ГЕОМЕТРИЯ
# =====================================================================
def _dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def angle_at(a, b, c) -> float:
    """Угол в точке b (градусы) между лучами b->a и b->c."""
    v1 = np.asarray(a, float) - np.asarray(b, float)
    v2 = np.asarray(c, float) - np.asarray(b, float)
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-6:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def grade_index(score: float) -> int:
    return 0 if score >= 90 else 1 if score >= 75 else 2 if score >= 55 else 3


# =====================================================================
#                      РЕЗУЛЬТАТЫ
# =====================================================================
@dataclass
class RepResult:
    index: int
    score: int
    depth: float                 # максимальная глубина (0 = бёдра на уровне коленей)
    knee_min: Optional[float]    # минимальный угол колена (вид сбоку)
    lean_max: Optional[float]
    valgus_max: Optional[float]
    t_down: float
    t_up: float
    view: str
    issues: List[str] = field(default_factory=list)


@dataclass
class FrameResult:
    has_pose: bool = False
    state: str = "stand"
    view: str = "side"
    side: str = "L"
    depth: float = -1.0
    knee_angle: Optional[float] = None
    lean: Optional[float] = None
    valgus: Optional[float] = None
    tilt: Optional[float] = None
    heel: Optional[float] = None
    flags: List[str] = field(default_factory=list)     # активные замечания для показа
    rep: Optional[RepResult] = None                    # завершённое в этом кадре повторение
    reps: int = 0
    max_depth: Optional[float] = None


# =====================================================================
#                      АНАЛИЗАТОР
# =====================================================================
class SquatAnalyzer:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.reset()

    def reset(self):
        self.reps: List[RepResult] = []
        self.state = "stand"
        self.view = "side"
        self._t_prev = None
        self._lost_since = None
        self._depth = None
        self._knee = None
        self._lean = None
        self._heel_base = None
        self._clear_rep()
        self._t_stand = 0.0

    def _clear_rep(self):
        self._max_depth = -9.0
        self._t_start = 0.0
        self._t_bottom = 0.0
        self._up_min = 9.0
        self._flag_time = {}
        self._hold = {}
        self._knee_min = None
        self._lean_max = None
        self._valgus_max = None

    # ---------------------------------------------------------------- вход
    def update(self, lm: Optional[Landmarks], t: float) -> FrameResult:
        cfg = self.cfg
        dt = 1 / 30 if self._t_prev is None else max(1e-3, min(0.5, t - self._t_prev))
        self._t_prev = t
        if lm is None:
            return self._lost(t)

        sv = {s: min(lm[i][2] for i in SIDE_PTS[s][:4]) for s in "LR"}
        best = max(sv, key=sv.get)
        if sv[best] < cfg.vis_min:
            return self._lost(t)
        self._lost_since = None

        P = lambda i: np.array(lm[i][:2], float)
        sh, hip, kn, an, heel, toe = (P(i) for i in SIDE_PTS[best])
        shank = _dist(kn, an)
        if shank < 10:                                  # человек слишком далеко
            return self._lost(t)

        # --- вид: сбоку или спереди
        torso = max(_dist((P(L_SH) + P(R_SH)) / 2, (P(L_HIP) + P(R_HIP)) / 2), 1.0)
        ratio = abs(P(L_HIP)[0] - P(R_HIP)[0]) / torso
        if self.view == "side" and ratio > cfg.front_on:
            self.view = "front"
        elif self.view == "front" and ratio < cfg.front_off:
            self.view = "side"
        front = self.view == "front"

        # --- глубина: (y_бедра − y_колена) / длина голени. 0 = бёдра на уровне коленей
        legs = ["L", "R"] if (front and min(sv.values()) >= cfg.vis_min) else [best]
        ds = []
        for s in legs:
            _, h, k, a, _, _ = (P(i) for i in SIDE_PTS[s])
            ds.append((h[1] - k[1]) / max(_dist(k, a), 1.0))
        d_raw = float(np.mean(ds))
        self._depth = d_raw if self._depth is None else cfg.ema * d_raw + (1 - cfg.ema) * self._depth
        depth = self._depth

        # --- угол колена
        k_raw = angle_at(hip, kn, an)
        self._knee = k_raw if self._knee is None else cfg.ema * k_raw + (1 - cfg.ema) * self._knee

        # --- наклон корпуса (только вид сбоку)
        lean = None
        if not front:
            if lm[best and SIDE_PTS[best][4]][2] >= cfg.vis_min and abs(toe[0] - heel[0]) > 3:
                fwd = 1.0 if toe[0] > heel[0] else -1.0
            else:
                fwd = 1.0 if kn[0] >= hip[0] else -1.0
            l_raw = math.degrees(math.atan2((sh[0] - hip[0]) * fwd, max(hip[1] - sh[1], 1e-6)
                                            if hip[1] > sh[1] else (hip[1] - sh[1])))
            self._lean = l_raw if self._lean is None else cfg.ema * l_raw + (1 - cfg.ema) * self._lean
            lean = self._lean

        # --- колени внутрь и перекос таза (вид спереди)
        valgus = tilt = None
        if front:
            mid_hip_x = (P(L_HIP)[0] + P(R_HIP)[0]) / 2
            vals = []
            for s in legs:
                _, h, k, a, _, _ = (P(i) for i in SIDE_PTS[s])
                if abs(a[1] - h[1]) < 1:
                    continue
                line_x = h[0] + (k[1] - h[1]) / (a[1] - h[1]) * (a[0] - h[0])
                inward = 1.0 if mid_hip_x >= h[0] else -1.0
                vals.append((k[0] - line_x) * inward / max(_dist(k, a), 1.0))
            valgus = max(vals) if vals else None
            tilt = abs(P(L_HIP)[1] - P(R_HIP)[1]) / shank

        # --- подъём пятки (вид сбоку; нужна калибровка стоя)
        heel_lift = None
        if not front and lm[SIDE_PTS[best][4]][2] >= cfg.vis_min and lm[SIDE_PTS[best][5]][2] >= cfg.vis_min:
            delta = heel[1] - toe[1]
            if self.state == "stand" and depth < cfg.stand_depth:
                self._heel_base = delta if self._heel_base is None else 0.9 * self._heel_base + 0.1 * delta
            elif self._heel_base is not None:
                heel_lift = (self._heel_base - delta) / shank

        self._step_state(depth, t, dt, front, lean, valgus, tilt, heel_lift)

        res = FrameResult(has_pose=True, state=self.state, view=self.view, side=best, depth=depth,
                          knee_angle=self._knee, lean=lean, valgus=valgus, tilt=tilt, heel=heel_lift,
                          reps=len(self.reps))
        res.rep = self._finished
        self._finished = None
        res.flags = self._active_flags(t)
        if self.state != "stand":
            res.max_depth = self._max_depth
        return res

    # ------------------------------------------------------- конечный автомат
    _finished: Optional[RepResult] = None

    def _begin_rep(self, t):
        self._clear_rep()
        self.state = "down"
        self._t_start = t

    def _step_state(self, d, t, dt, front, lean, valgus, tilt, heel):
        cfg = self.cfg
        self._finished = None
        if self.state == "stand":
            self._t_stand = t
            if d > cfg.start_depth:
                self._begin_rep(t)
        if self.state in ("down", "up"):
            if t - self._t_start > cfg.timeout_s:
                self.state = "stand"
                self._clear_rep()
                return
            self._collect(d, t, dt, front, lean, valgus, tilt, heel)
            if self.state == "down":
                if d >= self._max_depth:
                    self._max_depth, self._t_bottom = d, t
                if self._max_depth - d > cfg.rise_margin and self._max_depth > cfg.min_rep_depth:
                    self.state, self._up_min = "up", d
                elif d < cfg.stand_depth:                # вернулись, не присев
                    self.state = "stand"
                    self._clear_rep()
            elif self.state == "up":
                self._up_min = min(self._up_min, d)
                if d > self._max_depth:                  # снова пошли глубже — тот же повтор
                    self.state = "down"
                elif d < cfg.stand_depth:
                    self._finish_rep(t)

    def _collect(self, d, t, dt, front, lean, valgus, tilt, heel):
        cfg = self.cfg
        if self._knee is not None and not front:
            self._knee_min = self._knee if self._knee_min is None else min(self._knee_min, self._knee)
        if d <= cfg.zone_depth:
            return
        raw = set()
        if lean is not None:
            self._lean_max = lean if self._lean_max is None else max(self._lean_max, lean)
            if lean > cfg.lean_max:
                raw.add("lean")
        if valgus is not None:
            self._valgus_max = valgus if self._valgus_max is None else max(self._valgus_max, valgus)
            if valgus > cfg.valgus_max:
                raw.add("valgus")
        if tilt is not None and tilt > cfg.tilt_max:
            raw.add("tilt")
        if heel is not None and heel > cfg.heel_max:
            raw.add("heels")
        for k in raw:
            self._flag_time[k] = self._flag_time.get(k, 0.0) + dt
            self._hold[k] = t

    def _active_flags(self, t) -> List[str]:
        out = [k for k, tt in self._hold.items() if t - tt < self.cfg.hold_s]
        if self.state == "up":                           # подсказка про глубину во время подъёма
            if self._max_depth < self.cfg.ok_depth:
                out.append("shallow")
            elif self._max_depth < self.cfg.good_depth:
                out.append("almost")
        return sorted(set(out), key=ISSUE_ORDER.index)

    def _finish_rep(self, t):
        cfg = self.cfg
        md = self._max_depth
        issues = []
        score = 100.0
        if md < cfg.good_depth:
            score -= min(40.0, (cfg.good_depth - md) * 70.0)
            issues.append("shallow" if md < cfg.ok_depth else "almost")
        for k, tt in self._flag_time.items():
            if tt >= cfg.flag_time_s:
                issues.append(k)
                score -= PENALTY[k]
        t_down = self._t_bottom - self._t_stand
        if t_down < cfg.min_descent_s:
            issues.append("fast")
            score -= PENALTY["fast"]
        rep = RepResult(index=len(self.reps) + 1, score=int(round(max(0.0, score))), depth=round(md, 3),
                        knee_min=None if self._knee_min is None else round(self._knee_min, 1),
                        lean_max=None if self._lean_max is None else round(self._lean_max, 1),
                        valgus_max=None if self._valgus_max is None else round(self._valgus_max, 3),
                        t_down=round(t_down, 2), t_up=round(t - self._t_bottom, 2), view=self.view,
                        issues=sorted(issues, key=ISSUE_ORDER.index))
        self.reps.append(rep)
        self._finished = rep
        self.state = "stand"
        self._t_stand = t
        self._clear_rep()

    def _lost(self, t) -> FrameResult:
        if self._lost_since is None:
            self._lost_since = t
        if t - self._lost_since > 1.0 and self.state != "stand":
            self.state = "stand"
            self._clear_rep()
        return FrameResult(has_pose=False, state=self.state, view=self.view, reps=len(self.reps))

    # ------------------------------------------------------------ итоги
    def summary(self) -> dict:
        n = len(self.reps)
        return {"reps": n,
                "avg_score": round(sum(r.score for r in self.reps) / n, 1) if n else None,
                "details": [asdict(r) for r in self.reps]}


# =====================================================================
#                      ДЕТЕКТОР ПОЗЫ (MediaPipe)
# =====================================================================
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")
KEEP = [L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE]


def ensure_model(variant: str, path: Optional[str]) -> str:
    if path:
        return path
    cache = os.path.join(os.path.expanduser("~"), ".cache", "squat_coach")
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, "pose_landmarker_%s.task" % variant)
    if os.path.exists(f) and os.path.getsize(f) > 1_000_000:
        return f
    url = MODEL_URL.format(v=variant)
    print("Скачиваю модель позы (один раз, ~10–30 МБ):", url)
    try:
        urllib.request.urlretrieve(url, f + ".part")
        os.replace(f + ".part", f)
    except Exception as ex:
        sys.exit("Не удалось скачать модель: %s\nСкачайте файл вручную по ссылке\n  %s\n"
                 "и запустите с параметром  --model путь/к/файлу.task" % (ex, url))
    return f


class PoseBackend:
    """Обёртка над MediaPipe: поддерживает новый Tasks API и старый solutions."""

    def __init__(self, variant="full", model_path=None):
        import mediapipe as mp
        self.mp = mp
        self._last_ts = -1
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):      # старые версии
            self.kind = "solutions"
            self.pose = mp.solutions.pose.Pose(model_complexity={"lite": 0, "full": 1, "heavy": 2}[variant],
                                               smooth_landmarks=True, min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
        else:                                                               # новые версии
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
            opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=ensure_model(variant, model_path)),
                running_mode=vision.RunningMode.VIDEO, num_poses=1,
                min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.kind = "tasks"
            self.pose = vision.PoseLandmarker.create_from_options(opts)

    def process(self, bgr, t_ms: float) -> Optional[Landmarks]:
        import cv2
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.kind == "solutions":
            res = self.pose.process(rgb)
            pts = res.pose_landmarks.landmark if res.pose_landmarks else None
        else:
            ts = max(int(t_ms), self._last_ts + 1)          # метки времени должны строго расти
            self._last_ts = ts
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.pose.detect_for_video(img, ts)
            pts = res.pose_landmarks[0] if res.pose_landmarks else None
        if pts is None:
            return None
        return {i: (pts[i].x * w, pts[i].y * h, float(getattr(pts[i], "visibility", 1.0) or 0.0)) for i in KEEP}

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass


# =====================================================================
#                      ОТРИСОВКА
# =====================================================================
FONT_CANDIDATES = [
    "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "arial.ttf", "Arial.ttf", "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "NotoSans-Regular.ttf",
]
GREEN, RED, YELLOW, WHITE = (60, 200, 60), (50, 50, 230), (0, 200, 255), (255, 255, 255)   # BGR


def load_font(size: int):
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size), True
        except OSError:
            continue
    try:
        return ImageFont.load_default(size), False
    except TypeError:
        return ImageFont.load_default(), False


class Renderer:
    def __init__(self, lang: str, frame_h: int):
        self.size = max(14, frame_h // 26)
        self.font, cyr = load_font(self.size)
        self.small, _ = load_font(max(12, int(self.size * 0.8)))
        self.lang = lang if (cyr or lang == "en") else "en"
        self.t = MSG[self.lang]

    def draw(self, frame, lm: Optional[Landmarks], res: FrameResult, an: SquatAnalyzer):
        import cv2
        from PIL import Image, ImageDraw
        h, w = frame.shape[:2]
        texts = []                                              # (x, y, text, rgb, font)
        T = self.t
        rgb = lambda c: (c[2], c[1], c[0])

        # --- скелет
        if lm is not None and res.has_pose:
            bad = set(res.flags)
            legs = ("L", "R") if res.view == "front" else (res.side,)
            pt = lambda i: (int(lm[i][0]), int(lm[i][1]))
            for s in legs:
                sh, hip, kn, an_, heel, toe = SIDE_PTS[s]
                foot_bad = "heels" in bad
                segs = [(sh, hip, "lean" in bad), (hip, kn, False), (kn, an_, "valgus" in bad),
                        (an_, heel, foot_bad), (heel, toe, foot_bad), (an_, toe, foot_bad)]
                for a, b, is_bad in segs:
                    cv2.line(frame, pt(a), pt(b), RED if is_bad else GREEN, 4, cv2.LINE_AA)
                for i in (sh, hip, kn, an_):
                    cv2.circle(frame, pt(i), 7, RED if (i == kn and "valgus" in bad) else WHITE, -1, cv2.LINE_AA)
            if res.view == "front":
                cv2.line(frame, pt(L_SH), pt(R_SH), GREEN, 3, cv2.LINE_AA)
                cv2.line(frame, pt(L_HIP), pt(R_HIP), RED if "tilt" in bad else GREEN, 3, cv2.LINE_AA)
            if res.knee_angle is not None and res.view == "side":
                kx, ky = pt(SIDE_PTS[res.side][2])
                texts.append((kx + 12, ky - self.size, "%d°" % round(res.knee_angle), rgb(YELLOW), self.small))

        # --- панель статистики
        lines = ["%s: %d" % (T["reps"], res.reps),
                 "%s: %s  (%s)" % (T["phase"], T[res.state], T["view_front" if res.view == "front" else "view_side"])]
        if res.has_pose and res.view == "side" and res.lean is not None:
            lines.append("%s: %d°   %s: %d°" % (T["knee"], round(res.knee_angle or 0), T["torso"], round(res.lean)))
        if an.reps:
            lines.append("%s: %.0f" % (T["avg"], sum(r.score for r in an.reps) / len(an.reps)))
        ph = int(self.size * 1.35) * len(lines) + 12
        roi = frame[8:8 + ph, 8:8 + int(w * 0.46)]
        cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, roi)
        for n, ln in enumerate(lines):
            texts.append((16, 12 + int(n * self.size * 1.35), ln, (255, 255, 255), self.font if n == 0 else self.small))

        # --- шкала глубины справа
        if res.has_pose:
            x0, y0, y1 = w - 34, int(h * 0.15), int(h * 0.85)
            lo, hi = -1.1, 0.4
            ypos = lambda d: int(y0 + (np.clip(d, lo, hi) - lo) / (hi - lo) * (y1 - y0))
            cv2.rectangle(frame, (x0, y0), (x0 + 14, y1), (70, 70, 70), -1)
            cv2.rectangle(frame, (x0, ypos(res.depth)), (x0 + 14, y1) if False else (x0 + 14, ypos(res.depth) + 1), WHITE, -1)
            cv2.rectangle(frame, (x0, y0), (x0 + 14, ypos(res.depth)), (200, 160, 40), -1)
            cv2.line(frame, (x0 - 8, ypos(0.0)), (x0 + 22, ypos(0.0)), GREEN, 3)
            texts.append((x0 - int(self.size * 5.2), ypos(0.0) - int(self.size * 0.6), T["parallel"], rgb(GREEN), self.small))
            if res.max_depth is not None and res.max_depth > lo:
                cv2.line(frame, (x0 - 4, ypos(res.max_depth)), (x0 + 18, ypos(res.max_depth)), YELLOW, 2)

        # --- подсказки внизу
        y = h - 14
        out = []
        if not res.has_pose:
            out.append((T["no_pose"], YELLOW))
        elif res.flags:
            out.extend((T[k], RED if k not in ("almost",) else YELLOW) for k in res.flags)
        elif an.reps and res.state == "stand":
            r = an.reps[-1]
            col = GREEN if r.score >= 90 else YELLOW if r.score >= 55 else RED
            out.append(("%s %d: %d/100 — %s" % (T["rep"], r.index, r.score, T["grade"][grade_index(r.score)]), col))
            out.extend((T[k], YELLOW) for k in r.issues[:2])
            if not r.issues:
                out.append((T["perfect"], GREEN))
        for txt, col in reversed(out[:4]):
            bb = self.font.getbbox(txt)
            y -= int((bb[3] - bb[1]) * 1.6)
            texts.append((16, y, txt, rgb(col), self.font))

        # --- один проход PIL для всего текста (кириллица)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for x, yy, txt, col, font in texts:
            d.text((x, yy), txt, fill=col, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# =====================================================================
#                      ГЛАВНЫЙ ЦИКЛ
# =====================================================================
def run(source, backend, lang="ru", mirror=True, save_video=None, show=True, log=None,
        width=1280, height=720, max_frames=None, cfg=None) -> dict:
    import cv2
    is_cam = isinstance(source, int)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit("Не удалось открыть источник видео: %r" % (source,))
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    analyzer, renderer, writer = SquatAnalyzer(cfg), None, None
    t0, idx, last_new = time.perf_counter(), 0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_cam and mirror:
                frame = cv2.flip(frame, 1)
            t = (time.perf_counter() - t0) if is_cam else idx / fps_src
            if renderer is None:
                renderer = Renderer(lang, frame.shape[0])
            lm = backend.process(frame, t * 1000.0)
            res = analyzer.update(lm, t)
            if res.rep is not None:
                r = res.rep
                print("Повтор %d: %d/100  глубина %.2f  проблемы: %s" %
                      (r.index, r.score, r.depth, ", ".join(r.issues) or "нет"))
            frame = renderer.draw(frame, lm, res, analyzer)
            if save_video:
                if writer is None:
                    writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_src,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            if show:
                cv2.imshow("Squat Coach", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("r"):
                    analyzer.reset()
                if k == ord("m"):
                    mirror = not mirror
            idx += 1
            if max_frames and idx >= max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        backend.close()
    s = analyzer.summary()
    if log:
        with open(log, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    return s


def main():
    ap = argparse.ArgumentParser(description="Оценка техники приседаний по камере")
    ap.add_argument("--source", default="0", help="номер камеры или путь к видеофайлу")
    ap.add_argument("--lang", default="ru", choices=["ru", "en"])
    ap.add_argument("--model-size", default="full", choices=["lite", "full", "heavy"],
                    help="lite — быстрее, heavy — точнее")
    ap.add_argument("--model", default=None, help="путь к файлу .task (иначе скачается сам)")
    ap.add_argument("--no-mirror", action="store_true", help="не зеркалить изображение с камеры")
    ap.add_argument("--save-video", default=None, help="сохранить размеченное видео")
    ap.add_argument("--no-window", action="store_true", help="без окна (обработка файла)")
    ap.add_argument("--log", default=None, help="сохранить итоги тренировки в JSON")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    backend = PoseBackend(a.model_size, a.model)
    s = run(src, backend, a.lang, not a.no_mirror, a.save_video, not a.no_window, a.log, a.width, a.height)
    print("\nИтого повторений: %d, средняя оценка: %s" % (s["reps"], s["avg_score"]))


if __name__ == "__main__":
    main()
