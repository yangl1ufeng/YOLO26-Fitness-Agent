"""
YOLO26 居家健身姿态分析引擎
==============================
C成员第一阶段工作：
  1. 17个人体关键点 → 关节角度 + 时序特征
  2. 5个核心动作标准参数阈值
  3. 动作评分算法 (0-100, 三维度)
  4. 5类常见错误动作识别
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# ============================================================================
# 常量定义 — 与 YOLO26 COCO 17 关键点一致
# ============================================================================

KEYPOINT_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

# 骨架连线
SKELETON = [
    (5, 7), (7, 9),   # 左臂
    (6, 8), (8, 10),  # 右臂
    (5, 6),            # 肩连线
    (5, 11), (6, 12), # 躯干
    (11, 12),          # 髋连线
    (11, 13), (13, 15), # 左腿
    (12, 14), (14, 16), # 右腿
    (0, 1), (0, 2),   # 面部
    (1, 3), (2, 4),
]

# 左右侧关键点三元组: (近端, 关节, 远端)
SIDE_TRIPLETS = {
    "left": {
        "elbow":    (5, 7, 9),
        "knee":     (11, 13, 15),
        "hip":      (5, 11, 13),
        "shoulder": (7, 5, 11),
        "ankle":    (13, 15, None),  # 特殊: 膝-踝-垂直
    },
    "right": {
        "elbow":    (6, 8, 10),
        "knee":     (12, 14, 16),
        "hip":      (6, 12, 14),
        "shoulder": (8, 6, 12),
        "ankle":    (14, 16, None),
    },
}


# ============================================================================
# 基础几何运算
# ============================================================================

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """计算三点夹角 ∠ABC (B为顶点), 返回 0~180 度."""
    a, b, c = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), np.asarray(c, dtype=np.float32)
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    return 360.0 - angle if angle > 180.0 else angle


def calculate_vertical_angle(a: np.ndarray, b: np.ndarray) -> float:
    """计算向量 AB 与垂直向下方向 (0, 1) 的夹角, 返回 0~180 度."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    vec = b - a
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return 0.0
    cos_theta = vec[1] / norm  # 与垂直向下的点积
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def point_distance(a: np.ndarray, b: np.ndarray) -> float:
    """两点欧氏距离."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def point_to_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点 p 到线段 ab 的垂直距离."""
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    ab = b - a
    ap = p - a
    ab_norm = np.linalg.norm(ab)
    if ab_norm < 1e-6:
        return float(np.linalg.norm(ap))
    # 2D cross product: ab_x * ap_y - ab_y * ap_x
    cross = ab[0] * ap[1] - ab[1] * ap[0]
    return float(abs(cross) / ab_norm)


def valid_point(keypoints: np.ndarray, confidences: Optional[np.ndarray],
                idx: int, min_conf: float = 0.15) -> bool:
    """判断关键点是否有效."""
    if idx >= len(keypoints):
        return False
    x, y = keypoints[idx]
    if x <= 0 and y <= 0:
        return False
    if confidences is not None and confidences[idx] < min_conf:
        return False
    return True


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class JointAngles:
    """单帧所有关节角度."""
    knee_left: Optional[float] = None
    knee_right: Optional[float] = None
    hip_left: Optional[float] = None
    hip_right: Optional[float] = None
    elbow_left: Optional[float] = None
    elbow_right: Optional[float] = None
    shoulder_left: Optional[float] = None
    shoulder_right: Optional[float] = None
    trunk_angle: Optional[float] = None
    ankle_left: Optional[float] = None
    ankle_right: Optional[float] = None
    spread_state: Optional[float] = None  # 开合跳肢体展开度 0.0(闭合) ~ 1.0(展开)

    def mean_symmetric(self, attr: str) -> Optional[float]:
        """取左右侧均值."""
        l = getattr(self, f"{attr}_left")
        r = getattr(self, f"{attr}_right")
        vals = [v for v in (l, r) if v is not None]
        return float(np.mean(vals)) if vals else None

    def diff_symmetric(self, attr: str) -> Optional[float]:
        """取左右侧差值绝对值."""
        l = getattr(self, f"{attr}_left")
        r = getattr(self, f"{attr}_right")
        if l is not None and r is not None:
            return abs(l - r)
        return None

    def primary_angle(self, exercise: str) -> Optional[float]:
        """根据动作类型返回主角度."""
        primary_map = {
            "深蹲":       self.mean_symmetric("knee"),
            "俯卧撑":     self.mean_symmetric("elbow"),
            "平板支撑":   self.mean_symmetric("elbow"),
            "卷腹":       self.trunk_angle,
            "开合跳":     self.spread_state * 100.0 if self.spread_state is not None else None,  # 开合跳展开度缩放到 0-100°
            "引体向上":   self.mean_symmetric("elbow"),
            "臀桥":       self.mean_symmetric("hip"),
            "高抬腿":     self.mean_symmetric("hip"),
            "肩推":       self.mean_symmetric("elbow"),
            "侧平举":     self.mean_symmetric("shoulder"),
        }
        return primary_map.get(exercise)


@dataclass
class TemporalFeatures:
    """时序特征."""
    angular_velocity: float = 0.0       # 角速度 (°/s)
    smoothness: float = 0.0             # 平滑度 (jerk std, 越小越平滑)
    rhythm_consistency: float = 0.0     # 节奏一致性 (rep时长CV, 越小越一致)
    rom_consistency: float = 0.0        # 动作幅度一致性 (越小越一致)


@dataclass
class ExerciseStandard:
    """动作标准参数."""
    name: str                           # 动作名称
    primary_joint: str                  # 主监测关节
    target_low: float                   # 低位目标角度
    target_high: float                  # 高位目标角度
    low_range: Tuple[float, float]      # 低位有效范围 (min, max)
    high_range: Tuple[float, float]     # 高位有效范围 (min, max)
    count_trigger: str                  # "high" 或 "low"
    trunk_max: float                    # 躯干最大允许角度
    symmetry_joints: Tuple[str, ...]    # 需检查对称性的关节
    symmetry_max_diff: float            # 最大允许左右差异 (°)
    hold_threshold: Optional[float] = None  # 平板支撑等静态动作保持阈值


@dataclass
class ErrorInfo:
    """错误动作信息."""
    name: str                           # 错误名称
    severity: int                       # 严重程度 1-3
    message: str                        # 实时反馈消息
    suggestion: str                     # 修正建议


@dataclass
class ScoreResult:
    """评分结果."""
    total: float = 0.0                  # 总分 0-100
    angle_score: float = 0.0            # 关节角度得分 0-40
    temporal_score: float = 0.0         # 时序一致性得分 0-30
    symmetry_score: float = 0.0         # 对称性得分 0-30


@dataclass
class OverallRating:
    """总体评分报告 — 在运动结束后或阶段性输出.

    聚合整个运动过程的评分数据, 提供定性评级、趋势分析和改进建议.
    """

    # --- 评分等级常量 ---
    GRADE_EXCELLENT = "优秀"
    GRADE_GOOD = "良好"
    GRADE_AVERAGE = "一般"
    GRADE_NEEDS_IMPROVEMENT = "需改进"

    GRADE_THRESHOLDS = [
        (90, GRADE_EXCELLENT, "🌟", "动作标准，保持这个水准！"),
        (75, GRADE_GOOD, "👍", "整体不错，注意细节打磨"),
        (60, GRADE_AVERAGE, "📊", "基本完成，但需重点改进"),
        (0,  GRADE_NEEDS_IMPROVEMENT, "💪", "建议放慢节奏，关注姿势准确度"),
    ]

    TREND_IMPROVING = "进步中"
    TREND_STABLE = "稳定"
    TREND_DECLINING = "下滑中"

    # --- 字段 ---
    total_score: float = 0.0            # 0-100 加权总分
    grade: str = ""                     # 定性等级
    grade_emoji: str = ""               # 等级对应的 emoji
    grade_message: str = ""             # 等级对应的鼓励消息
    dimension_breakdown: str = ""       # 中文分维度解释
    trend: str = ""                     # 进步中/稳定/下滑中
    highlight: str = ""                 # 亮点（最好的维度）
    weakness: str = ""                  # 短板（最需要改进的维度）
    suggestion: str = ""                # 综合改进建议
    # 分维度均值
    avg_angle_score: float = 0.0
    avg_temporal_score: float = 0.0
    avg_symmetry_score: float = 0.0
    # 运动统计
    total_reps: int = 0
    total_duration_seconds: float = 0.0

    @classmethod
    def compute_grade(cls, total_score: float) -> tuple:
        """根据总分返回 (grade, emoji, message)."""
        for threshold, grade, emoji, msg in cls.GRADE_THRESHOLDS:
            if total_score >= threshold:
                return grade, emoji, msg
        return cls.GRADE_NEEDS_IMPROVEMENT, "💪", "建议放慢节奏，关注姿势准确度"

    @classmethod
    def compute_trend(cls, score_history: list, window: int = 5) -> str:
        """根据分数历史判断趋势.

        Args:
            score_history: 按时间排序的分数列表 (最近的在末尾).
            window: 对比窗口大小.

        Returns:
            TREND_IMPROVING | TREND_STABLE | TREND_DECLINING
        """
        if len(score_history) < window * 2:
            return cls.TREND_STABLE

        early_avg = sum(score_history[:window]) / window
        late_avg = sum(score_history[-window:]) / window
        diff = late_avg - early_avg

        if diff > 5:
            return cls.TREND_IMPROVING
        elif diff < -5:
            return cls.TREND_DECLINING
        else:
            return cls.TREND_STABLE


@dataclass
class AnalysisResult:
    """每帧分析结果."""
    angles: JointAngles = field(default_factory=JointAngles)
    temporal: TemporalFeatures = field(default_factory=TemporalFeatures)
    phase: str = "等待"
    count: int = 0
    hold_time: float = 0.0              # 平板支撑等动作的保持时间
    errors: list = field(default_factory=list)
    score: ScoreResult = field(default_factory=ScoreResult)
    overall: Optional[OverallRating] = None  # 运动结束后的总体评分报告


# ============================================================================
# 1. 关节角度提取器
# ============================================================================

class JointAngleExtractor:
    """从 YOLO26 输出的 17 个关键点提取全部关节角度."""

    def extract(self, keypoints: np.ndarray,
                confidences: Optional[np.ndarray] = None) -> JointAngles:
        angles = JointAngles()

        angles.knee_left   = self._joint_angle(keypoints, confidences, "knee", "left")
        angles.knee_right  = self._joint_angle(keypoints, confidences, "knee", "right")
        angles.hip_left    = self._joint_angle(keypoints, confidences, "hip", "left")
        angles.hip_right   = self._joint_angle(keypoints, confidences, "hip", "right")
        angles.elbow_left  = self._joint_angle(keypoints, confidences, "elbow", "left")
        angles.elbow_right = self._joint_angle(keypoints, confidences, "elbow", "right")
        angles.shoulder_left  = self._joint_angle(keypoints, confidences, "shoulder", "left")
        angles.shoulder_right = self._joint_angle(keypoints, confidences, "shoulder", "right")
        angles.ankle_left  = self._ankle_vertical_angle(keypoints, confidences, "left")
        angles.ankle_right = self._ankle_vertical_angle(keypoints, confidences, "right")
        angles.trunk_angle = self._trunk_vertical_angle(keypoints, confidences)
        angles.spread_state = self._compute_spread_state(keypoints, confidences, angles)

        return angles

    def _compute_spread_state(self, keypoints, confidences, angles: JointAngles) -> float:
        """计算开合跳肢体展开度 (0.0=闭合, 1.0=展开).

        使用像素距离而非关节角度 — 2D 摄像头中像素距离动态范围更大、噪声更低:
        - 手臂: 手腕相对肩膀的高度差 (像素) / 躯干高度 (像素)
        - 腿部: 双脚踝间距 / 肩宽
        """
        # 参考长度: 躯干高度 (肩中点到髋中点)
        torso_height = 100.0
        if (valid_point(keypoints, confidences, 5) and
                valid_point(keypoints, confidences, 6) and
                valid_point(keypoints, confidences, 11) and
                valid_point(keypoints, confidences, 12)):
            shoulder_mid_y = (keypoints[5][1] + keypoints[6][1]) / 2.0
            hip_mid_y = (keypoints[11][1] + keypoints[12][1]) / 2.0
            torso_height = max(abs(hip_mid_y - shoulder_mid_y), 50.0)

        # 1. 手臂展开: 手腕高于肩膀 → 正值, 手腕低于肩膀 → 负值
        arm_raises = []
        for wrist_id, shoulder_id in [(9, 5), (10, 6)]:
            if (valid_point(keypoints, confidences, wrist_id) and
                    valid_point(keypoints, confidences, shoulder_id)):
                # 手腕在肩上方时为正 (像素坐标系 y 向下)
                raise_px = float(keypoints[shoulder_id][1] - keypoints[wrist_id][1])
                arm_raises.append(raise_px / torso_height)

        if arm_raises:
            mean_raise = float(np.mean(arm_raises))
            # 典型范围: -0.8 (手臂下垂, 手腕在髋旁) ~ +1.0 (手臂举过头顶)
            arm_spread = max(0.0, min(1.0, (mean_raise + 0.8) / 1.8))
        else:
            arm_spread = 0.0

        # 2. 腿部展开: 踝距 / 肩宽
        leg_spread = 0.0
        if (valid_point(keypoints, confidences, 15) and
                valid_point(keypoints, confidences, 16) and
                valid_point(keypoints, confidences, 5) and
                valid_point(keypoints, confidences, 6)):
            ankle_dist = point_distance(keypoints[15], keypoints[16])
            shoulder_width = point_distance(keypoints[5], keypoints[6])
            if shoulder_width > 0:
                ratio = ankle_dist / shoulder_width
                # 典型范围: 0.7 (脚并拢) ~ 2.0 (脚大幅分开)
                leg_spread = max(0.0, min(1.0, (ratio - 0.7) / 1.3))

        # 综合: 手臂占 60%, 腿部占 40%
        return round(0.6 * arm_spread + 0.4 * leg_spread, 3)

    def _joint_angle(self, keypoints, confidences, joint_name: str,
                     side: str) -> Optional[float]:
        """通用关节角度计算."""
        ids = SIDE_TRIPLETS[side][joint_name]
        if joint_name == "ankle":
            return self._ankle_vertical_angle(keypoints, confidences, side)
        if all(valid_point(keypoints, confidences, i) for i in ids):
            return calculate_angle(keypoints[ids[0]], keypoints[ids[1]], keypoints[ids[2]])
        return None

    def _ankle_vertical_angle(self, keypoints, confidences, side: str) -> Optional[float]:
        """踝关节角度: 膝-踝连线与垂直线的夹角."""
        ids = SIDE_TRIPLETS[side]["ankle"]  # (knee, ankle, None)
        knee_idx, ankle_idx = ids[0], ids[1]
        if valid_point(keypoints, confidences, knee_idx) and valid_point(keypoints, confidences, ankle_idx):
            return calculate_vertical_angle(keypoints[knee_idx], keypoints[ankle_idx])
        return None

    def _trunk_vertical_angle(self, keypoints, confidences) -> Optional[float]:
        """躯干倾角: 肩中点→髋中点 与垂直线的夹角."""
        shoulder_ids = [5, 6]
        hip_ids = [11, 12]
        if all(valid_point(keypoints, confidences, i) for i in shoulder_ids + hip_ids):
            shoulder_mid = (keypoints[5] + keypoints[6]) / 2
            hip_mid = (keypoints[11] + keypoints[12]) / 2
            return calculate_vertical_angle(hip_mid, shoulder_mid)
        return None


# ============================================================================
# 2. 时序特征提取器
# ============================================================================

class TemporalFeatureExtractor:
    """滑动窗口时序特征提取.

    默认窗口 90 帧 (约 3 秒 @ 30fps).
    """

    def __init__(self, window_size: int = 90):
        self.window_size = window_size
        self.angle_history: deque = deque(maxlen=window_size)
        self.timestamp_history: deque = deque(maxlen=window_size)
        self.rep_durations: list = []  # 已完成 rep 的时长记录
        self.rep_peaks: list = []      # rep peak 值记录
        self._last_phase: str = "等待"
        self._rep_start_time: Optional[float] = None

    def update(self, angle_value: Optional[float], phase: str,
               timestamp: Optional[float] = None) -> TemporalFeatures:
        """添加一帧数据并返回当前时序特征."""
        if timestamp is None:
            timestamp = time.time()

        if angle_value is not None:
            self.angle_history.append(angle_value)
            self.timestamp_history.append(timestamp)
        else:
            self.angle_history.append(self.angle_history[-1] if self.angle_history else 0.0)
            self.timestamp_history.append(timestamp)

        # rep 计时
        self._track_rep(phase, timestamp)

        return TemporalFeatures(
            angular_velocity=self._calc_velocity(),
            smoothness=self._calc_smoothness(),
            rhythm_consistency=self._calc_rhythm_consistency(),
            rom_consistency=self._calc_rom_consistency(),
        )

    def _track_rep(self, phase: str, timestamp: float):
        """追踪 rep 开始/结束及运动幅度."""
        # rep 完成: 低位→高位
        if self._last_phase == "低位" and phase == "高位":
            if self._rep_start_time is not None:
                duration = timestamp - self._rep_start_time
                self.rep_durations.append(duration)
            self._rep_start_time = None
            # 记录本次 rep 的峰值 (angle_history 最大值)
            if len(self.angle_history) > 0:
                self.rep_peaks.append(max(self.angle_history))
        elif self._last_phase == "高位" and phase == "低位":
            self._rep_start_time = timestamp
        self._last_phase = phase

    def _calc_velocity(self) -> float:
        """角速度 (°/s): 最近两帧的变化率."""
        if len(self.angle_history) < 2 or len(self.timestamp_history) < 2:
            return 0.0
        da = self.angle_history[-1] - self.angle_history[-2]
        dt = max(self.timestamp_history[-1] - self.timestamp_history[-2], 1e-6)
        return abs(da / dt)

    def _calc_smoothness(self) -> float:
        """平滑度: 角加速度 (jerk) 的标准差, 越小越平滑."""
        if len(self.angle_history) < 3:
            return 0.0
        vel = np.diff(list(self.angle_history))
        if len(vel) < 2:
            return 0.0
        acc = np.diff(vel)
        return float(np.std(acc))

    def _calc_rhythm_consistency(self) -> float:
        """节奏一致性: rep 持续时间的变异系数 (CV), 0 表示完美一致."""
        if len(self.rep_durations) < 2:
            return 0.0
        durations = np.array(self.rep_durations[-10:])  # 最近 10 个 rep
        mean_d = np.mean(durations)
        if mean_d < 1e-6:
            return 0.0
        return float(np.std(durations) / mean_d)

    def _calc_rom_consistency(self) -> float:
        """动作幅度一致性: rep peak 值的变异系数."""
        if len(self.rep_peaks) < 2:
            return 0.0
        peaks = np.array(self.rep_peaks[-10:])
        mean_p = np.mean(peaks)
        if mean_p < 1e-6:
            return 0.0
        return float(np.std(peaks) / mean_p)

    def reset(self):
        self.angle_history.clear()
        self.timestamp_history.clear()
        self.rep_durations.clear()
        self.rep_peaks.clear()
        self._last_phase = "等待"
        self._rep_start_time = None


# ============================================================================
# 3. 五类核心动作标准参数
# ============================================================================

EXERCISE_STANDARDS: dict[str, ExerciseStandard] = {
    "深蹲": ExerciseStandard(
        name="深蹲",
        primary_joint="knee_angle",
        target_low=90.0,
        target_high=170.0,
        low_range=(70.0, 110.0),
        high_range=(155.0, 180.0),
        count_trigger="high",
        trunk_max=35.0,
        symmetry_joints=("knee",),
        symmetry_max_diff=25.0,
    ),
    "俯卧撑": ExerciseStandard(
        name="俯卧撑",
        primary_joint="elbow_angle",
        target_low=90.0,
        target_high=170.0,
        low_range=(70.0, 110.0),
        high_range=(155.0, 180.0),
        count_trigger="high",
        trunk_max=20.0,
        symmetry_joints=("elbow", "shoulder"),
        symmetry_max_diff=25.0,
    ),
    "平板支撑": ExerciseStandard(
        name="平板支撑",
        primary_joint="elbow_angle",
        target_low=90.0,
        target_high=90.0,
        low_range=(70.0, 110.0),
        high_range=(70.0, 110.0),
        count_trigger="high",       # 不用计数, 计时
        trunk_max=12.0,
        symmetry_joints=("elbow", "knee"),
        symmetry_max_diff=25.0,
        hold_threshold=90.0,
    ),
    "卷腹": ExerciseStandard(
        name="卷腹",
        primary_joint="trunk_angle",
        target_low=40.0,
        target_high=5.0,
        low_range=(25.0, 55.0),
        high_range=(0.0, 15.0),
        count_trigger="high",
        trunk_max=55.0,
        symmetry_joints=("shoulder",),
        symmetry_max_diff=25.0,
    ),
    "开合跳": ExerciseStandard(
        name="开合跳",
        primary_joint="spread_state",
        target_low=0.0,
        target_high=100.0,
        low_range=(-10.0, 65.0),    # 闭合: spread_angle ≤ 65° (超宽, 适配所有体型)
        high_range=(30.0, 110.0),   # 展开: spread_angle ≥ 30° (与闭合大重叠度, if-elif 保证不会误触发)
        count_trigger="high",
        trunk_max=25.0,
        symmetry_joints=("elbow", "knee"),
        symmetry_max_diff=25.0,
    ),
    "引体向上": ExerciseStandard(
        name="引体向上",
        primary_joint="elbow_angle",
        target_low=160.0,
        target_high=55.0,
        low_range=(140.0, 180.0),
        high_range=(35.0, 80.0),
        count_trigger="high",
        trunk_max=15.0,
        symmetry_joints=("elbow", "shoulder"),
        symmetry_max_diff=25.0,
    ),
    "臀桥": ExerciseStandard(
        name="臀桥",
        primary_joint="hip_angle",
        target_low=100.0,
        target_high=175.0,
        low_range=(80.0, 125.0),
        high_range=(165.0, 180.0),
        count_trigger="high",
        trunk_max=20.0,
        symmetry_joints=("knee", "hip"),
        symmetry_max_diff=25.0,
    ),
    "高抬腿": ExerciseStandard(
        name="高抬腿",
        primary_joint="hip_angle",
        target_low=170.0,
        target_high=95.0,
        low_range=(150.0, 180.0),
        high_range=(70.0, 115.0),
        count_trigger="high",
        trunk_max=15.0,
        symmetry_joints=("knee", "hip"),
        symmetry_max_diff=25.0,
    ),
    "肩推": ExerciseStandard(
        name="肩推",
        primary_joint="elbow_angle",
        target_low=70.0,
        target_high=170.0,
        low_range=(50.0, 90.0),
        high_range=(155.0, 180.0),
        count_trigger="high",
        trunk_max=15.0,
        symmetry_joints=("elbow", "shoulder"),
        symmetry_max_diff=25.0,
    ),
    "侧平举": ExerciseStandard(
        name="侧平举",
        primary_joint="shoulder_angle",
        target_low=10.0,
        target_high=90.0,
        low_range=(0.0, 30.0),
        high_range=(75.0, 105.0),
        count_trigger="high",
        trunk_max=12.0,
        symmetry_joints=("elbow", "shoulder"),
        symmetry_max_diff=25.0,
    ),
}


# ============================================================================
# 4. 动作评分算法
# ============================================================================

class MovementScorer:
    """三维度动作评分器 (0-100 分).

    - 关节角度得分: 0-40 分 (目标角度接近度)
    - 时序一致性得分: 0-30 分 (节奏 + 平滑度)
    - 对称性得分: 0-30 分 (左右平衡)

    时序平滑: EMA 平滑角度序列 + EMA 平滑各子项得分，减少单帧波动.
    相位感知: 使用 PoseAnalyzer 传入的实际相位选择目标角度，避免自推断偏差.
    """

    def __init__(self, exercise_name: str, smooth_alpha: float = 0.7):
        self.exercise_name = exercise_name
        self.standard = EXERCISE_STANDARDS.get(exercise_name)
        self.smooth_alpha = smooth_alpha  # EMA 平滑系数
        self.angle_tolerance = 12.5        # 角度高斯容差 (°), 可外部调参
        self._angle_samples: list = []       # 原始角度值
        self._smoothed_angles: list = []     # EMA 平滑后角度值 (保留兼容)
        self._angle_records: list = []       # [(smoothed_angle, target), ...] 每帧存自己的目标
        self._symmetry_diffs: dict[str, list] = {}  # 每帧各关节左右差值
        self._per_joint_history: dict[str, list[float]] = {}  # {joint_key: [angle, ...]} 逐关节最近 60 帧
        self._current_phase: str = "高位"     # 从 PoseAnalyzer 传入的实际相位

        # EMA 平滑后的得分缓存 (None = 尚未初始化)
        self._smooth_angle_score: Optional[float] = None
        self._smooth_temporal_score: Optional[float] = None
        self._smooth_symmetry_score: Optional[float] = None

        # 分数历史 (用于趋势分析和总体评分)
        self._score_history: list[float] = []     # 每帧 total 分数
        self._angle_score_history: list[float] = []
        self._temporal_score_history: list[float] = []
        self._symmetry_score_history: list[float] = []

    def update_angle(self, angle_value: Optional[float], phase: str):
        """记录一帧的角度（应用 EMA 平滑，存储相位用于目标选择）."""
        if angle_value is not None and phase != "等待":
            self._angle_samples.append(float(angle_value))
            self._current_phase = phase
            # EMA 平滑: smoothed = α * raw + (1-α) * prev_smoothed
            prev = self._smoothed_angles[-1] if self._smoothed_angles else float(angle_value)
            smoothed = self.smooth_alpha * float(angle_value) + (1 - self.smooth_alpha) * prev
            self._smoothed_angles.append(smoothed)

            # 存 (角度, 动态目标) 对 — 过渡期用实际角度作为目标避免误罚
            if self.standard:
                target = self._dynamic_target(phase, smoothed)
                self._angle_records.append((smoothed, target))
                # 只保留最近 60 帧 (约 2 秒)
                if len(self._angle_records) > 60:
                    self._angle_records.pop(0)

    def _dynamic_target(self, phase: str, smoothed_angle: float) -> float:
        """计算动态目标角度 — 过渡期用实际角度，避免移动中被误罚.

        当角度处于两个相位的有效范围之间 (过渡区), 直接用当前角度作为目标,
        意味着"正在移动中"不扣分. 只有在目标相位内才用标准目标衡量精度.
        """
        if not self.standard:
            return smoothed_angle

        low_min, low_max = self.standard.low_range
        high_min, high_max = self.standard.high_range

        if phase in ("低位", "保持"):
            # 在低位有效范围内 → 合标, 不罚
            if low_min <= smoothed_angle <= low_max:
                return smoothed_angle
            # 在过渡区 → 移动中, 不罚
            if low_max < smoothed_angle < high_min:
                return smoothed_angle
            # 超出范围 (太高或太低) → 用目标角度惩罚
            return self.standard.target_low
        else:
            # 在高位有效范围内 → 合标, 不罚
            if high_min <= smoothed_angle <= high_max:
                return smoothed_angle
            # 在过渡区 → 移动中, 不罚
            if low_max < smoothed_angle < high_min:
                return smoothed_angle
            # 超出范围 → 用目标角度惩罚
            return self.standard.target_high

    def update_symmetry(self, angles: JointAngles):
        """记录一帧的对称性数据 + 逐关节角度历史（供标准差计算）."""
        # --- 对称性差值 ---
        if self.standard is not None:
            for joint in self.standard.symmetry_joints:
                diff = angles.diff_symmetric(joint)
                if diff is not None:
                    if joint not in self._symmetry_diffs:
                        self._symmetry_diffs[joint] = []
                    self._symmetry_diffs[joint].append(diff)

        # --- 逐关节原始角度（供诊断层计算滑动窗口标准差）---
        _JOINT_ATTRS = [
            "knee_left", "knee_right", "hip_left", "hip_right",
            "elbow_left", "elbow_right", "shoulder_left", "shoulder_right",
            "ankle_left", "ankle_right",
        ]
        for attr in _JOINT_ATTRS:
            val = getattr(angles, attr, None)
            if val is not None:
                if attr not in self._per_joint_history:
                    self._per_joint_history[attr] = []
                self._per_joint_history[attr].append(float(val))
                # 只保留最近 60 帧
                if len(self._per_joint_history[attr]) > 60:
                    self._per_joint_history[attr].pop(0)

        # 躯干角度单独记录
        if angles.trunk_angle is not None:
            if "trunk" not in self._per_joint_history:
                self._per_joint_history["trunk"] = []
            self._per_joint_history["trunk"].append(float(angles.trunk_angle))
            if len(self._per_joint_history["trunk"]) > 60:
                self._per_joint_history["trunk"].pop(0)

    # ---- 诊断数据暴露 (供 DiagnosticContextBuilder 使用) ----

    @property
    def angle_records(self) -> list:
        """返回 (smoothed_angle, target) 记录列表（拷贝，最多 60 条）."""
        return list(self._angle_records)

    @property
    def symmetry_diffs(self) -> dict:
        """返回各关节左右差值记录（拷贝）."""
        return {k: list(v) for k, v in self._symmetry_diffs.items()}

    @property
    def per_joint_history(self) -> dict[str, list[float]]:
        """返回逐关节角度历史（拷贝），供标准差计算."""
        return {k: list(v) for k, v in self._per_joint_history.items()}

    @property
    def score_history(self) -> dict:
        """返回分维度历史（拷贝，最多 60 条）."""
        return {
            "total": list(self._score_history),
            "angle": list(self._angle_score_history),
            "temporal": list(self._temporal_score_history),
            "symmetry": list(self._symmetry_score_history),
        }

    def get_diagnostic_data(self) -> dict:
        """返回诊断所需的所有内部数据，供 DiagnosticContextBuilder 使用."""
        return {
            "_standard": self.standard,
            "angle_records": self.angle_records,
            "symmetry_diffs": self.symmetry_diffs,
            "per_joint_history": self.per_joint_history,
            "score_history": self.score_history,
            "angle_tolerance": self.angle_tolerance,
            "smooth_alpha": self.smooth_alpha,
            "current_phase": self._current_phase,
            "smooth_scores": {
                "angle": self._smooth_angle_score,
                "temporal": self._smooth_temporal_score,
                "symmetry": self._smooth_symmetry_score,
            },
        }

    def compute(self, temporal: TemporalFeatures) -> ScoreResult:
        """计算最终评分 (含帧间 EMA 平滑，减少单帧误差)."""
        angle_score = self._score_angle()
        temporal_score = self._score_temporal(temporal)
        symmetry_score = self._score_symmetry()

        # EMA 平滑各子项得分，避免帧间剧烈跳动
        alpha = 0.6  # 得分平滑系数
        if self._smooth_angle_score is None:
            self._smooth_angle_score = angle_score
            self._smooth_temporal_score = temporal_score
            self._smooth_symmetry_score = symmetry_score
        else:
            self._smooth_angle_score = alpha * angle_score + (1 - alpha) * self._smooth_angle_score
            self._smooth_temporal_score = alpha * temporal_score + (1 - alpha) * self._smooth_temporal_score
            self._smooth_symmetry_score = alpha * symmetry_score + (1 - alpha) * self._smooth_symmetry_score

        total = self._smooth_angle_score + self._smooth_temporal_score + self._smooth_symmetry_score
        total = round(min(total, 100.0), 1)

        # 记录分数历史 (用于总体评分和趋势分析)
        self._score_history.append(total)
        self._angle_score_history.append(round(self._smooth_angle_score, 1))
        self._temporal_score_history.append(round(self._smooth_temporal_score, 1))
        self._symmetry_score_history.append(round(self._smooth_symmetry_score, 1))

        return ScoreResult(
            total=total,
            angle_score=round(self._smooth_angle_score, 1),
            temporal_score=round(self._smooth_temporal_score, 1),
            symmetry_score=round(self._smooth_symmetry_score, 1),
        )

    def _score_angle(self) -> float:
        """关节角度得分 (0-40).

        高斯衰减: score = 40 * exp(-(mean_dev/tolerance)²)
        使用每帧记录时对应的目标角度计算偏差，避免相位切换时
        旧帧角度被新目标误判（如站立的170°被下蹲目标90°判为偏差80°）.
        """
        if not self.standard or not self._angle_records:
            return 0.0

        tolerance = self.angle_tolerance

        # 取最近 30 条 (angle, target) 记录 (~1 秒)
        recent = self._angle_records[-30:]

        # 每条记录用自己存入时的 target 算偏差
        deviations = [abs(angle - target) for angle, target in recent]
        mean_dev = float(np.mean(deviations))

        return 40.0 * math.exp(-((mean_dev / tolerance) ** 2))

    def _score_temporal(self, temporal: TemporalFeatures) -> float:
        """时序一致性得分 (0-30).

        - 节奏稳定性: 15分, CV < 35% 得满分. CV=0 且无 rep 数据时不给满分.
        - 动作平滑度: 15分, jerk 线性映射.
        """
        # CV=0 可能是无数据, 取不低于 0.015 避免未运动就得满分
        effective_cv = max(temporal.rhythm_consistency, 0.015)
        rhythm_score = 15.0 * max(0.0, 1.0 - effective_cv / 0.35)
        smooth_score = 15.0 * max(0.0, 1.0 - temporal.smoothness / 50.0)
        return rhythm_score + smooth_score

    def _score_symmetry(self) -> float:
        """对称性得分 (0-30).

        每个关注关节: 左右差异 < max_diff 得满分, 线性衰减.
        无数据时返回中性分 15，不盲目给满分.
        """
        if not self.standard or not self._symmetry_diffs:
            return 15.0  # 无数据返回中性分

        scores = []
        for joint in self.standard.symmetry_joints:
            diffs = self._symmetry_diffs.get(joint, [])
            if not diffs:
                scores.append(1.0)
                continue
            mean_diff = float(np.mean(diffs))
            max_allowed = self.standard.symmetry_max_diff
            # 线性衰减: diff=0 → 1.0, diff=max_allowed → 0.0
            joint_score = max(0.0, 1.0 - mean_diff / max_allowed)
            scores.append(joint_score)

        return 30.0 * float(np.mean(scores))

    def get_overall_rating(self, total_reps: int = 0,
                           duration_seconds: float = 0.0) -> OverallRating:
        """聚合整个运动过程的评分数据, 生成总体评分报告.

        Args:
            total_reps: 总重复次数.
            duration_seconds: 运动总时长 (秒).

        Returns:
            OverallRating: 包含定性评级、趋势分析和改进建议的总体报告.
        """
        if not self._score_history:
            # 无数据时返回默认报告
            grade, emoji, msg = OverallRating.compute_grade(0.0)
            return OverallRating(
                total_score=0.0,
                grade=grade, grade_emoji=emoji, grade_message=msg,
                dimension_breakdown="暂无评分数据",
                trend=OverallRating.TREND_STABLE,
                highlight="无", weakness="无",
                suggestion="开始运动以获取评分反馈",
                total_reps=total_reps,
                total_duration_seconds=duration_seconds,
            )

        # --- 1. 加权总分 (取最近 30 帧的均值, 反映当前状态) ---
        recent = self._score_history[-30:]
        total_score = round(float(np.mean(recent)), 1)

        # --- 2. 定性等级 ---
        grade, emoji, msg = OverallRating.compute_grade(total_score)

        # --- 3. 分维度均值 (归一化到百分制便于比较) ---
        avg_angle = round(float(np.mean(self._angle_score_history[-30:])), 1)
        avg_temporal = round(float(np.mean(self._temporal_score_history[-30:])), 1)
        avg_symmetry = round(float(np.mean(self._symmetry_score_history[-30:])), 1)

        # 归一化到 0-100
        angle_pct = avg_angle / 40.0 * 100
        temporal_pct = avg_temporal / 30.0 * 100
        symmetry_pct = avg_symmetry / 30.0 * 100

        dims = [
            ("关节角度", angle_pct, avg_angle, 40.0),
            ("时序节奏", temporal_pct, avg_temporal, 30.0),
            ("左右对称", symmetry_pct, avg_symmetry, 30.0),
        ]
        dims.sort(key=lambda x: x[1], reverse=True)

        # --- 4. 亮点 & 短板 & 分维度解释 ---
        best_name, best_pct, best_raw, best_max = dims[0]
        worst_name, worst_pct, worst_raw, worst_max = dims[2]

        highlight = f"{best_name} ({best_raw:.0f}/{best_max:.0f})"
        weakness = f"{worst_name} ({worst_raw:.0f}/{worst_max:.0f})"

        breakdown_parts = []
        for name, pct, raw, max_val in dims:
            if pct >= 85:
                level = "优秀"
            elif pct >= 70:
                level = "良好"
            elif pct >= 50:
                level = "一般"
            else:
                level = "需改进"
            breakdown_parts.append(f"{name}: {raw:.0f}/{max_val:.0f}（{level}）")
        dimension_breakdown = "；".join(breakdown_parts)

        # --- 5. 趋势 ---
        trend = OverallRating.compute_trend(self._score_history)

        # --- 6. 综合改进建议 ---
        suggestion = self._generate_suggestion(worst_name, total_score, grade)

        return OverallRating(
            total_score=total_score,
            grade=grade, grade_emoji=emoji, grade_message=msg,
            dimension_breakdown=dimension_breakdown,
            trend=trend,
            highlight=highlight, weakness=weakness,
            suggestion=suggestion,
            avg_angle_score=avg_angle,
            avg_temporal_score=avg_temporal,
            avg_symmetry_score=avg_symmetry,
            total_reps=total_reps,
            total_duration_seconds=round(duration_seconds, 1),
        )

    def _generate_suggestion(self, worst_dim: str, total_score: float,
                             grade: str) -> str:
        """根据短板维度和总分生成改进建议."""
        dim_suggestions = {
            "关节角度": "注意控制动作幅度，确保每次动作都做到位。"
                       "深蹲时大腿与地面平行，俯卧撑时胸部贴近地面。",
            "时序节奏": "尝试保持均匀的动作节奏，建议下放2秒、发力1秒。"
                       "避免借助惯性完成动作。",
            "左右对称": "注意左右两侧均衡发力，可以对着镜子检查身体是否歪斜。"
                       "弱侧可以先做单侧训练来弥补差距。",
        }

        base = dim_suggestions.get(worst_dim, "建议关注动作质量，持续练习。")

        if grade == OverallRating.GRADE_EXCELLENT:
            return f"表现优异！继续保持当前水准。{base}"
        elif grade == OverallRating.GRADE_GOOD:
            return f"整体不错。{base}"
        elif grade == OverallRating.GRADE_AVERAGE:
            return f"还有提升空间，建议重点改善{worst_dim}。{base}"
        else:
            return f"建议先放慢节奏，专注动作质量而非数量。{base}"

    def reset(self):
        self._angle_samples.clear()
        self._smoothed_angles.clear()
        self._angle_records.clear()
        self._symmetry_diffs.clear()
        self._per_joint_history.clear()
        self._smooth_angle_score = None
        self._smooth_temporal_score = None
        self._smooth_symmetry_score = None
        self._score_history.clear()
        self._angle_score_history.clear()
        self._temporal_score_history.clear()
        self._symmetry_score_history.clear()
        self._current_phase = "高位"


# ============================================================================
# 5. 常见错误动作识别
# ============================================================================

class ErrorDetector:
    """五类常见错误动作检测器."""

    # 连续帧阈值: 避免误报
    CONSECUTIVE_FRAMES = 5

    def __init__(self):
        self._error_counter: dict[str, int] = {}  # 错误名 → 连续帧计数

    def detect(self, angles: JointAngles, keypoints: np.ndarray,
               confidences: Optional[np.ndarray], phase: str,
               exercise: str) -> list[ErrorInfo]:
        """检测所有适用错误, 返回当前活跃的错误列表."""
        errors = []
        methods = {
            "深蹲":     [self._detect_knee_valgus, self._detect_back_rounding],
            "俯卧撑":   [self._detect_hip_sagging_pushup, self._detect_elbow_flare],
            "平板支撑": [self._detect_hip_sagging_plank],
            "卷腹":     [self._detect_neck_strain],
            "开合跳":   [self._detect_incomplete_spread],
            "引体向上": [self._detect_pullup_swing, self._detect_elbow_flare],
            "臀桥":     [self._detect_bridge_asymmetry, self._detect_hip_sagging_pushup],
            "高抬腿":   [self._detect_high_knee_lean, self._detect_knee_valgus],
            "肩推":     [self._detect_shoulder_press_arch, self._detect_elbow_flare],
            "侧平举":   [self._detect_lateral_raise_swing, self._detect_elbow_flare],
        }

        for detector in methods.get(exercise, []):
            error = detector(angles, keypoints, confidences, phase)
            if error:
                key = detector.__name__
                self._error_counter[key] = self._error_counter.get(key, 0) + 1
                if self._error_counter[key] >= self.CONSECUTIVE_FRAMES:
                    errors.append(error)
            else:
                self._error_counter.pop(detector.__name__, None)

        return errors

    # --- 错误 1: 深蹲膝盖内扣 ---
    def _detect_knee_valgus(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """膝关节水平偏移超过踝间距 15% 判定为膝盖内扣."""
        if phase != "低位":
            return None
        for side, ids in [("left", (11, 13, 15)), ("right", (12, 14, 16))]:
            hip_i, knee_i, ankle_i = ids
            if not all(valid_point(kp, conf, i) for i in ids):
                continue
            ankle_dist = point_distance(kp[ankle_i], kp[hip_i])
            # 膝的水平偏移 (相对髋-踝连线中点)
            knee_offset = point_to_line_distance(kp[knee_i], kp[hip_i], kp[ankle_i])
            if ankle_dist > 1 and knee_offset / ankle_dist > 0.15:
                side_cn = "左膝" if side == "left" else "右膝"
                return ErrorInfo(
                    name="膝盖内扣",
                    severity=2,
                    message=f"检测到{side_cn}内扣",
                    suggestion="保持膝盖与脚尖方向一致，有意识地将膝盖向外打开",
                )
        return None

    # --- 错误 2: 俯卧撑塌腰 ---
    def _detect_hip_sagging_pushup(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """髋点偏离肩-踝连线超过体长 8%."""
        if phase != "低位":
            return None
        return self._check_hip_sag(kp, conf, ratio=0.08, name="俯卧撑塌腰",
                                   suggestion="收紧核心和臀部，保持身体呈一条直线")

    # --- 错误 3: 深蹲弓背 ---
    def _detect_back_rounding(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """躯干前倾 > 45°（即躯干角 < 135°）。不限阶段 — 不良体态在任何阶段都应被检测。"""
        if angles.trunk_angle is not None and angles.trunk_angle < 135.0:
            return ErrorInfo(
                name="深蹲弓背",
                severity=2,
                message=f"躯干前倾 {angles.trunk_angle:.0f}°，疑似弓背",
                suggestion="挺胸收腹，保持背部直立，目视前方",
            )
        return None

    # --- 错误 4: 平板支撑塌腰 ---
    def _detect_hip_sagging_plank(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """髋点偏离肩-踝连线超过体长 6%."""
        return self._check_hip_sag(kp, conf, ratio=0.06, name="平板支撑塌腰",
                                   suggestion="收紧腹部和臀部，避免髋部下垂或上抬")

    # --- 错误 5: 卷腹颈部用力 ---
    def _detect_neck_strain(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """鼻-肩距离变化与躯干角变化比率 > 0.5，说明过度收下巴."""
        if phase != "低位":
            return None
        # 检查鼻(0)到左肩(5)和右肩(6)中点距离 vs 躯干角
        if valid_point(kp, conf, 0) and valid_point(kp, conf, 5) and valid_point(kp, conf, 6):
            shoulder_mid = (kp[5] + kp[6]) / 2
            nose_to_shoulder = point_distance(kp[0], shoulder_mid)
            # 经验阈值: 正常卷腹鼻-肩距离变化有限
            if nose_to_shoulder < 15.0:  # 像素阈值, 太近说明收下巴
                return ErrorInfo(
                    name="卷腹颈部用力",
                    severity=1,
                    message="检测到颈部过度用力",
                    suggestion="双手轻扶耳侧，下巴微收保持一拳距离，用腹部发力而非颈部",
                )
        return None

    # --- 辅助错误: 俯卧撑肘部过度外展 ---
    def _detect_elbow_flare(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """肩关节角度 > 120° 或 < 50° 表示肘部位置不当."""
        if phase != "低位":
            return None
        sh_l = angles.shoulder_left
        sh_r = angles.shoulder_right
        if sh_l is not None and (sh_l > 120.0 or sh_l < 50.0):
            return ErrorInfo(
                name="肘部外展",
                severity=1,
                message="检测到肘部位置不当",
                suggestion="肘部与身体保持约45°夹角，避免过度外展",
            )
        if sh_r is not None and (sh_r > 120.0 or sh_r < 50.0):
            return ErrorInfo(
                name="肘部外展",
                severity=1,
                message="检测到肘部位置不当",
                suggestion="肘部与身体保持约45°夹角，避免过度外展",
            )
        return None

    # --- 辅助错误: 开合跳不完整 ---
    def _detect_incomplete_spread(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """检测开合跳时手脚是否充分展开."""
        if phase != "高位":
            return None
        # 检查手腕是否过肩
        if valid_point(kp, conf, 9) and valid_point(kp, conf, 5):
            if kp[9][1] > kp[5][1]:  # 手腕在肩下方
                return ErrorInfo(
                    name="手臂未充分展开",
                    severity=1,
                    message="开合跳时手臂未举过头顶",
                    suggestion="跳起时手臂充分向上伸展过头顶",
                )
        return None

    def _check_hip_sag(self, kp, conf, ratio: float,
                       name: str, suggestion: str) -> Optional[ErrorInfo]:
        """通用髋部下塌检测."""
        required = [5, 6, 11, 12, 15, 16]
        if not all(valid_point(kp, conf, i) for i in required):
            return None
        shoulder_mid = (kp[5] + kp[6]) / 2
        hip_mid = (kp[11] + kp[12]) / 2
        ankle_mid = (kp[15] + kp[16]) / 2
        body_length = point_distance(shoulder_mid, ankle_mid)
        if body_length < 1:
            return None
        hip_deviation = point_to_line_distance(hip_mid, shoulder_mid, ankle_mid)
        if hip_deviation / body_length > ratio:
            return ErrorInfo(name=name, severity=2,
                             message=f"检测到髋部下塌 (偏离 {hip_deviation/body_length*100:.0f}%)",
                             suggestion=suggestion)
        return None

    # --- 辅助错误: 侧平举身体晃动 ---
    def _detect_lateral_raise_swing(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """侧平举时躯干倾角 > 15° 表示身体借力晃动."""
        if phase != "高位":
            return None
        if angles.trunk_angle is not None and angles.trunk_angle < 165.0:
            return ErrorInfo(
                name="身体晃动借力",
                severity=1,
                message=f"躯干倾斜 {angles.trunk_angle:.0f}°，疑似借力",
                suggestion="保持躯干稳定直立，仅用肩部发力完成侧平举",
            )
        return None

    # --- 辅助错误: 引体向上摆动 ---
    def _detect_pullup_swing(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """引体向上时躯干倾角 > 12° 表示身体摆动借力."""
        if angles.trunk_angle is not None and angles.trunk_angle < 168.0:
            return ErrorInfo(
                name="身体摆动",
                severity=2,
                message=f"躯干倾斜 {angles.trunk_angle:.0f}°，疑似摆动借力",
                suggestion="收紧核心，控制身体稳定，避免借助惯性摆动",
            )
        return None

    # --- 辅助错误: 臀桥不对称 ---
    def _detect_bridge_asymmetry(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """臀桥高位时左右髋角差异 > 10° 表示发力不对称."""
        if phase != "高位":
            return None
        hip_l = angles.hip_left
        hip_r = angles.hip_right
        if hip_l is not None and hip_r is not None:
            diff = abs(hip_l - hip_r)
            if diff > 10.0:
                return ErrorInfo(
                    name="臀桥不对称",
                    severity=1,
                    message=f"左右髋角相差 {diff:.0f}°",
                    suggestion="均匀发力，确保双侧臀部同时抬起",
                )
        return None

    # --- 辅助错误: 高抬腿身体后仰 ---
    def _detect_high_knee_lean(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """高抬腿时躯干倾角 > 18° 表示身体后仰."""
        if angles.trunk_angle is not None and angles.trunk_angle < 162.0:
            return ErrorInfo(
                name="身体后仰",
                severity=2,
                message=f"躯干倾斜 {angles.trunk_angle:.0f}°，疑似后仰",
                suggestion="保持上身挺直微前倾，核心收紧，目视前方",
            )
        return None

    # --- 辅助错误: 肩推弓背 ---
    def _detect_shoulder_press_arch(self, angles: JointAngles, kp, conf, phase) -> Optional[ErrorInfo]:
        """肩推时躯干倾角 > 15° 表示过度弓背."""
        if angles.trunk_angle is not None and angles.trunk_angle < 165.0:
            return ErrorInfo(
                name="肩推弓背",
                severity=2,
                message=f"躯干倾斜 {angles.trunk_angle:.0f}°，疑似弓背借力",
                suggestion="收紧核心，保持背部直立，避免过度后仰借力",
            )
        return None

    def reset(self):
        self._error_counter.clear()


# ============================================================================
# 6. PoseAnalyzer 主类
# ============================================================================

class PoseAnalyzer:
    """姿态分析器 — 对外统一接口.

    用法:
        analyzer = PoseAnalyzer("深蹲")
        result = analyzer.analyze_frame(keypoints, confidences)
        score = analyzer.get_score()
        errors = analyzer.get_errors()
    """

    def __init__(self, exercise_name: str):
        if exercise_name not in EXERCISE_STANDARDS:
            raise ValueError(f"不支持的动作: {exercise_name}. "
                             f"支持: {list(EXERCISE_STANDARDS.keys())}")
        self.exercise_name = exercise_name
        self.standard = EXERCISE_STANDARDS[exercise_name]

        self._angle_extractor = JointAngleExtractor()
        self._temporal_extractor = TemporalFeatureExtractor()
        self._scorer = MovementScorer(exercise_name)
        self._error_detector = ErrorDetector()

        # 计数状态机
        self.count = 0
        self.phase = "等待"
        self._hold_start: Optional[float] = None
        self.hold_time = 0.0

        # 运动时长追踪
        self._session_start_time: Optional[float] = None  # 首次进入运动相位的时间
        self._session_active: bool = False

        # 主角度 EMA 预平滑 (用于时序特征提取, 降低快速动作的 jerk 噪声)
        self._smooth_primary_val: Optional[float] = None

    @property
    def scorer(self) -> MovementScorer:
        """公开 MovementScorer 实例，供 DiagnosticContextBuilder 读取诊断数据."""
        return self._scorer

    def apply_tuning(self, **kwargs):
        """运行时调参 — 同步更新 PoseAnalyzer.standard 和 MovementScorer.standard.

        由于两个类各自从 EXERCISE_STANDARDS 查表得到独立的 ExerciseStandard 对象,
        调参时必须同时更新两者, 确保相位判断和评分使用同一套参数.

        可调参数:
            target_low, target_high, symmetry_max_diff, trunk_max,
            angle_tolerance, smooth_alpha
        """
        std = self.standard
        scorer = self._scorer

        for key, value in kwargs.items():
            if value is None:
                continue
            if hasattr(std, key):
                setattr(std, key, value)
            if hasattr(scorer, key):
                setattr(scorer, key, value)

    def analyze_frame(self, keypoints: np.ndarray,
                      confidences: Optional[np.ndarray] = None) -> AnalysisResult:
        """处理一帧关键点数据, 返回完整分析结果."""
        # 1. 提取关节角度
        angles = self._angle_extractor.extract(keypoints, confidences)

        # 2. 获取主角度值
        primary_val = angles.primary_angle(self.exercise_name)

        # 3. 相位检测与计数
        self._update_phase_and_count(angles, primary_val)

        # 4. 时序特征 — 使用 EMA 平滑角度, 降低快速动作 (如开合跳) 的 jerk 噪声
        if primary_val is not None:
            if self._smooth_primary_val is None:
                self._smooth_primary_val = float(primary_val)
            else:
                # alpha=0.7: 更快响应, 减少预热期 lag 导致的初始分数虚高
                self._smooth_primary_val = (0.7 * float(primary_val)
                                            + 0.3 * self._smooth_primary_val)
            temporal_val = self._smooth_primary_val
        else:
            temporal_val = None
        temporal = self._temporal_extractor.update(temporal_val, self.phase)

        # 5. 错误检测
        errors = self._error_detector.detect(angles, keypoints, confidences,
                                              self.phase, self.exercise_name)

        # 6. 评分更新
        self._scorer.update_angle(primary_val, self.phase)
        self._scorer.update_symmetry(angles)
        score = self._scorer.compute(temporal)

        # 7. 无运动时限制得分 — 空站/静止不动不应得高分
        #    进入"低位"(动态动作)或"保持"(静态动作)后正常评分
        if self.count == 0 and self.phase not in ("低位", "保持"):
            score.total = min(score.total, 50.0)

        # 8. 在 rep 里程碑时自动生成总体评分报告
        overall = None
        if self._session_active and self.count > 0 and self.count % 5 == 0:
            prev_count = getattr(self, '_last_milestone_count', 0)
            if self.count != prev_count:
                overall = self.get_overall_rating()
                self._last_milestone_count = self.count

        return AnalysisResult(
            angles=angles,
            temporal=temporal,
            phase=self.phase,
            count=self.count,
            hold_time=self.hold_time,
            errors=errors,
            score=score,
            overall=overall,
        )

    def _update_phase_and_count(self, angles: JointAngles,
                                 primary_val: Optional[float]):
        """运动相位状态机 + 计数."""
        std = self.standard
        low_min, low_max = std.low_range
        high_min, high_max = std.high_range

        if primary_val is None:
            return

        if std.hold_threshold is not None:
            # 平板支撑等静态动作: 计时
            if low_min <= primary_val <= low_max:
                if self._hold_start is None:
                    self._hold_start = time.time()
                    self.phase = "保持"
                    self._mark_session_start()
                else:
                    self.hold_time = time.time() - self._hold_start
                    self.phase = "保持"
            else:
                self._hold_start = None
                self.phase = "姿态调整"
            return

        # 动态动作计数
        prev_phase = self.phase
        if std.count_trigger == "high":
            if primary_val <= low_max:
                self.phase = "低位"
            elif primary_val >= high_min and self.phase == "低位":
                self.count += 1
                self.phase = "高位"
            elif primary_val >= high_min and self.phase == "等待":
                self.phase = "高位"
        else:
            if primary_val >= high_min:
                self.phase = "高位"
            elif primary_val <= low_max and self.phase == "高位":
                self.count += 1
                self.phase = "低位"
            elif primary_val <= low_max and self.phase == "等待":
                self.phase = "低位"

        # 追踪运动开始时间
        if prev_phase == "等待" and self.phase != "等待":
            self._mark_session_start()

    def _mark_session_start(self):
        """标记运动会话开始时间."""
        if self._session_start_time is None:
            self._session_start_time = time.time()
            self._session_active = True

    def get_score(self) -> ScoreResult:
        return self._scorer.compute(self._temporal_extractor.update(None, self.phase))

    def get_overall_rating(self) -> OverallRating:
        """获取整个运动过程的总体评分报告.

        聚合从运动开始到当前的所有评分数据, 生成包含定性评级、
        趋势分析和改进建议的总体报告.

        可在运动进行中调用（获取阶段性评级），也可在运动结束后调用。

        Returns:
            OverallRating: 总体评分报告.
        """
        duration = 0.0
        if self._session_start_time is not None:
            duration = time.time() - self._session_start_time
        return self._scorer.get_overall_rating(
            total_reps=self.count,
            duration_seconds=duration,
        )

    def get_errors(self) -> list[ErrorInfo]:
        """获取当前活跃错误（简化版，不传关键点则返回空）."""
        return []

    def reset(self):
        self.count = 0
        self.phase = "等待"
        self._hold_start = None
        self.hold_time = 0.0
        self._session_start_time = None
        self._session_active = False
        self._smooth_primary_val = None
        self._temporal_extractor.reset()
        self._scorer.reset()
        self._error_detector.reset()


# ============================================================================
# 自测代码
# ============================================================================

def _self_test():
    """模块自测: 用合成关键点验证各组件计算正确性."""
    print("=" * 60)
    print("pose_analyzer 自测")
    print("=" * 60)

    # 合成一帧标准站姿关键点 (17, 2), 模拟身高约 170cm 在 640x480 图像上
    kp = np.array([
        [320, 80],   # 0  nose
        [305, 70],   # 1  left_eye
        [335, 70],   # 2  right_eye
        [295, 75],   # 3  left_ear
        [345, 75],   # 4  right_ear
        [280, 140],  # 5  left_shoulder
        [360, 140],  # 6  right_shoulder
        [240, 220],  # 7  left_elbow
        [400, 220],  # 8  right_elbow
        [210, 300],  # 9  left_wrist
        [430, 300],  # 10 right_wrist
        [290, 280],  # 11 left_hip
        [350, 280],  # 12 right_hip
        [280, 380],  # 13 left_knee
        [360, 380],  # 14 right_knee
        [275, 470],  # 15 left_ankle
        [365, 470],  # 16 right_ankle
    ], dtype=np.float32)
    conf = np.ones(17, dtype=np.float32) * 0.9

    # --- 测试 1: 关节角度提取 ---
    print("\n[1] 关节角度提取")
    extractor = JointAngleExtractor()
    angles = extractor.extract(kp, conf)
    print(f"  左膝角度: {angles.knee_left:.1f}°")
    print(f"  右膝角度: {angles.knee_right:.1f}°")
    print(f"  左肘角度: {angles.elbow_left:.1f}°")
    print(f"  右肘角度: {angles.elbow_right:.1f}°")
    print(f"  左髋角度: {angles.hip_left:.1f}°")
    print(f"  右髋角度: {angles.hip_right:.1f}°")
    print(f"  躯干倾角: {angles.trunk_angle:.1f}°")

    # 站姿应接近 180° 膝角
    assert angles.knee_left is not None and angles.knee_left > 150, f"站姿膝角应 >150°, 实际 {angles.knee_left:.1f}"
    assert angles.knee_right is not None and angles.knee_right > 150, f"站姿膝角应 >150°, 实际 {angles.knee_right:.1f}"
    print("  [PASS] 站姿膝角验证通过")

    # --- 测试 2: 时序特征 ---
    print("\n[2] 时序特征提取")
    temporal_ext = TemporalFeatureExtractor(window_size=30)
    # 模拟 10 帧稳定角度
    for _ in range(10):
        features = temporal_ext.update(170.0, "高位")
    print(f"  角速度: {features.angular_velocity:.2f} °/s")
    print(f"  平滑度: {features.smoothness:.2f}")
    assert features.angular_velocity < 1.0, f"稳定帧角速度应接近0, 实际 {features.angular_velocity:.2f}"
    print("  [PASS] 稳定时序验证通过")

    # --- 测试 3: 动作标准参数 ---
    print("\n[3] 动作标准参数")
    all_exercises = ["深蹲", "俯卧撑", "平板支撑", "卷腹", "开合跳",
                     "引体向上", "臀桥", "高抬腿", "肩推", "侧平举"]
    for name in all_exercises:
        std = EXERCISE_STANDARDS.get(name)
        assert std is not None, f"缺少动作: {name}"
        print(f"  {name}: 主关节={std.primary_joint}, "
              f"低位={std.low_range}, 高位={std.high_range}")
    print(f"  [PASS] 全部{len(all_exercises)}个动作已定义")

    # --- 测试 4: 评分算法 ---
    print("\n[4] 评分算法")
    scorer = MovementScorer("深蹲")
    # 模拟完美的深蹲角度: 170° 高位保持
    for _ in range(20):
        scorer.update_angle(170.0, "高位")
        scorer.update_symmetry(angles)
    temporal = TemporalFeatures(angular_velocity=5.0, smoothness=2.0,
                                 rhythm_consistency=0.05, rom_consistency=0.03)
    score = scorer.compute(temporal)
    print(f"  总分: {score.total:.1f}/100")
    print(f"  角度得分: {score.angle_score:.1f}/40")
    print(f"  时序得分: {score.temporal_score:.1f}/30")
    print(f"  对称得分: {score.symmetry_score:.1f}/30")
    assert score.total > 70, f"完美动作得分应 >70, 实际 {score.total:.1f}"
    print("  [PASS] 评分验证通过")

    # --- 测试 4b: 总体评分报告 ---
    print("\n[4b] 总体评分报告")
    overall = scorer.get_overall_rating(total_reps=10, duration_seconds=30.0)
    print(f"  综合评分: {overall.total_score:.1f}/100")
    print(f"  等级: {overall.grade_emoji} {overall.grade}")
    print(f"  趋势: {overall.trend}")
    print(f"  亮点: {overall.highlight}")
    print(f"  短板: {overall.weakness}")
    print(f"  分维度: {overall.dimension_breakdown}")
    print(f"  建议: {overall.suggestion}")
    assert overall.grade in ("优秀", "良好"), f"完美动作等级应为优秀/良好, 实际 {overall.grade}"
    print("  [PASS] 总体评分报告验证通过")

    # --- 测试 5: 错误检测 ---
    print("\n[5] 错误检测")

    # 5a: 深蹲膝盖内扣 — 构造膝部内扣的关键点
    kp_valgus = kp.copy()
    kp_valgus[13] = [265, 380]  # 左膝向内偏移
    kp_valgus[14] = [375, 380]  # 右膝向内偏移
    detector = ErrorDetector()
    angles_bad = extractor.extract(kp_valgus, conf)
    errors = detector.detect(angles_bad, kp_valgus, conf, "低位", "深蹲")
    # 连续调用 5 次以上触发
    for _ in range(6):
        errors = detector.detect(angles_bad, kp_valgus, conf, "低位", "深蹲")
    knee_valgus_errors = [e for e in errors if e.name == "膝盖内扣"]
    if knee_valgus_errors:
        print(f"  [PASS] 深蹲膝盖内扣: 已检测 — {knee_valgus_errors[0].suggestion}")
    else:
        print("  [WARN] 深蹲膝盖内扣: 未触发 (阈值可能需要调整)")

    # 5b: 深蹲弓背 — 躯干倾角异常
    kp_round = kp.copy()
    kp_round[5] = [310, 160]   # 肩前移
    kp_round[6] = [390, 160]
    kp_round[11] = [320, 280]  # 髋前移
    kp_round[12] = [380, 280]
    angles_round = extractor.extract(kp_round, conf)
    detector2 = ErrorDetector()
    for _ in range(6):
        errors = detector2.detect(angles_round, kp_round, conf, "低位", "深蹲")
    back_errors = [e for e in errors if e.name == "深蹲弓背"]
    if back_errors:
        print(f"  [PASS] 深蹲弓背: 已检测 — {back_errors[0].suggestion}")
    else:
        print(f"  [WARN] 深蹲弓背: 未触发 (躯干角={angles_round.trunk_angle:.1f}°)")

    # 5c: 俯卧撑塌腰 — 构造塌腰关键点
    kp_sag = np.array([
        [320, 70],   # 0  nose
        [310, 60],   # 1
        [330, 60],   # 2
        [300, 65],   # 3
        [340, 65],   # 4
        [280, 150],  # 5  shoulder
        [360, 150],  # 6
        [250, 220],  # 7  elbow
        [390, 220],  # 8
        [220, 190],  # 9  wrist
        [420, 190],  # 10
        [290, 300],  # 11 hip (偏低)
        [350, 300],  # 12 hip
        [285, 380],  # 13 knee
        [355, 380],  # 14 knee
        [280, 460],  # 15 ankle
        [360, 460],  # 16 ankle
    ], dtype=np.float32)
    angles_sag = extractor.extract(kp_sag, conf)
    detector3 = ErrorDetector()
    for _ in range(6):
        errors = detector3.detect(angles_sag, kp_sag, conf, "低位", "俯卧撑")
    sag_errors = [e for e in errors if "塌腰" in e.name]
    if sag_errors:
        print(f"  [PASS] 俯卧撑塌腰: 已检测 — {sag_errors[0].suggestion}")
    else:
        print("  [WARN] 俯卧撑塌腰: 未触发 (阈值可能需要调整)")

    # 5d: 卷腹颈部用力
    kp_neck = kp.copy()
    kp_neck[0] = [320, 100]  # 鼻子离肩太近
    angles_neck = extractor.extract(kp_neck, conf)
    # 调整躯干角模拟卷腹
    angles_neck.trunk_angle = 40.0
    detector4 = ErrorDetector()
    for _ in range(6):
        errors = detector4.detect(angles_neck, kp_neck, conf, "低位", "卷腹")
    neck_errors = [e for e in errors if e.name == "卷腹颈部用力"]
    if neck_errors:
        print(f"  [PASS] 卷腹颈部用力: 已检测 — {neck_errors[0].suggestion}")
    else:
        print("  [WARN] 卷腹颈部用力: 未触发 (阈值可能需要调整)")

    print("\n" + "=" * 60)
    print("自测完成")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
