import time

class TrafficFSM:
    def __init__(
        self,
        conf_threshold=0.5,
        min_consecutive_frames=1,
        state_timeout=5.0,
        min_bbox_area_sign=2000,
        min_bbox_area_traffic_light=1000,
        roi_x_min=0.15,
        roi_x_max=0.85,
    ):
        self.conf_threshold = conf_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.state_timeout = state_timeout
        self.min_bbox_area_sign = min_bbox_area_sign
        self.min_bbox_area_traffic_light = min_bbox_area_traffic_light
        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max

        self.traffic_light_classes = {"red-light", "green-light"}
        
        self.priority = {
            "red-light": 5,
            "prohibition-sign": 4,
            "left-turn-sign": 3,
            "right-turn-sign": 3,
            "straight-ahead-sign": 3,
            "green-light": 1,
        }

        self.ALL_DIRECTIONS = ["LEFT", "FORWARD", "RIGHT"]
        self.reset()

    def reset(self):
        """Reset trạng thái ban đầu - Dùng khi test ảnh tĩnh trên Notebook"""
        self.active_sign_directions = list(self.ALL_DIRECTIONS)
        self.is_stopped = False
        self.last_detection_time = time.time()

    def _parse_box(self, det, img_w):
        b = det.get("bbox") or det.get("box")
        if not b: return 0, 0
        if "box" in det:
            return b[2] * b[3], (b[0] + b[2] / 2.0) / img_w
        return (b[2] - b[0]) * (b[3] - b[1]), ((b[0] + b[2]) / 2.0) / img_w

    def _is_valid_bbox(self, det, img_w, img_h):
        area, center_x = self._parse_box(det, img_w)
        if area == 0: return False

        min_area = (
            self.min_bbox_area_traffic_light
            if det.get("class_name") in self.traffic_light_classes
            else self.min_bbox_area_sign
        )
        # Chỉ lọc theo Area và ROI X
        return (area >= min_area) and (self.roi_x_min <= center_x <= self.roi_x_max)

    def _get_valid_detections(self, detections, img_w, img_h):
        return [
            d for d in detections
            if d.get("confidence", 0.0) >= self.conf_threshold
            and d.get("class_name") in self.priority
            and self._is_valid_bbox(d, img_w, img_h)
        ]

    def update(self, detections, img_w=640, img_h=480):
        now = time.time()
        valid_dets = self._get_valid_detections(detections, img_w, img_h)

        if valid_dets:
            self.last_detection_time = now

            # 1. CẬP NHẬT ĐÈN GIAO THÔNG
            has_red = any(d["class_name"] == "red-light" for d in valid_dets)
            has_green = any(d["class_name"] == "green-light" for d in valid_dets)

            if has_red and has_green:
                red_det = max((d for d in valid_dets if d["class_name"] == "red-light"), key=lambda d: self._parse_box(d, img_w)[0])
                green_det = max((d for d in valid_dets if d["class_name"] == "green-light"), key=lambda d: self._parse_box(d, img_w)[0])
                self.is_stopped = self._parse_box(red_det, img_w)[0] > self._parse_box(green_det, img_w)[0]
            elif has_red:
                self.is_stopped = True
            elif has_green:
                self.is_stopped = False

            # 2. CẬP NHẬT BIỂN BÁO HƯỚNG ĐI
            sign_dets = [d for d in valid_dets if d["class_name"] not in self.traffic_light_classes]
            if sign_dets:
                sign_dets.sort(
                    key=lambda d: (self.priority.get(d["class_name"], 0), self._parse_box(d, img_w)[0]),
                    reverse=True
                )
                best_sign = sign_dets[0]["class_name"]

                if best_sign == "left-turn-sign":
                    self.active_sign_directions = ["LEFT"]
                elif best_sign == "right-turn-sign":
                    self.active_sign_directions = ["RIGHT"]
                elif best_sign == "straight-ahead-sign":
                    self.active_sign_directions = ["FORWARD"]
                elif best_sign == "prohibition-sign":
                    # Biển cấm đi thẳng -> Chỉ cho phép Rẽ Trái và Rẽ Phải
                    self.active_sign_directions = ["LEFT", "RIGHT"]

        else:
            if now - self.last_detection_time > self.state_timeout:
                self.active_sign_directions = list(self.ALL_DIRECTIONS)
                self.is_stopped = False

        if self.is_stopped:
            return ["STOP"]
        
        return self.active_sign_directions