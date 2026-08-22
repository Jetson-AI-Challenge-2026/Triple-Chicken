import time


class IntersectionDecisionMaker:
    """
    Tầng Ra quyết định (Decision Making Layer)
    Kết hợp:
      - YOLO (Mệnh lệnh luật giao thông)
      - Road Geometry Model (Khả năng vật lý của làn đường: Straight, Left, Right)
      - Bounding Box Area/Y (Khoảng cách)
    """
    def __init__(
        self,
        controller,
        min_area_trigger=3500,  # Ngưỡng diện tích (w * h) đủ gần để thực hiện rẽ
        y_bottom_trigger=320,   # Tọa độ Y_max của box tiệm cận nửa dưới ảnh
        cooldown_time=3.0,      # Thời gian khóa nhận diện sau khi vừa rẽ xong (giây)
        memory_hold_time=0.8    # Thời gian giữ bộ nhớ tạm nếu YOLO bị lỡ frame (giây)
    ):
        self.controller = controller
        self.min_area_trigger = min_area_trigger
        self.y_bottom_trigger = y_bottom_trigger
        self.cooldown_time = cooldown_time
        self.memory_hold_time = memory_hold_time
        
        self.last_action_time = 0
        self.last_detected_sign = None
        self.last_detected_time = 0

    def process_detections(self, detections, lane_steering, possible_directions):
        """
        detections: Danh sách dict từ YOLO [{'label': 'TURN_LEFT', 'box': [...]}]
        lane_steering: Góc lái tính toán từ mô hình bám đường ONNX.
        possible_directions: List/Dict hướng có làn đường từ Mô hình mới.
                             Ví dụ: ['STRAIGHT', 'LEFT'] hoặc {'straight': True, 'left': True, 'right': False}
        """
        now = time.time()
        
        # Chuyển đổi possible_directions về dạng list chữ hoa thống nhất
        if isinstance(possible_directions, dict):
            available_paths = [k.upper() for k, v in possible_directions.items() if v]
        else:
            available_paths = [d.upper() for d in possible_directions]

        # 1. Trạng thái COOLDOWN: Vừa rẽ xong, tạm bỏ qua biển báo để tránh rẽ 2 lần
        if now - self.last_action_time < self.cooldown_time:
            self.controller.set_steering(lane_steering)
            self.controller.set_throttle(self.controller.base_throttle)
            return "STATE_COOLDOWN"

        # 2. Tìm biển báo quan trọng trong frame hiện tại
        target_sign = None
        for det in detections:
            if det['label'] in ['TURN_LEFT', 'TURN_RIGHT', 'STOP', 'RED_LIGHT']:
                target_sign = det
                break

        # 3. Memory Buffer chống lỡ frame YOLO
        if target_sign is not None:
            self.last_detected_sign = target_sign
            self.last_detected_time = now
        else:
            if self.last_detected_sign is not None and (now - self.last_detected_time < self.memory_hold_time):
                target_sign = self.last_detected_sign

        # 4. Nếu KHÔNG thấy biển báo -> Chạy theo làn đường tự nhiên
        if target_sign is None:
            self.controller.set_steering(lane_steering)
            self.controller.set_throttle(self.controller.base_throttle)
            return f"STATE_LANE_FOLLOWING (Paths: {available_paths})"

        # 5. Trích xuất thông tin Bounding Box
        x1, y1, x2, y2 = target_sign['box']
        box_area = (x2 - x1) * (y2 - y1)
        y_max = y2
        label = target_sign['label']

        # 6. Xử lý ĐÈN ĐỎ / STOP
        if label in ['RED_LIGHT', 'STOP']:
            if box_area > 1500:
                self.controller.stop()
                return f"STATE_WAITING_{label}"

        # 7. Xử lý RẼ TRÁI / RẼ PHẢI tại Giao lộ
        if label in ['TURN_LEFT', 'TURN_RIGHT']:
            # Kiểm tra khoảng cách kích hoạt
            is_close_enough = (box_area >= self.min_area_trigger) or (y_max >= self.y_bottom_trigger)

            if not is_close_enough:
                # Tiến vào giao lộ đi chậm
                self.controller.set_steering(lane_steering)
                self.controller.set_throttle(self.controller.base_throttle * 0.8)
                return f"STATE_APPROACHING_{label} (Area: {int(box_area)})"
            
            else:
                # ĐÃ ĐẾN VÙNG KÍCH HOẠT -> ĐỐI CHIẾU VỚI MÔ HÌNH HƯỚNG ĐƯỜNG MỚI
                
                # --- KIỂM TRA RẼ TRÁI ---
                if label == 'TURN_LEFT':
                    if 'LEFT' in available_paths or 'TURN_LEFT' in available_paths:
                        print(f"[ACTION CONFIRMED] Biển báo RẼ TRÁI + Làn đường TRÁI hợp lệ!")
                        self.controller.move_cm(distance_cm=15, speed_factor=1.0)
                        self.controller.turn_left(duration=1.2)
                        
                        self.last_action_time = time.time()
                        self.last_detected_sign = None
                        return "STATE_EXECUTED_TURN_LEFT"
                    else:
                        print(f"[WARNING] Biển báo RẼ TRÁI nhưng KHÔNG CÓ làn đường trái! Hủy rẽ.")
                        self.controller.set_steering(lane_steering)
                        self.controller.set_throttle(self.controller.base_throttle)
                        return "STATE_CANCELLED_NO_LEFT_LANE"

                # --- KIỂM TRA RẼ PHẢI ---
                elif label == 'TURN_RIGHT':
                    if 'RIGHT' in available_paths or 'TURN_RIGHT' in available_paths:
                        print(f"[ACTION CONFIRMED] Biển báo RẼ PHẢI + Làn đường PHẢI hợp lệ!")
                        self.controller.move_cm(distance_cm=15, speed_factor=1.0)
                        self.controller.turn_right(duration=1.0)
                        
                        self.last_action_time = time.time()
                        self.last_detected_sign = None
                        return "STATE_EXECUTED_TURN_RIGHT"
                    else:
                        print(f"[WARNING] Biển báo RẼ PHẢI nhưng KHÔNG CÓ làn đường phải! Hủy rẽ.")
                        self.controller.set_steering(lane_steering)
                        self.controller.set_throttle(self.controller.base_throttle)
                        return "STATE_CANCELLED_NO_RIGHT_LANE"

        return "STATE_LANE_FOLLOWING"