"""
File: navigation_fsm.py
Mục đích: Quản lý FSM điều khiển chuyển động xe, hỗ trợ mở rộng vòng cua 
khi gặp vật cản/góc khuất và tự động lùi thoát kẹt.
"""

import time

class NavigationFSM:
    def __init__(self, confirm_frames=3, block_threshold=0.9, stop_timeout=30.0, max_reverse_cycles=5):
        self.confirm_frames = confirm_frames
        self.block_threshold = block_threshold
        self.stop_timeout = stop_timeout  # Thời gian chờ tối đa khi gặp STOP (giây)
        self.max_reverse_cycles = max_reverse_cycles  # Số chu kỳ lùi tối đa cho phép
        
        # Các biến quản lý trạng thái
        self.maneuver_state = 'DRIVE'
        self.maneuver_start_time = 0.0
        self.blocked_frame_count = 0
        self.active_reverse_steer = 0.0
        self.active_forward_steer = 0.0
        
        # Bộ đếm số chu kỳ lùi đã thực hiện trong 1 lần kẹt
        self.reverse_cycles_count = 0
        
        # Hướng lùi mặc định khi đi thẳng (sẽ luân phiên giữa -1.0 và 1.0)
        self.default_straight_reverse_steer = -1.0
        
        # Biến bổ sung quản lý chờ STOP / Đèn đỏ
        self.stop_start_time = None
        self.last_valid_direction = "FORWARD"  # Hướng lưu lại trước khi STOP

    def reset(self):
        """Reset trạng thái về ban đầu"""
        self.maneuver_state = 'DRIVE'
        self.maneuver_start_time = 0.0
        self.blocked_frame_count = 0
        self.stop_start_time = None
        self.reverse_cycles_count = 0
        self.default_straight_reverse_steer = -1.0

    def update(self, intended_direction, prob_blocked, car, now):
        """
        Cập nhật trạng thái điều khiển xe.
        """
        # 0. XỬ LÝ ĐẶC BIỆT KHI CÓ LỆNH STOP / ĐÈN ĐỎ
        if intended_direction == "STOP":
            # Đánh dấu thời điểm bắt đầu dừng lần đầu tiên
            if self.stop_start_time is None:
                self.stop_start_time = now

            # Kiểm tra xem đã vượt quá 30 giây chờ chưa
            if now - self.stop_start_time >= self.stop_timeout:
                # Quá 30s -> Bỏ qua STOP, dùng lại hướng hợp lệ gần nhất để đi tiếp
                intended_direction = self.last_valid_direction
            else:
                # Chưa quá 30s -> Giữ nguyên lệnh dừng
                self.maneuver_state = 'DRIVE'
                self.maneuver_start_time = 0.0
                self.blocked_frame_count = 0
                self.reverse_cycles_count = 0
                car.set_steering(0.0)
                car.set_throttle(0.0)
                return 'STOP', 0
        else:
            # Nhận tín hiệu di chuyển hợp lệ khác STOP -> Reset bộ đếm thời gian chờ STOP
            self.stop_start_time = None
            if intended_direction in ["FORWARD", "STRAIGHT", "GO_STRAIGHT", "MOVE_FORWARD", "LEFT", "TURN_LEFT", "RIGHT", "TURN_RIGHT"]:
                self.last_valid_direction = intended_direction

        # 1. Debounce lọc nhiễu tín hiệu BLOCK
        if prob_blocked >= self.block_threshold:
            self.blocked_frame_count += 1
        else:
            self.blocked_frame_count = max(0, self.blocked_frame_count - 1)

        # 2. Máy trạng thái (State Machine)
        if self.maneuver_state == 'DRIVE':
            if self.blocked_frame_count >= self.confirm_frames:
                # Phát hiện BLOCK -> Kích hoạt chuỗi hành động lùi điều chỉnh
                self.maneuver_state = 'REVERSE_TURNING'
                self.maneuver_start_time = now
                self.reverse_cycles_count = 1  # Bắt đầu tính chu kỳ lùi thứ 1

                # LỰA CHỌN GÓC LÙI & TIẾN TÙY THEO HƯỚNG MONG WE WANT
                if intended_direction in ["RIGHT", "TURN_RIGHT"]:
                    # Lùi bẻ lái TRÁI để mở rộng góc đầu xe sang phải
                    self.active_reverse_steer = -1.0  
                    self.active_forward_steer = 0.8   
                
                elif intended_direction in ["LEFT", "TURN_LEFT"]:
                    # Lùi bẻ lái PHẢI
                    self.active_reverse_steer = 1.0   
                    self.active_forward_steer = -0.8  
                
                else: # FORWARD/STRAIGHT hoặc mặc định (Luân phiên Trái / Phải)
                    self.active_reverse_steer = self.default_straight_reverse_steer
                    self.active_forward_steer = -self.default_straight_reverse_steer * 0.8
                    
                    # Đảo chiều hướng lùi cho lần kẹt tiếp theo
                    self.default_straight_reverse_steer *= -1.0

                car.set_steering(self.active_reverse_steer)
                car.set_throttle(-0.22)

            else:
                # Đường FREE -> Di chuyển bình thường theo hướng FSM chỉ định
                if intended_direction in ["FORWARD", "STRAIGHT", "GO_STRAIGHT", "MOVE_FORWARD"]:
                    car.set_steering(0.0)
                    car.set_throttle(0.15)
                elif intended_direction in ["LEFT", "TURN_LEFT"]:
                    car.set_steering(-0.6)
                    car.set_throttle(0.22)
                elif intended_direction in ["RIGHT", "TURN_RIGHT"]:
                    car.set_steering(0.6)
                    car.set_throttle(0.22)
                else:
                    car.set_steering(0.0)
                    car.set_throttle(0.0)

        elif self.maneuver_state == 'REVERSE_TURNING':
            # Thực hiện lùi mở góc trong 1.5 giây
            if now - self.maneuver_start_time < 1.5:
                car.set_steering(self.active_reverse_steer)
                car.set_throttle(-0.22)
            else:
                self.maneuver_state = 'PAUSE'
                self.maneuver_start_time = now
                car.set_steering(0.0)
                car.set_throttle(0.0)

        elif self.maneuver_state == 'PAUSE':
            # Tạm dừng 0.3 giây ngắt quán tính bảo vệ phần cứng
            if now - self.maneuver_start_time < 0.3:
                car.set_steering(0.0)
                car.set_throttle(0.0)
            else:
                self.maneuver_state = 'CHECK_FORWARD'
                self.maneuver_start_time = now
                car.set_steering(self.active_forward_steer)
                car.set_throttle(0.25)

        elif self.maneuver_state == 'CHECK_FORWARD':
            # Tiến theo góc điều chỉnh mới trong 1.2 giây
            if now - self.maneuver_start_time < 1.2:
                car.set_steering(self.active_forward_steer)
                car.set_throttle(0.25)
            else:
                # Kiểm tra lại xem đã thoát kẹt/mở góc cua thành công chưa
                if self.blocked_frame_count < self.confirm_frames:
                    # Đã thoát kẹt thành công -> Reset toàn bộ bộ đếm
                    self.maneuver_state = 'DRIVE'
                    self.blocked_frame_count = 0
                    self.reverse_cycles_count = 0
                    self.default_straight_reverse_steer = -1.0
                else:
                    # Vẫn còn kẹt -> Kiểm tra số chu kỳ đã lùi
                    if self.reverse_cycles_count < self.max_reverse_cycles:
                        # Tăng số chu kỳ lùi và tiếp tục lặp lại
                        self.reverse_cycles_count += 1
                        self.maneuver_state = 'REVERSE_TURNING'
                        self.maneuver_start_time = now
                        
                        # Nếu đi thẳng thì tiếp tục đảo chiều góc lùi cho chu kỳ mới
                        if intended_direction not in ["LEFT", "TURN_LEFT", "RIGHT", "TURN_RIGHT"]:
                            self.active_reverse_steer = self.default_straight_reverse_steer
                            self.active_forward_steer = -self.default_straight_reverse_steer * 0.8
                            self.default_straight_reverse_steer *= -1.0

                        car.set_steering(self.active_reverse_steer)
                        car.set_throttle(-0.22)
                    else:
                        # Đã vượt quá số chu kỳ lùi cho phép -> Ngắt chu trình lùi, trả về DRIVE
                        self.maneuver_state = 'DRIVE'
                        self.blocked_frame_count = 0
                        self.reverse_cycles_count = 0
                        car.set_steering(0.0)
                        car.set_throttle(0.0)

        return self.maneuver_state, self.blocked_frame_count