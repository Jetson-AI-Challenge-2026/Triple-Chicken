import cv2
import numpy as np

class LaneFollower:
    def __init__(self, img_w=640, img_h=480):
        self.img_w = img_w
        self.img_h = img_h

        # Tọa độ căn chỉnh lại chuẩn hơn cho camera đặt ở đầu xe
        src_points = np.float32([
            [int(img_w * 0.15), int(img_h * 0.55)],  # Đỉnh trái (lùi xuống/rộng ra chút)
            [int(img_w * 0.85), int(img_h * 0.55)],  # Đỉnh phải
            [int(img_w * 0.95), int(img_h * 0.95)],  # Đáy phải
            [int(img_w * 0.05), int(img_h * 0.95)]   # Đáy trái
        ])

        dst_points = np.float32([
            [int(img_w * 0.2), 0],
            [int(img_w * 0.8), 0],
            [int(img_w * 0.8), img_h],
            [int(img_w * 0.2), img_h]
        ])
        
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.lane_width_px = 300  # Độ rộng dải làn tiêu chuẩn trên ảnh BEV (pixel)

    def get_bev(self, frame):
        """Chuyển ảnh camera gốc sang góc nhìn BEV (RGB)"""
        return cv2.warpPerspective(frame, self.M, (self.img_w, self.img_h))

    def get_binary_lane(self, bev_frame):
        """Lọc màu HSV vạch trắng/đỏ trên BEV và khử nhiễu nét ngang"""
        hsv = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2HSV)

        # Lọc màu trắng (vạch làn / vạch đi bộ)
        lower_white = np.array([0, 0, 130])
        upper_white = np.array([180, 60, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Lọc màu đỏ (vạch lề / biển báo)
        lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        binary = cv2.bitwise_or(mask_white, mask_red)

        # Morphological filter dạng dọc để ưu tiên bắt vạch làn đường kéo dài
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)

    def process_sliding_window(self, binary_img, allowed_dirs):
        """
        Xác định tâm làn đường bằng Sliding Window dựa theo danh sách hướng được phép:
        - allowed_dirs: Được truyền từ FSM hoặc Model Deep Learning (e.g. ['LEFT'], ['FORWARD'])
        """
        img_h, img_w = binary_img.shape
        vis_frame = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

        # 1. Quét Histogram ở dải đáy ảnh để tìm vị trí xuất phát của 2 vạch
        left_strip = binary_img[int(img_h * 0.75):, :int(img_w * 0.35)]
        right_strip = binary_img[int(img_h * 0.75):, int(img_w * 0.65):]

        hist_left = np.sum(left_strip, axis=0)
        hist_right = np.sum(right_strip, axis=0)

        leftx_base = np.argmax(hist_left) if np.max(hist_left) > 0 else int(img_w * 0.1)
        rightx_base = (np.argmax(hist_right) + int(img_w * 0.65)) if np.max(hist_right) > 0 else int(img_w * 0.9)

        # 2. Cấu hình Sliding Window
        nwindows = 8
        window_height = img_h // nwindows
        margin = 40
        minpix = 30

        nonzero = binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_pts, right_pts = [], []

        for window in range(nwindows):
            win_y_low = img_h - (window + 1) * window_height
            win_y_high = img_h - window * window_height

            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
                left_pts.append((leftx_current, (win_y_low + win_y_high) // 2))

            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))
                right_pts.append((rightx_current, (win_y_low + win_y_high) // 2))

            # Vẽ ô cửa sổ trượt
            cv2.rectangle(vis_frame, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 1)
            cv2.rectangle(vis_frame, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 1)

        car_center = img_w // 2

        # 3. Điều hướng theo thông tin `allowed_dirs` nhận vào
        if allowed_dirs == ["RIGHT"]:
            # Chỉ được rẽ phải -> Bám biên vạch bên phải
            target_lane_center = rightx_current - (self.lane_width_px // 2)
        elif allowed_dirs == ["LEFT"]:
            # Chỉ được rẽ trái -> Bám biên vạch bên trái
            target_lane_center = leftx_current + (self.lane_width_px // 2)
        else:
            # Đi thẳng hoặc linh hoạt theo vạch phát hiện được
            if len(left_pts) > 0 and len(right_pts) > 0:
                target_lane_center = (leftx_current + rightx_current) // 2
            elif len(left_pts) > 0:
                target_lane_center = leftx_current + (self.lane_width_px // 2)
            elif len(right_pts) > 0:
                target_lane_center = rightx_current - (self.lane_width_px // 2)
            else:
                target_lane_center = car_center

        # 4. Tính toán góc đánh lái (Steering Output)
        error = target_lane_center - car_center
        steering = np.clip(error * 0.012, -1.0, 1.0)
        speed = 0.2

        # Trực quan hóa điểm mục tiêu (Đỏ) & Tâm xe (Xanh dương)
        cv2.circle(vis_frame, (target_lane_center, img_h - 30), 8, (0, 0, 255), -1)
        cv2.circle(vis_frame, (car_center, img_h - 30), 8, (255, 0, 0), -1)

        return steering, speed, vis_frame