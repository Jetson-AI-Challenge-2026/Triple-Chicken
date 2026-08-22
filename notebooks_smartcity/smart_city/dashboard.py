import cv2
import numpy as np
from IPython.display import Image


def render_dashboard(
    display_handle,
    frame,
    detections,
    traffic_processor,
    maneuver_state,
    road_status,
    blocked_prob,
    actual_steering,
    actual_throttle,
    fps_real,
    frame_count,
    latency_ms,
    log_panel_width=350,
):
    """Vẽ Bounding Boxes và Log Panel Dashboard lên ảnh."""
    # 1. Vẽ Bounding Boxes
    debug_frame = traffic_processor.draw_bboxes(frame, detections)

    # 2. Tạo Panel màu đen bên phải
    h, w, c = debug_frame.shape
    log_panel = np.zeros((h, log_panel_width, c), dtype=np.uint8)

    # 3. Cấu hình Font & Màu sắc
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)
    status_color = (
        (0, 0, 255) if maneuver_state != 'DRIVE' else (0, 255, 0)
    )  # Đỏ nếu né, xanh nếu bình thường

    y_offset = 30
    line_height = 25

    # Header
    cv2.putText(
        log_panel,
        '=== CAR DASHBOARD ===',
        (10, y_offset),
        font,
        0.6,
        (0, 255, 255),
        2,
    )
    y_offset += 40

    # Maneuver State & Road Status
    cv2.putText(
        log_panel,
        'Maneuver: ',
        (10, y_offset),
        font,
        font_scale,
        text_color,
        font_thickness,
    )
    cv2.putText(
        log_panel,
        f'{maneuver_state}',
        (110, y_offset),
        font,
        font_scale,
        status_color,
        2,
    )
    y_offset += line_height

    cv2.putText(
        log_panel,
        f'Road: {road_status} ({blocked_prob:.2f})',
        (10, y_offset),
        font,
        font_scale,
        text_color,
        font_thickness,
    )
    y_offset += line_height

    # Controls
    cv2.putText(
        log_panel,
        '--- Controls ---',
        (10, y_offset),
        font,
        font_scale,
        (150, 150, 150),
        font_thickness,
    )
    y_offset += line_height
    cv2.putText(
        log_panel,
        f'Steering: {actual_steering:.2f}',
        (20, y_offset),
        font,
        font_scale,
        text_color,
        font_thickness,
    )
    y_offset += line_height
    cv2.putText(
        log_panel,
        f'Throttle: {actual_throttle:.2f}',
        (20, y_offset),
        font,
        font_scale,
        text_color,
        font_thickness,
    )
    y_offset += 40

    # System Metrics
    cv2.putText(
        log_panel,
        '--- System ---',
        (10, y_offset),
        font,
        font_scale,
        (150, 150, 150),
        font_thickness,
    )
    y_offset += line_height
    cv2.putText(
        log_panel,
        f'FPS: {fps_real:.1f}',
        (10, y_offset),
        font,
        font_scale,
        text_color,
        font_thickness,
    )
    y_offset += line_height
    cv2.putText(
        log_panel,
        f'Frame: {frame_count}',
        (10, y_offset),
        font,
        font_scale,
        text_color,
        font_thickness,
    )
    y_offset += line_height
    cv2.putText(
        log_panel,
        f'Latency: {latency_ms:.1f} ms',
        (10, y_offset),
        font,
        font_scale,
        text_color,
        font_thickness,
    )

    # Detections
    if len(detections) > 0:
        y_offset += 40
        cv2.putText(
            log_panel,
            '--- Detections ---',
            (10, y_offset),
            font,
            font_scale,
            (150, 150, 150),
            font_thickness,
        )
        y_offset += line_height
        for d in detections[:5]:
            det_str = f"- {d['class_name']} ({d['confidence']:.2f})"
            cv2.putText(
                log_panel,
                det_str,
                (10, y_offset),
                font,
                0.4,
                (0, 200, 0),
                font_thickness,
            )
            y_offset += 20

    # 4. Ghép ảnh camera + panel và push lên UI
    final_display_frame = np.hstack((debug_frame, log_panel))
    rgb_frame = cv2.cvtColor(final_display_frame, cv2.COLOR_BGR2RGB)
    _, jpeg = cv2.imencode(
        '.jpg', rgb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    )
    display_handle.update(Image(data=jpeg.tobytes()))