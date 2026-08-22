import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)


class RacecarController:

    def __init__(self, base_throttle=0.15):
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar

            self.car = NvidiaRacecar()
        except ImportError:

            class MockCar:

                def __init__(self):
                    self.steering = 0.0
                    self.throttle = 0.0

            self.car = MockCar()

        self.base_throttle = base_throttle

    def set_steering(self, val):
        self.car.steering = max(-1.0, min(1.0, float(val)))

    def set_throttle(self, val):
        self.car.throttle = max(-1.0, min(1.0, float(val)))

    def stop(self):
        """Dừng xe an toàn bằng cách ngắt ga và trả thẳng lái."""
        self.set_throttle(0.0)
        self.set_steering(0.0)

    def execute_action(self, action_name, custom_steer=None):
        action = str(action_name).upper()

        if custom_steer is not None:
            self.set_steering(custom_steer)
            self.set_throttle(self.base_throttle)
            return

        if action in ["FORWARD", "GREEN_LIGHT"]:
            self.set_steering(0.0)
            self.set_throttle(self.base_throttle)
        elif action == "STOP":
            self.stop()
        elif action == "TURN_LEFT":
            self.set_steering(-1.)
            self.set_throttle(self.base_throttle)
        elif action == "TURN_RIGHT":
            self.set_steering(1.)
            self.set_throttle(self.base_throttle)