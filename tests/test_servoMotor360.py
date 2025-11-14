# test_servoMotor360_safe.py
from adafruit_servokit import ServoKit
import time

CHANNEL = 0  # S0→0, S1→1, etc.
kit = ServoKit(channels=16)
s = kit.continuous_servo[CHANNEL]

try:
    print("Forward 0.7s")
    s.throttle = -0.1
    time.sleep(0.8)

    print("Stop 1s")
    s.throttle = 0.1
    time.sleep(0.1)

    print("Reverse .7s")
    s.throttle = 0.3
    time.sleep(0.7)

    print("Done.")
finally:
    # Always neutral first
    try:
        s.throttle = 0
        time.sleep(0.0)
    except Exception:
        pass

    # Then hard-disable PWM on THIS channel
    try:
        kit._pca.channels[CHANNEL].duty_cycle = 0
    except Exception:
        pass

    # Optional: fully sleep the chip if you’re done with it
    try:
        kit._pca.deinit()
    except Exception:
        pass

    print("Channel stopped and released.")
