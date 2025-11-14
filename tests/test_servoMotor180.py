# test_servoMotor180.py
from adafruit_servokit import ServoKit
import time

CHANNEL = 2  # S2 → channel index 2
kit = ServoKit(channels=16)

# MG90S 180° servos often like a slightly wider pulse range
# You can fine-tune this if movement is limited or noisy
s = kit.servo[CHANNEL]
s.set_pulse_width_range(500, 2500)

def move_to(angle, hold=0.6):
    """Move to a specific angle, wait briefly."""
    s.angle = angle
    print(f" → {angle}°")
    time.sleep(hold)

try:
    print("Testing 180° servo on channel 2")
    print("Centering...")
    move_to(90, 1.5)

    print("Sweep 0° → 180° → 0°")
    for angle in range(0, 181, 30):
        move_to(angle)
    for angle in range(180, -1, -30):
        move_to(angle)

    print("Small oscillation around center")
    for _ in range(3):
        for angle in (80, 100, 90):
            move_to(angle, 0.4)

    print("Done.")
finally:
    # Park at center, then disable PWM output on this channel
    try:
        s.angle = 90
        time.sleep(0.5)
        kit._pca.channels[CHANNEL].duty_cycle = 0
    except Exception:
        pass
    print("Servo centered and PWM released.")
