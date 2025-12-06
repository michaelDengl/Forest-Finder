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
    print("Centering at 90°...")
    move_to(90, 0)

    print("Move 90° → 0° → 90°")
    move_to(0)       # go down to 0°
    time.sleep(1.0)
    move_to(90)      # return back to 90°

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
