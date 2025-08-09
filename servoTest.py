import RPi.GPIO as GPIO
import time

# === CONFIGURATION ===
SERVO_PIN = 18  # BCM pin number (change if needed)

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Start PWM at 50Hz (standard for hobby servos)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)  # 0% duty cycle, servo idle

def set_angle(angle):
    """Move servo to specified angle (0–180)."""
    duty = 2 + (angle / 18)  # maps 0–180° to ~2–12% duty cycle
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)  # allow servo to move
    pwm.ChangeDutyCycle(0)  # stop sending signal to avoid jitter

try:
    while True:
        print("Moving to 0°")
        set_angle(0)
        time.sleep(1)

        print("Moving to 90°")
        set_angle(90)
        time.sleep(1)

        print("Moving to 180°")
        set_angle(180)
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping servo test.")
    pwm.stop()
    GPIO.cleanup()
