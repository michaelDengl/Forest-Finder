import RPi.GPIO as GPIO
import time

SERVO_PIN = 18  # BCM pin for PWM signal
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz standard servo frequency
pwm.start(0)

def set_angle(angle):
    """Set servo to a specific angle (0-180)."""
    duty = 2 + (angle / 18)  # Map 0-180° to ~2-12% duty cycle
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.02)  # Small delay for movement

try:
    print("Resetting to 0°")
    set_angle(90)
    time.sleep(1)

    print("Slowly moving to 180°")
    for angle in range(0, 181, 5):  # Step 1 degree at a time
        set_angle(angle)
        time.sleep(0.001)  # Adjust speed (higher = slower)

    print("Reached 180°")

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    pwm.stop()
    GPIO.cleanup()
