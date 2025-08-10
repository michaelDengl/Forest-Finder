# dropCard.py
import RPi.GPIO as GPIO
import time

SERVO_PIN = 18  # BCM pin for PWM signal

def drop_card():
    """Drop card using a smooth servo sweep from 0° to 180°."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)

    pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz for standard servo
    pwm.start(0)

    def set_angle(angle):
        """Set servo to a specific angle (0–180)."""
        duty = 2 + (angle / 18)
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.02)

    try:
        set_angle(90)
        time.sleep(1)

        for angle in range(0, 181, 5):
            set_angle(angle)
            time.sleep(0.001)

    finally:
        pwm.stop()
        GPIO.cleanup()
