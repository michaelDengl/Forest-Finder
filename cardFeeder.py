# cardFeeder.py
import time
import RPi.GPIO as GPIO

# --- Pins (BCM) ---
ENA = 13   # PWM-capable pin for speed (phys 33)
IN1 = 5    # Direction A (phys 29)
IN2 = 6    # Direction B (phys 31)

FREQ_HZ = 1000  # PWM frequency

GPIO.setmode(GPIO.BCM)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

pwm = GPIO.PWM(ENA, FREQ_HZ)
pwm.start(0)

def set_motor(speed_percent: int):
    """
    speed_percent in range -100..100
    >0 = forward, <0 = reverse, 0 = stop
    """
    sp = max(-100, min(100, speed_percent))
    if sp > 0:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        pwm.ChangeDutyCycle(sp)
    elif sp < 0:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        pwm.ChangeDutyCycle(-sp)
    else:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        pwm.ChangeDutyCycle(0)

try:

    duty = 100
    print(f"Drop a card...")
    set_motor(-duty)
    time.sleep(0.6)
    set_motor(0)
    time.sleep(0.0)

    set_motor(duty)
    time.sleep(1.1)
    set_motor(0)
    time.sleep(0.0)

    set_motor(-duty)
    time.sleep(0.6)
    set_motor(0)

finally:
    pwm.stop()
    GPIO.cleanup()
