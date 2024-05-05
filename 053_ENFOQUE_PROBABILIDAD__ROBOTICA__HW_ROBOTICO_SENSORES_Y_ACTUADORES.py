import RPi.GPIO as GPIO
import time

# Configurar los pines GPIO para el LED y el sensor ultrasónico
LED_PIN = 18  # Pin GPIO para el LED
TRIG_PIN = 23  # Pin GPIO para el pin de activación del sensor ultrasónico (TRIG)
ECHO_PIN = 24  # Pin GPIO para el pin de eco del sensor ultrasónico (ECHO)

# Configurar el modo de los pines GPIO
GPIO.setmode(GPIO.BCM)

# Configurar el LED como salida
GPIO.setup(LED_PIN, GPIO.OUT)

# Configurar el sensor ultrasónico (TRIG como salida y ECHO como entrada)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# Función para encender/apagar el LED
def toggle_led(state):
    GPIO.output(LED_PIN, state)

# Función para medir la distancia utilizando el sensor ultrasónico
def measure_distance():
    # Generar un pulso corto en el pin TRIG
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, GPIO.LOW)
    
    # Esperar hasta que el pin ECHO se active
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
    
    # Esperar hasta que el pin ECHO se desactive
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
    
    # Calcular la duración del pulso y convertirla en distancia
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # La velocidad del sonido es aproximadamente 343 m/s (17150 cm/s)
    distance = round(distance, 2)
    
    return distance

try:
    while True:
        # Medir la distancia utilizando el sensor ultrasónico
        distance = measure_distance()
        
        # Si la distancia es menor que 20 cm, encender el LED, de lo contrario, apagarlo
        if distance < 20:
            toggle_led(GPIO.HIGH)
        else:
            toggle_led(GPIO.LOW)
        
        print("Distancia:", distance, "cm")
        time.sleep(1)  # Esperar 1 segundo antes de la próxima medición

except KeyboardInterrupt:
    # Limpiar los pines GPIO y salir
    GPIO.cleanup()
