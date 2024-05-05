import pygame
import numpy as np

# Inicializar Pygame
pygame.init()

# Definir los parámetros del entorno
WIDTH, HEIGHT = 800, 600  # Dimensiones de la ventana
BG_COLOR = (255, 255, 255)  # Color de fondo

# Definir los parámetros del robot
ROBOT_RADIUS = 20  # Radio del robot
ROBOT_COLOR = (255, 0, 0)  # Color del robot
ROBOT_SPEED = 5  # Velocidad de movimiento del robot

# Definir la clase Robot
class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, ROBOT_COLOR, (int(self.x), int(self.y)), ROBOT_RADIUS)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

# Crear la ventana de visualización
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Software Robótico")

# Crear un objeto Robot en el centro de la pantalla
robot = Robot(WIDTH // 2, HEIGHT // 2)

# Bucle principal del programa
running = True
while running:
    # Manejo de eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Control del robot utilizando teclas de flecha
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        robot.move(-ROBOT_SPEED, 0)
    if keys[pygame.K_RIGHT]:
        robot.move(ROBOT_SPEED, 0)
    if keys[pygame.K_UP]:
        robot.move(0, -ROBOT_SPEED)
    if keys[pygame.K_DOWN]:
        robot.move(0, ROBOT_SPEED)

    # Limpiar la pantalla
    screen.fill(BG_COLOR)

    # Dibujar el robot en la pantalla
    robot.draw(screen)

    # Actualizar la pantalla
    pygame.display.flip()

# Salir de Pygame
pygame.quit()
