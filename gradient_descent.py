import pygame
from random import uniform


LEARN_RATE = 0.1


#Initialise pygame
pygame.init()
window = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Gradient Descent")


#The backgorung curve function
func = lambda x: 2 * x**4 + 0.7 * x**3 - 2 * x**2 + 1

#The derivative of the background curve
gradient = lambda x: 8 * x**3 + 2.1 * x**2 - 4 * x

#Convert world point to screen point
convert = lambda point: [int(point[0] * 250 + 250), 500 - int(point[1] * 250)]


def move():
    #Move the point using the gradient descent algorithm
    global pos

    x = pos[0]
    grad = gradient(x)

    new_x = -grad * LEARN_RATE + x
    new_y = func(new_x)

    pos = [new_x, new_y]


def get_curve_points():
    #Get points for drawing the background curve
    points = []
    for i in range(500):
        x = i - 250
        x /= 250
        y = func(x)

        points.append((i, 500 - int(y * 250)))

    return points


def get_gradient_points():
    #Get points for drawing the graient line
    grad = gradient(pos[0])
    y_intercept = pos[1] - grad * pos[0]

    x1 = pos[0] - 0.3
    x2 = pos[0] + 0.3

    y1 = grad * x1 + y_intercept
    y2 = grad * x2 + y_intercept

    return (x1, y1), (x2, y2)


def draw():
    window.fill((0, 0, 0))

    #Draw background curve
    points = get_curve_points()
    for i, x in enumerate(points):
        if i > 0:
            pygame.draw.line(window, (135, 206, 235), x, points[i - 1], 4)
    
    #Draw gradient
    start, end = get_gradient_points()
    pygame.draw.line(window, (255, 0, 0), convert(start), convert(end), 2)

    #Draw current point
    pygame.draw.circle(window, (255, 255, 255), convert(pos), 6)

    pygame.display.update()


def init_pos():
    #Initialise position to random point on curve
    global pos

    x = uniform(-1, 1)
    pos = [x, func(x)]


def main():
    #Main loop
    while True:
        for event in pygame.event.get():            
            if event.type == pygame.MOUSEBUTTONDOWN:
                move()
                draw()
            elif event.type == pygame.KEYDOWN:
                init_pos()
                draw()
            elif event.type == pygame.QUIT:
                quit()


init_pos()
draw()
main()