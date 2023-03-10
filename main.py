import pygame
import pickle
import numpy
from PIL import Image
from time import sleep


#Initialise pygame
pygame.init()
window = pygame.display.set_mode((450, 250))
pygame.draw.line(window, (255, 255, 255), (254, 0), (254, 250), 4)


def check_click():
    #Allows drawing and erasing
    mouse_x, mouse_y = pygame.mouse.get_pos()

    if pygame.mouse.get_pressed(3)[0]:
        if 0 <= mouse_x <= 244 and 0 <= mouse_y <= 250:
            pygame.draw.circle(window, (255, 255, 255), (mouse_x, mouse_y), 8)
    elif pygame.mouse.get_pressed(3)[2]:
        if 0 <= mouse_x <= 244 and 0 <= mouse_y <= 250:
            pygame.draw.circle(window, (0, 0, 0), (mouse_x, mouse_y), 10)

    pygame.display.update()


def get_pixels():
    #Get the pixel values of the drawn image
    new_window = pygame.transform.rotate(window, 90)
    new_window = pygame.transform.flip(new_window, False, True)
    pixel_array = pygame.surfarray.array2d(new_window)

    #Convert to 2D array
    pixels = []
    for i, x in enumerate(pixel_array):
        pixels.append([])
        for y in x[:250]:
            if y == 0:
                pixels[i].append(0)
            else:
                pixels[i].append(255)

    #Resize the array to 28 by 28
    image = Image.fromarray(numpy.array(pixels, dtype=numpy.uint8), mode="L")
    image = image.resize((28, 28))
    pixel_values = list(image.getdata())

    return pixel_values


def load_network(filename):
    with open(filename, "rb") as file:
        network = pickle.load(file)

    return network


def pass_to_network():
    #Input the values into the neural network
    pixel_values = get_pixels()
    inputs = pixel_values
    network = load_network("better_network.pickle")
    
    display_results(network.calculate_output(inputs))


def display_results(outputs):
    #Draw the results
    new_outputs = {}
    for i, x in enumerate(outputs):
        new_outputs[x] = i

    new_outputs = dict(sorted(new_outputs.items(), key=lambda x:x[0], reverse=True))

    pygame.draw.rect(window, (0, 0, 0), (258, 0, 200, 300))
    font = pygame.font.Font('freesansbold.ttf', 18)

    i = 1
    for output, value in new_outputs.items():
        text = font.render(f"{value}: {output * 100 :.2f}%", True, (255, 255, 255))
        txt_rect = text.get_rect()
        txt_rect.center = (350, int(250 / 10 * i) - 15)
        window.blit(text, txt_rect)
        i += 1

    pygame.display.update()


while True:
    check_click()

    keys = pygame.key.get_pressed()

    if keys[pygame.K_SPACE]:
        pass_to_network()
        sleep(0.25)
    elif keys[pygame.K_c]:
        pygame.draw.rect(window, (0, 0, 0), (0, 0, 250, 250))
        pygame.display.update()

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            quit()
