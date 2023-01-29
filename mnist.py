import pickle
import pygame
import numpy
from os import getcwd
from PIL import Image
from random import randint


#Initialise pygame
pygame.init()
window = pygame.display.set_mode((450, 250))


def get_test_data():
    #Get the MNIST data from a file and return the input and output data
    with open(f"{getcwd()}\\MNIST dataset\\mnist_data.csv", "r") as file:
        data = file.read()

    data = data.split("\n")
    data.pop(0)

    inputs = []
    outputs = []
    for i in data[:100]:
        line = i.split(",")

        output = int(line[0])
        input = [int(i) for i in line[1:]]

        inputs.append(input)
        outputs.append(output)

    return inputs, outputs


def test_network(inputs, outputs, model):
    #Test the network against data
    total_correct = 0
    answers = []

    for i, x in zip(inputs, outputs):
        output, all_outputs = model.classify(i)
        
        if output == x:
            total_correct += 1

        answers.append((i, all_outputs))

    print(f"Accuracy: {total_correct / len(inputs) * 100}%")

    return answers


def load_network(filename):
    #Load a saved network
    with open(filename, "rb") as file:
        network = pickle.load(file)

    return network


def draw_answer(image, outputs):
    #Draw the image along with the predictions
    window.fill((255, 255, 255))

    font = pygame.font.Font('freesansbold.ttf', 18)

    s_outputs = sorted(outputs, reverse=True)

    #Draw the outputs
    for inx, output in enumerate(s_outputs):
        value = outputs.index(output)
        text = font.render(f"{value}: {output * 100 :.2f}%", True, (255, 255, 255))

        txt_rect = text.get_rect()
        txt_rect.center = (350, int(250 / 10 * (inx + 1)) - 15)

        window.blit(text, txt_rect)

    #Convert to 2D array
    pixels = []
    for i in range(28):
        pixels.append([])
        for x in range(28):
            value = image[i * 28 + x]
            pixels[-1].append(value)

    #Resize image
    image = Image.fromarray(numpy.array(pixels, dtype=numpy.uint8), mode="L")
    image = image.resize((250, 250))
    pixels = list(image.getdata())

    #Draw image
    for i in range(250):
        for x in range(250):
            pixel = pixels[i * 250 + x]
            pygame.draw.circle(window, (pixel, pixel, pixel), (x, i), 1)

    pygame.display.update()


def main():
    model = load_network("better_network.pickle")
    inputs, outputs = get_test_data()
    answers = test_network(inputs, outputs, model)

    image, outputs = answers[randint(0, len(answers) - 1)]
    draw_answer(image, outputs)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                image, outputs = answers[randint(0, len(answers) - 1)]
                draw_answer(image, outputs)

main()