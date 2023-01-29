import pygame
import new_neural_network


#Initialise pygame
pygame.init()
window = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Neural Network Visulisation")


def draw():
    #Draw background
    for i in range(200):
        for x in range(200):
            input = (i * 2, x * 2)
            output = model.calculate_output(input)

            if output[0] > 0.5:
                colour = (255, 0, 0)
            else:
                colour = (0, 0, 255)

            pygame.draw.circle(window, colour, (i * 2, x * 2), 2)

    #Draw points
    inputs, outputs = get_data()

    for i, x in zip(inputs, outputs):
        if x[0] > 0.5:
            colour = (255, 0, 0)
        else:
            colour = (0, 0, 255)

        pygame.draw.circle(window, (0, 0, 0), i, 4)
        pygame.draw.circle(window, colour, i, 3)


    pygame.display.update()


def train(n):
    #Train the neural network
    inputs, outputs = get_data()

    new_neural_network.mini_batch_descent(model, inputs, outputs)

    #Only draw every 3 iterations (for efficiency)
    if n % 3 == 0:
        draw()


def get_data():
    #Read the data from a file and return the input and output data
    with open("data.txt", "r") as file:
        data = file.read()

    data = data.split("\n")

    inputs = []
    outputs = []
    for i in data:
        line = [int(i) for i in i.split(",")]

        inputs.append(line[:2])
        outputs.append([line[2]])

    return inputs, outputs


def main():
    #Main loop
    n = 0
    while True:
        n += 1
        train(n)


#Create neural network model
model = new_neural_network.Neural_network(0.1, 50, (2, 10, 1))


main()