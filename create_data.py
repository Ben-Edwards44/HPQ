from random import randint


def write(x, y, value):
    with open("data.txt", "a") as file:
        file.write(f"{x},{y},{value}\n")


def main():
    for _ in range(200):
        x = randint(0, 400)
        y = randint(0, 400)

        if y < 1.5 * x:
            value = 1
        else:
            value = 0

        write(x, y, value)


main()