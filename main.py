from utils import *
import numpy as np
import nn

def center_of_mass(arr):
    rows, cols = arr.shape
    total_mass = arr.sum()
    center_of_mass_x = np.sum(np.arange(cols) * arr) / total_mass
    center_of_mass_y = np.sum(np.arange(rows) * arr.T) / total_mass
    return center_of_mass_x, center_of_mass_y

def shift_center_of_mass(arr):
    center_x, center_y = center_of_mass(arr)
    rows, cols = arr.shape
    target_center_x = (cols - 1) / 2
    target_center_y = (rows - 1) / 2
    shift_x = int(target_center_x - center_x)
    shift_y = int(target_center_y - center_y)

    shifted_arr = np.roll(arr, shift_x, axis=1)
    shifted_arr = np.roll(shifted_arr, shift_y, axis=0)

    return shifted_arr

WIN = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Handwritten Alphanumeric Detector!")

def init_grid(rows, cols):
    grid = []

    for i in range(rows):
        grid.append([])
        for _ in range(cols):
            grid[i].append(0)

    return grid

def draw_grid(win, grid):
    for i, rows in enumerate(grid):
        for j, pixel in enumerate(rows):
            pygame.draw.rect(win, (pixel,pixel,pixel), (j*PIXEL_SIZE, i*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

def get_row_col_from_pos(pos):
    x,y = pos
    row = int(y // PIXEL_SIZE)
    col = int(x // PIXEL_SIZE)

    if row >= ROWS:
        raise IndexError

    return row, col

def draw(win, grid):
    win.fill(BG_COLOR)
    draw_grid(win, grid)
    pygame.display.update()

def paint(grid, row, col):
    grid[row][col] = 255
    grid[row-1][col] = min(grid[row-1][col]+120, 255)
    grid[row+1][col] = min(grid[row+1][col]+120, 255)
    grid[row][col-1] = min(grid[row][col-1]+120, 255)
    grid[row][col+1] = min(grid[row][col+1]+120, 255)

network = nn.NeuralNetwork([784,30,30,26])
network.biases = load_object("alphabets_biases.pickle")
network.weights = load_object("alphabets_weights.pickle")

running = True
clock = pygame.time.Clock()
grid = init_grid(ROWS, COLS)

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()

            try:
                row, col = get_row_col_from_pos(pos)
                paint(grid, row, col)
            except IndexError:
                grid = np.array(grid)

                # print(grid)

                grid = grid.reshape((784, 1))

                # shifted_array = shift_center_of_mass(grid)

                # temp = network.feedforward(shifted_array)
                temp = network.feedforward(grid)
                
                i = np.argmax(temp)
                print("--------------")
                print("The character is",chr(ord('A')+i))
                print("--------------")

                grid = init_grid(ROWS, COLS)
    
    draw(WIN, grid)

pygame.quit()