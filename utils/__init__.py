from .settings import *
from .functions import *
import pygame

pygame.init()
pygame.font.init()

def get_font(size):
    return pygame.font.SysFont("comicsans", size)