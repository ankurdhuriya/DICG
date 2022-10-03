import math
from PIL import Image, ImageDraw

cell_side = 50
a_size = cell_side / 2.0
b_size = cell_side * math.sqrt(3) / 2.0
height = b_size * 2.0

def get_canvas_centre(x, y):
    cx = cell_side + (3 * x * a_size)
    cy = b_size + (y * height)

    if x % 2: cy += b_size

    return (cx, cy)

def get_hex_cell_vertices(cx, cy):
    x_far_west = cx - cell_side
    x_near_west = cx - a_size
    x_near_east = cx + a_size
    x_far_east = cx + cell_side
    y_north = cy - b_size
    y_mid = cy
    y_south = cy + b_size

    return (
    (x_near_west, y_north), (x_near_east, y_north), (x_far_east, y_mid), (x_near_east, y_south), (x_near_west, y_south),
    (x_far_west, y_mid))

def draw_cell(cx, cy, line_color):
    draw = ImageDraw.Draw(image)

    coord = get_hex_cell_vertices(cx, cy)

    draw.line((coord[0], coord[1]), fill=line_color)
    draw.line((coord[1], coord[2]), fill=line_color)
    draw.line((coord[2], coord[3]), fill=line_color)
    draw.line((coord[3], coord[4]), fill=line_color)
    draw.line((coord[4], coord[5]), fill=line_color)
    draw.line((coord[5], coord[0]), fill=line_color)

def write_text(cx, cy, text):
    ImageDraw.Draw(image).text((cx-cell_side/3.0, cy), text=text, fill='blue')

def fill_cell(row, col, text, fill):
    cx, cy = get_canvas_centre(row, col)
    ImageDraw.Draw(image).ellipse(((cx-a_size, cy-a_size), (cx+a_size, cy+a_size)), fill=fill)
    ImageDraw.Draw(image).text((cx-cell_side/3.0, cy), text=text)

def draw_grid(shape, full_obs, fill='white', line_color='black'):
    rows, cols = shape
    img_width = math.floor(3 * a_size * rows + a_size + cell_side)
    img_height = math.floor(height * cols + b_size + cell_side)

    global image
    image = Image.new("RGB", (img_width, img_height), color=fill)

    for row in range(rows):
        for col in range(cols):
            cx, cy = get_canvas_centre(row, col)
            draw_cell(cx, cy, line_color)
            text = f'({row}, {col})' if full_obs[row][col] == '0' else full_obs[row][col]
            write_text(cx, cy, text=text)

    return image