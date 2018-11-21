from imutils.perspective import four_point_transform
import numpy as np
import imutils
import math
import cv2

COLS = 2
ROWS = 2
IMAGE_PATH = "images/pizarra.png"

def get_board_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    boardPoints = []
    maxContourArea = 0

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.01 * peri, True)

        if w > 100 and h > 100 and len(approx) == 4 and cv2.contourArea(c) > maxContourArea:
            cv2.polylines(img, [approx], True, (0, 0, 255), 2)
            boardPoints = approx
            maxContourArea = cv2.contourArea(c)

    return boardPoints

def get_sprites_from_board(image, cols, rows, m = 0):

    # Cut the margins of the board
    h, w, _ = image.shape
    ratio = h / 500.0
    margin = int(ratio * m)
    image = image[margin:(h - 2 * margin), margin:(w - 2 * margin)]

    # Save the original image before resize
    original = image.copy()

    # Resize the board
    h, w, _ = image.shape
    ratio = h / 500.0
    img = imutils.resize(image, height=500)

    # Extract edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 10, 50)

    #Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # Sort contours from left-to-right and top-to-bottom
    cnts = sorted(cnts, key=(lambda x: cv2.boundingRect(x)[0] + cv2.boundingRect(x)[1] * cols), reverse=False)
    sprites = []

    # Find for the sprites rectangles in the board
    for c in cnts:

        # Approximate the polygon that best fits the contour
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.01 * peri, True)

        # Check if the approximated polygon is a rectangle (if the number of points is 4)
        if len(approx) == 4:
            sprite = four_point_transform(original, np.multiply(approx.reshape((4, 2)), ratio))
            h, w, _ = sprite.shape
            margin = int(ratio * m)
            sprite = sprite[margin:(h-2*margin), margin:(w-2*margin)]
            sprites.append(sprite)

    # Get the n largest sprites, where n is the total number of expected sprites (cols*rows)
    sprites = sorted(sprites, key=(lambda x: x.shape[0]*x.shape[1]), reverse=True)[:cols*rows]

    return sprites


def filter_max_figures(sprites):
    # Define dilatation kernel
    kernel = np.array(([0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]), np.uint8)

    # Init return variables
    max_contours = []
    min_w = math.inf
    min_h = math.inf

    # Show all the sprites
    for sprite in sprites:

        # Binarize the image
        gray = cv2.cvtColor(sprite, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Dilate the binarized image to close small holes
        dilation = cv2.dilate(thresh, kernel, iterations=2)

        # Find the contours in the image
        cnts = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # Look for the largest contour in each area
        max_area = 0
        max_bb = None
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area = w * h
            if area > max_area:
                max_bb = (x, y, w, h)
                max_area = area

        (x, y, w, h) = max_bb
        min_w = min(min_w, w)
        min_h = min(min_h, h)
        max_contours.append(sprite[y:(y + h), x:(x + w)])

    # Resize all the sprites to fit the size of the smallest
    for i in range(len(max_contours)):
        max_contours[i] = cv2.resize(max_contours[i], (min_w, min_h), interpolation=cv2.INTER_AREA)

    return max_contours

def generate_animation_texture(sprites):

    # Get spites shape to generate the matrix of the full animation image
    h, w, _ = sprites[0].shape
    anim = np.zeros((h, w * 4), dtype="uint8")

    # Bianrize the sprites and combine them in the animation matrix
    for sprite, i in zip(sprites, range(0, len(sprites))):
        gray = cv2.cvtColor(sprite, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        anim[0:h, (i * w):((i + 1) * w)] = 255 - thresh

    # Convert the animation from gray scale to BGR scale
    anim = cv2.cvtColor(anim, cv2.COLOR_GRAY2BGR)

    return anim

# Load image
img = cv2.imread(IMAGE_PATH)

# Resize the image to h=500 and store the ratio
h, _, _ = img.shape
ratio = h/500.0
original = img.copy()
img = imutils.resize(img, height=500)

# Show the original image
cv2.imshow("Original image", img)

# Get board points
board_points = get_board_points(img)

# Get the board content from the original image
board = four_point_transform(original, np.multiply(board_points.reshape((4, 2)), ratio))

# Extract the sprites from the board
sprites = get_sprites_from_board(board, COLS, ROWS, 5)

# Look for the largest contour in each image
max_contours = filter_max_figures(sprites)

# Generate the texture for the animation from the sprites images
anim = generate_animation_texture(max_contours)

# Replace all white pixel with cyan
anim[np.where((anim==[255,255,255]).all(axis=2))] = [255,255,0]

# Show the animation texture
cv2.imshow("Animation texture", anim)
cv2.waitKey(0)

# Withe the texture in a file
cv2.imwrite("anims/test.png", anim)
