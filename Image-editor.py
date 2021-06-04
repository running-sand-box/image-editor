import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#functions that restricts input value within input bounds
def clampInt(_value, _min, _max):
    if (_value < _min): return _min
    elif (_value > _max): return _max
    else: return _value
    
def equalorAboveLowerBound(_value, _lowerBound):
    return _value - 1 >= _lowerBound

def belowUpperBound(_value, _upperBound):
    return _value + 1 < _upperBound
"""
Parameters:
    _value: input value that needs to be clamped
    _min: smallest integer that _value can take
    _max: largest integer that _value can take
    
returns _value such that it is within _min and _max and _value is int
"""

#function that checks if input x and y coordinates are within input bounds
def inBound(_x, _y, _maxX, _maxY, _minX = 0, _minY = 0):
    return equalorAboveLowerBound(_x, _minX) and equalorAboveLowerBound(_y, _minY) and belowUpperBound(_x, _maxX) and belowUpperBound(_y, _maxY)
"""
Parameters:
    _x: x coordinate that needs to be checked
    _y: y coordinate that needs to be checked
    _minX: (default at 0) smallest value _x can take
    _minY: (default at 0) smallest value _y can take
    _maxX: adds 1 to the largest value _x can take
    _maxY: adds 1 to the largest value _y can take
    
returns True if x and y values are within bounds    
returns False if x or y value is not within bounds
"""

#[c] is used instead of [0],[1],[2] to keep it neater
def change_brightness(image, value): 
    newImage = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=int)
    
    for x in range(0, newImage.shape[0]): 
        for y in range(0, newImage.shape[1]): 
            newImage[x][y] = [clampInt(image[x][y][c] + value, 0, 255) for c in range(0, newImage.shape[2])]
    
    return newImage
   
def change_contrast(image, value): 
    newImage = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=int) 
    
    for x in range(0, newImage.shape[0]): 
        for y in range(0, newImage.shape[1]): 
            newImage[x][y] = [clampInt(((259 * (value + 255)) / (255 *  (259 - value))) * (image[x][y][c] - 128) + 128, 0, 255) for c in range(0, newImage.shape[2])]
    
    return newImage

def grayscale(image): 
    newImage = np.copy(image) 
    
    for x in range(len(image)): 
        for y in range(len(image[x])): 
            newImage[x][y] = [0.3 * image[x][y][0] + 0.59 * image[x][y][1] + 0.11 * image[x][y][2] for c in range(0, len(image[x][y]))]
    
    return newImage

def getPixelNeighbours(_image, _x, _y):
    return [[_image[_x + x, _y + y] for x in range(-1, 2)] for y in range(-1, 2) if inBound(_x, _y, _image.shape[0], _image.shape[1], 0, 0)]
"""
Parameters:
    _image: the image that is being worked on
    _x: the x coordinate of the "root" pixel
    _y: the y coordinate of the "root" pixel

returns the 2D matrix of neighbours, root of matrix is the (1,1) element
"""

def getAllNeighbours(_image):
    return [[getPixelNeighbours(_image, x, y) for y in range(0, _image.shape[1])] for x in range(0, _image.shape[0])]
"""
Parameter:
    _image: the image that is being worked on
    
returns 2D array of neighbours
"""

def convolution(_matrixAEntry, _matrixBEntry):
    return _matrixAEntry * _matrixBEntry

def getConvolutionSum(_neighbours, _kernel, _colour):
    convolutionSum = 0
    
    for x in range(0, len(_neighbours)):
        for y in range(0, len(_neighbours[x])):
            convolutionSum += convolution(_kernel[x][y], _neighbours[x][y][_colour])

    return convolutionSum

def convolutionFilter(_image, _kernel, _brighten):
    allNeighbours = getAllNeighbours(_image)
    newImage = applyConvolutionToImage(_image, allNeighbours, _kernel)
    
    if _brighten:
        for x in range(0, len(newImage)):
            for y in range(0, len(newImage[x])):
                if inBound(x, y, len(newImage), len(newImage[x]), 0, 0):
                    newImage[x][y] = [clampInt(newImage[x][y][c] + 128, 0, 255) for c in range(0, 3)]
    
    return newImage

def applyConvolutionToImage(_image, _allNeighbours, _kernel):
    # Do not use np.zeros_like(_image)
    # There seems to be some sort of clamp on the values in the 3rd dimension
    # It causes values to wrap around 255 (I think) and back
    # If that doesn't make sense - Then the simple way to explain it 
    # is that it keeps the values between 0 and 255, which is not 
    # particularly useful tbh
    newImage = np.zeros((_image.shape[0], _image.shape[1], _image.shape[2]), dtype=int)
    
    for x in range(0, _image.shape[0]):
        for y in range(0, _image.shape[1]):
            if inBound(x, y, _image.shape[0], _image.shape[1], 0, 0):
                newImage[x][y] = [getConvolutionSum(_allNeighbours[x][y], _kernel, c) for c in range(0, _image.shape[2])]
            else:
                # If the pixel is at the side, 
                # we just reassign the same pixel 
                newImage[x][y] = _image[x][y]
            
    return newImage

def blur_effect(image):
    kernel = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625]
    ]

    return convolutionFilter(image, kernel, False)

def edge_detection(image):
    kernel = [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ]
    
    return convolutionFilter(image, kernel, True)

def embossed(image):
    kernel = [
        [-1, -1, 0],
        [-1, 0, 1],
        [0, 1, 1]
    ]         
    
    return convolutionFilter(image, kernel, True)

def rectangle_select(_image, _pixelA, _pixelB):
    newMask = np.zeros((_image.shape[0], _image.shape[1]), dtype=int)
    
    for x in range(_pixelA[0], _pixelB[0] + 1):
        for y in range(_pixelA[1], _pixelB[1] + 1):
            newMask[x][y] = 1
            
    return newMask

def calculateColourDistance(_rA, _gA, _bA, _rB, _gB, _bB):
    deltaR = int(_rA) - int(_rB)
    deltaG = int(_gA) - int(_gB)
    deltaB = int(_bA) - int(_bB)
    avgR = (int(_rA) + int(_rB)) / 2
    
    return math.sqrt((2 + avgR / 256) * (deltaR ** 2) + 4 * (deltaG ** 2) + (2 + (255 - avgR) / 256) * (deltaB ** 2))
    
def getDirectNeighbours(_x, _y, _maxX, _maxY, _minX = 0, _minY = 0):
    neighbours = []
    
    if equalorAboveLowerBound(_x, _minX):
        neighbours.append((_x - 1, _y))
    if equalorAboveLowerBound(_y, _minY):
        neighbours.append((_x, _y - 1))
    if belowUpperBound(_x, _maxX):
        neighbours.append((_x + 1, _y))
    if belowUpperBound(_y, _maxY):
        neighbours.append((_x, _y + 1))
    
    return neighbours

# We assume _pixel is a 2 element tuple
def floodFill(_image, _pixel, _thres):
    newMask = np.full((_image.shape[0], _image.shape[1]), -1)
    pixelsToCheck = [_pixel]
    
    origPixelColour = _image[_pixel[0]][_pixel[1]]
    
    while len(pixelsToCheck) > 0:
        currentPixel = pixelsToCheck.pop()
        currX = currentPixel[0]
        currY = currentPixel[1]
        
        if (newMask[currX, currY] != -1):
            continue
        
        currPixelColour = _image[currX][currY]
        
        if calculateColourDistance(origPixelColour[0], origPixelColour[1], origPixelColour[2], currPixelColour[0], currPixelColour[1], currPixelColour[2]) <= _thres:
            newMask[currX, currY] = 1
            neighbours = getDirectNeighbours(currX, currY, len(_image), len(_image[currX]))
            pixelsToCheck.extend(neighbours)
        else:
            newMask[currX, currY] = 0
    
    # Set the rest
    for x in range(0, len(newMask)):
        for y in range(0, len(newMask[x])):
            if (newMask[x][y] == -1):
                newMask[x][y] = 0
    
    return np.array(newMask, dtype=float)

def magic_wand_select(_image, _pixel, _thres):     
    return floodFill(_image, _pixel, _thres) # to be removed when filling this function

def applyMask(_oldImage, _newImage, _mask):
    newImage = np.zeros((_mask.shape[0], _mask.shape[1], 3), dtype=int)
    
    for x in range(0, len(_mask)):
        for y in range(0, len(_mask[x])):
            if (_mask[x][y] == 1):
                newImage[x][y] = _newImage[x][y]
            elif (_mask[x][y] == 0):
                newImage[x][y] = _oldImage[x][y]
        
    return newImage

def compute_edge(mask):           
    rsize, csize = len(mask), len(mask[0]) 
    edge = np.zeros((rsize,csize))
    if np.all((mask == 1)): return edge        
    for r in range(rsize):
        for c in range(csize):
            if mask[r][c]!=0:
                if r==0 or c==0 or r==len(mask)-1 or c==len(mask[0])-1:
                    edge[r][c]=1
                    continue
                
                is_edge = False                
                for var in [(-1,0),(0,-1),(0,1),(1,0)]:
                    r_temp = r+var[0]
                    c_temp = c+var[1]
                    if 0<=r_temp<rsize and 0<=c_temp<csize:
                        if mask[r_temp][c_temp] == 0:
                            is_edge = True
                            break
    
                if is_edge == True:
                    edge[r][c]=1
            
    return edge

def save_image(filename, image):
    mpimg.imsave(filename,image)

def load_image(filename):
    img = mpimg.imread(filename)
    if len(img[0][0])==4: # if png file
        img = np.delete(img, 3, 2)
    if type(img[0][0][0])==np.float32:  # if stored as float in [0,..,1] instead of integers in [0,..,255]
        img = img*255
        img = img.astype(np.uint8)
    mask = np.ones((len(img),len(img[0]))) # create a mask full of "1" of the same size of the loaded image
    return img, mask

def display_image(image, mask):
    # if using Spyder, please go to "Tools -> Preferences -> IPython console -> Graphics -> Graphics Backend" and select "inline"
    tmp_img = image.copy()
    edge = compute_edge(mask)
    for r in range(len(image)):
        for c in range(len(image[0])):
            if edge[r][c] == 1:
                tmp_img[r][c][0]=255
                tmp_img[r][c][1]=0
                tmp_img[r][c][2]=0
 
    plt.imshow(tmp_img)
    plt.axis('off')
    plt.show()
    print("Image size is",str(len(image)),"x",str(len(image[0])))

def tryGetInt(_optionalText):
    value = -1
    while True: 
        try: value = int(input(_optionalText)) 
        except: print("You have entered a non-integer value. Please try again.") 
        else: break 
    return value 
    
def getBoundedInt(_min, _max, _optionalText = "Please enter an integer value: "): 
    value = _min - 1
    while value < _min or value >= _max: 
        value = tryGetInt(_optionalText)
        
        if (value < _min or value >= _max):
            print("Please enter a value between", _min, "(Inclusive) and", _max, "(Exclusive)") 
    return value

def tryLoadImage():
    img = mask = np.array([])
    image = input("Enter file name: ") 
    while True: 
        try: 
            img, mask = load_image(image) 
        except: 
            print("You have entered an invalid file.") 
            image = input("Enter file name: ") 
        else: 
            break 
    
    return img, mask

def menu():
    
    img = mask = np.array([])  
    newImg = newMask = np.array([])
    imageLoaded = False
    applyNewMask = False
    running = True
 
    while running:
        if imageLoaded:
            #Display Full Menu
            print ('''Menu
                --------------------------- 
                What do you want to do ?
                e - exit
                l - load a picture
                s - save the current picture
                1 - adjust brightness
                2 - adjust contrast
                3 - apply grayscale
                4 - apply blur
                5 - edge detection
                6 - embossed
                7 - rectangle select
                8 - magic wand select''')
                
            option = input("Your choice: ")
            if option == 'e': 
                print("Thank you for using our image editor program. See you soon.")
                running = False
                break
            elif option == 'l': 
                img, mask = tryLoadImage()
                newImg = img
            elif option == 's': 
                if newImg.size > 0:
                    newName = datetime.datetime.now().strftime("%Y-%m-%d %H%M")
                    save_image(newName + ".png", newImg.astype('uint8')) 
                    print("Image saved as", newName + ".png")
                else:
                    print("No image to save.")
            elif option == '1': 
                changedImage = change_brightness(newImg, getBoundedInt(0, 255))
                
                if (applyNewMask): 
                    newImg = applyMask(newImg, changedImage, newMask)
                else:
                    newImg = np.copy(changedImage)
                    
                display_image(newImg, mask)
            elif option == '2': 
                changedImage = change_contrast(newImg, getBoundedInt(0, 255))
                
                if (applyNewMask): 
                    newImg = applyMask(newImg, changedImage, newMask)
                else:
                    newImg = np.copy(changedImage)
                    
                display_image(newImg, mask)
            elif option == '3': 
                changedImage = grayscale(newImg)
                
                if (applyNewMask): 
                    newImg = applyMask(newImg, changedImage, newMask)
                else:
                    newImg = np.copy(changedImage)
                    
                display_image(newImg, mask)
            elif option == '4': 
                changedImage = blur_effect(newImg)
                
                if (applyNewMask): 
                    newImg = applyMask(newImg, changedImage, newMask)
                else:
                    newImg = np.copy(changedImage)
                    
                display_image(newImg, mask)
            elif option == '5': 
                changedImage = edge_detection(newImg)
                
                if (applyNewMask): 
                    newImg = applyMask(newImg, changedImage, newMask)
                else:
                    newImg = np.copy(changedImage)
                
                display_image(newImg, mask)
            elif option == '6': 
                changedImage = embossed(newImg)
                
                if (applyNewMask): 
                    newImg = applyMask(newImg, changedImage, newMask)
                else:
                    newImg = np.copy(changedImage)
                
                display_image(newImg, mask)
            elif option == '7': 
                inputTupleA = (getBoundedInt(0, img.shape[0], "Please enter top-left pixel position x: "), getBoundedInt(0, img.shape[1], "Please enter top-left pixel position y: "))
                inputTupleB = (getBoundedInt(inputTupleA[0], img.shape[0], "Please enter bottom-right pixel position x: "), getBoundedInt(inputTupleA[1], img.shape[1], "Please enter bottom-right pixel position y: "))
                newMask = rectangle_select(img, inputTupleA, inputTupleB)
                applyNewMask = True
            elif option == '8': 
                inputTuple = (getBoundedInt(0, img.shape[0], "Please enter pixel position x: "), getBoundedInt(0, img.shape[1], "Please enter pixel position y: "))
                newMask = magic_wand_select(img, inputTuple, tryGetInt("Please enter threshold: "))
                applyNewMask = True
            else: 
                print("You have an entered an invalid option. Please try again.") 
                continue 
            
        else:
            #Display Initial Menu
            print ('''Menu
                --------------------------- 
                What do you want to do ?
                e - exit  
                l - load a picture''')
            option = input("Your choice: ")    
            if option == 'e':
                print ("Thank you for using the image editor.")
                running = False
            elif option == 'l':
                img, mask = tryLoadImage()   
                newImg = img
                display_image(img, mask)
                imageLoaded = True
            else:
                print("You have entered an invalid option, please try again.")
                continue
            
if __name__ == "__main__":
    menu()
    
    
