#implementing menu interface

def menu():
    # img = np.array([])
    imageLoaded = False
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
                print ("Thank you for using the image editor")
                running = False
            elif option == 'l':  
                print ("Please enter a filename and load an image from that file")
                # img = mpimg.imred(filename) 
            elif option == 's':
                print ("Please enter a filename and save the current image in that file")
                # mpimg.imsave (filename,img)
            elif option == '1':
                print ("Brightness updated")
            elif option == '2':
                print ("Contrast updated")
            elif option == '3':
                print ("Grayscale applied")
            elif option == '4':
                print ("Blur effect applied")                
            elif option == '5':
                print ("Edge detection effect applied")
            elif option == '6':     
                print ("Emboss effect applied")
            elif option == '7':
                print ("Rectangle selection implemented")
            elif option == '8':   
                print ("Magic wand selection implemented")
            else:
                print("You have entered an invalid option, please try again.")
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
                # img = mpimg.imred(filename) 
                imageLoaded = True
            else:
                print("You have entered an invalid option, please try again.")
                continue
            
if __name__ == "__main__":
    menu()
    
    
        

