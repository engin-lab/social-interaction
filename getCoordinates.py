import cv2
import numpy as np

# Initialize a list to store coordinates


img = None

# Define the mouse callback function
def click_event(event, x, y, flags, param):
    global coordinates
    global img
    if event == cv2.EVENT_LBUTTONDOWN:  # Left button click
        coordinates.append((x, y))  # Append click position to the coordinates list
        
        # Draw a circle at click position
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        
        # Display the coordinates on the image
        text = f"({x}, {y})"
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.25, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('image', img)  # Show the image with the mark and coordinates

def get_coordinates(file_location):
    global coordinates
    coordinates = [] 
    global img
    img = cv2.imread(file_location)
    if img is None:
        print(f"Failed to load image from '{file_location}'. Please check the file path and try again.")
        return
    cv2.imshow('image', img)
    # Set the mouse callback function for the 'image' window
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Optionally, save the annotated image
    #cv2.imwrite('annotated_frame.jpg', img)
    # Print or save the coordinates
    # Convert your polygon into the format required by cv2.polylines()
    # This involves converting the list of tuples into a numpy array of shape (1, -1, 2)
    # and specifying the data type as int32
    polygon_points = np.array(coordinates, np.int32)
    polygon_points = polygon_points.reshape((-1, 1, 2))
    # Draw the polygon
    # The arguments are the image, the points, a flag indicating whether the polygon is closed,
    # the color (BGR format), and the line thickness
    cv2.polylines(img, [polygon_points], True, (0, 255, 0), 3)
    # Display the image with the drawn polygon
    cv2.imshow('Arena', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    # Optionally, save the image with the drawn polygon
    cv2.imwrite('arena_marked_image.jpg', img)
    img = cv2.imread(file_location)
    return coordinates

#def main(file_location):
#    #file_location = input("What is the location of the file? please provide full path?  ")# Load an image
#    global img 
#    img = cv2.imread(file_location)
#    if img is None:
#        print(f"Failed to load image from '{file_location}'. Please check the file path and try again.")
#        return
#    cv2.imshow('image', img)
#    coordinates = get_coordinates()
#    return coordinates
