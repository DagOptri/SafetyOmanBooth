import cv2
import screeninfo

# RTSP stream URL (replace with your RTSP stream link)
rtsp_url = 'rtsp://admin:admin@192.168.0.163/live/av0'

# Function to get screen size
def get_screen_resolution():
    screen = screeninfo.get_monitors()[0]  # Get the first monitor info
    return screen.width, screen.height

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Get the screen resolution
screen_width, screen_height = get_screen_resolution()

# Mouse callback function to capture and print pixel coordinates
def get_pixel_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: X={x}, Y={y}")

# Create a named window for the stream
cv2.namedWindow('RTSP Stream', cv2.WINDOW_NORMAL)

# Set mouse callback function for the window
cv2.setMouseCallback('RTSP Stream', get_pixel_coordinates)

# Display the RTSP stream
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame to fit the screen resolution while maintaining aspect ratio
    height, width, _ = frame.shape
    scale_width = screen_width / width
    scale_height = screen_height / height
    scale = min(scale_width, scale_height)  # Scale to fit within the screen

    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Display the resized frame
    cv2.imshow('RTSP Stream', resized_frame)

    # Press 'q' to exit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
