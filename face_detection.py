import cv2

# Load Haar cascade for face detection
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Check if cascade loaded properly
if face.empty():
    print("Error loading cascade classifier")
    exit()

# Start video capture from default camera
video_cap = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    ret, video_data = video_cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the video with detections
    cv2.imshow("video_live", video_data)

    # Break loop if 'a' key is pressed
    if cv2.waitKey(10) == ord("a"):
        break

# Release camera and close windows
video_cap.release()
cv2.destroyAllWindows()
