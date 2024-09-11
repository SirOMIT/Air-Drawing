import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define a canvas to draw on
canvas = np.zeros((480, 640, 3), dtype="uint8")

# Stack to store canvas history for undo
canvas_history = []
prev_x, prev_y = 0, 0
drawing = False
erasing = False

def is_two_fingers_extended(landmarks):
    """
    Check if both the index and middle fingers are extended.
    """
    index_finger_tip = landmarks[8]  # Index finger tip
    index_finger_mcp = landmarks[5]  # Index finger base (MCP joint)
    middle_finger_tip = landmarks[12]  # Middle finger tip
    middle_finger_mcp = landmarks[9]  # Middle finger base (MCP joint)

    # Both fingers are extended if their tips are above their base MCP joint (in y-direction)
    return index_finger_tip.y < index_finger_mcp.y and middle_finger_tip.y < middle_finger_mcp.y

def is_hand_closed(landmarks):
    """
    Check if the hand is closed (most fingers bent).
    A simple check: if the tips of all fingers are below their respective base joints.
    """
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_finger_tip = landmarks[8]
    index_finger_mcp = landmarks[5]
    middle_finger_tip = landmarks[12]
    middle_finger_mcp = landmarks[9]
    ring_finger_tip = landmarks[16]
    ring_finger_mcp = landmarks[13]
    pinky_finger_tip = landmarks[20]
    pinky_finger_mcp = landmarks[17]

    # Check if most of the finger tips are below their base joints (suggesting a fist)
    return (thumb_tip.y > thumb_ip.y and
            index_finger_tip.y > index_finger_mcp.y and
            middle_finger_tip.y > middle_finger_mcp.y and
            ring_finger_tip.y > ring_finger_mcp.y and
            pinky_finger_tip.y > pinky_finger_mcp.y)

def save_canvas_state():
    """Save the current canvas state for undo."""
    canvas_history.append(canvas.copy())

def undo_last_drawing():
    """Undo the last drawing action."""
    global canvas
    if canvas_history:
        canvas = canvas_history.pop()  # Revert to the last saved state

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a natural selfie view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB (since Mediapipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hand tracking
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Drawing hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of the index finger tip (landmark 8)
            h, w, _ = frame.shape
            index_finger_tip = hand_landmarks.landmark[8]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Check if two fingers (index and middle) are extended to start drawing
            if is_two_fingers_extended(hand_landmarks.landmark):
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                    save_canvas_state()  # Save canvas state before starting new drawing
                else:
                    # If drawing, draw a line on the canvas
                    if not erasing:  # Only draw if not erasing
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)
                    prev_x, prev_y = x, y
                drawing = True
            else:
                drawing = False
                prev_x, prev_y = 0, 0

            # Check if the hand is closed to erase
            if is_hand_closed(hand_landmarks.landmark):
                erasing = True
                canvas = np.zeros((480, 640, 3), dtype="uint8")  # Clear the canvas
                canvas_history.clear()  # Reset history when canvas is cleared
            else:
                erasing = False

    else:
        prev_x, prev_y = 0, 0

    # Combine the frame and canvas
    output_frame = cv2.add(frame, canvas)

    # Show the result
    cv2.imshow("Air Drawing Tracker", output_frame)

    # Press 'u' for undo
    if cv2.waitKey(1) & 0xFF == ord('u'):
        undo_last_drawing()

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
