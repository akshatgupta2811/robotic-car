import cv2
import mediapipe as mp
import math
import socket
import time
import threading

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# --- Direction to motor speed mapping ---
def direction_to_speeds(direction):
    """Convert gesture direction to left|right motor speeds"""
    speed_map = {
        "RIGHT": (80, -80),    # Turn right: left forward, right backward
        "LEFT":  (-80, 80),    # Turn left: left backward, right forward  
        "UP":    (120, 120),   # Move forward: both forward
        "DOWN":  (-100, -100), # Move backward: both backward
        "NONE":  (0, 0)        # Stop: both motors off
    }
    return speed_map.get(direction, (0, 0))

def classify_angle(theta_deg):
    """Classify finger angle into direction"""
    if -45 < theta_deg <= 45:
        return "RIGHT"
    elif 45 < theta_deg <= 135:
        return "UP"
    elif theta_deg > 135 or theta_deg <= -135:
        return "LEFT" 
    elif -135 < theta_deg <= -45:
        return "DOWN"
    return "NONE"

# --- Socket client setup ---
ESP_IP = "192.168.4.1"
ESP_PORT = 81
RECONNECT_DELAY = 3

# Global variables for ESP communication
sock = None
latest_sensor_data = "No data"
connection_status = "Disconnected"

def esp_communication_thread():
    """Separate thread to handle ESP32 communication"""
    global sock, latest_sensor_data, connection_status
    
    while True:
        if sock is None:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((ESP_IP, ESP_PORT))
                connection_status = "Connected"
                print(f"[SUCCESS] Connected to ESP32 at {ESP_IP}:{ESP_PORT}")
            except socket.error as e:
                connection_status = "Disconnected"
                print(f"[ERROR] Connection failed: {e}")
                time.sleep(RECONNECT_DELAY)
                continue
        
        try:
            # Set a short timeout for receiving data
            sock.settimeout(0.1)
            response = sock.recv(4096)  # Larger buffer for sensor data
            if response:
                # Parse the latest sensor reading from the stream
                data_str = response.decode('utf-8').strip()
                if data_str:
                    # Extract the most recent complete sensor reading
                    lines = data_str.split()
                    if len(lines) >= 4:
                        # Format: orientation, left_sensor, front_sensor, right_sensor
                        latest_sensor_data = f"Orient:{lines[0][:5]} L:{lines[1]} F:{lines[2]} R:{lines[3]}"
        except socket.timeout:
            pass  # No data received this cycle
        except socket.error as e:
            print(f"[ERROR] Communication error: {e}")
            sock.close()
            sock = None
            connection_status = "Reconnecting..."
            
        time.sleep(0.1)

def send_motor_command(left_speed, right_speed):
    """Send motor command to ESP32"""
    global sock
    if sock:
        try:
            command = f"{left_speed}|{right_speed}\n"
            sock.send(command.encode('utf-8'))
            return True
        except socket.error:
            return False
    return False

# Start the ESP communication thread
esp_thread = threading.Thread(target=esp_communication_thread, daemon=True)
esp_thread.start()

print("[INFO] Starting gesture recognition (ESP communication in background)")

# --- Main processing loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame and process with MediaPipe
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        h, w, _ = frame.shape

        direction = "NONE"
        angle_text = "No hand detected"

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            
            # Get index finger base (MCP) and tip coordinates
            x0, y0 = int(lm[5].x * w), int(lm[5].y * h)  # MCP joint
            x1, y1 = int(lm[8].x * w), int(lm[8].y * h)  # Fingertip
            
            # Calculate angle (invert y for proper coordinate system)
            dx, dy = x1 - x0, y0 - y1
            theta_deg = math.degrees(math.atan2(dy, dx))
            angle_text = f"{theta_deg:.1f}°"
            direction = classify_angle(theta_deg)

            # Draw hand landmarks and finger vector
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (x0, y0), 8, (0,140,255), -1)  # Base (orange)
            cv2.circle(frame, (x1, y1), 8, (0,255,0), -1)    # Tip (green)
            cv2.line(frame, (x0, y0), (x1, y1), (255,255,255), 3)

        # Convert direction to motor speeds and send command
        left_speed, right_speed = direction_to_speeds(direction)
        send_success = send_motor_command(left_speed, right_speed)
        
        # Display info on frame
        cv2.putText(frame, f"Angle: {angle_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Direction: {direction}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Motors: L={left_speed}, R={right_speed}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
        cv2.putText(frame, f"Status: {connection_status}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"Sensors: {latest_sensor_data}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Hand Gesture Robot Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    # Cleanup
    cap.release()
    if sock:
        try:
            send_motor_command(0, 0)  # Stop motors
            sock.close()
        except:
            pass
    cv2.destroyAllWindows()
    print("[INFO] Cleanup completed")
