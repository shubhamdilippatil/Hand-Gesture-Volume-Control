import cv2
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# ================= WINDOWS VOLUME =================

devices = AudioUtilities.GetSpeakers()

interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(
    interface,
    POINTER(IAudioEndpointVolume)
)


# ================= MEDIAPIPE =================

model_path = "hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# ================= WEBCAM =================

cap = cv2.VideoCapture(0)

# Current volume
current_volume = volume.GetMasterVolumeLevelScalar() * 100


# ================= MAIN LOOP =================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect hand
    result = detector.detect(mp_image)


    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Thumb tip = 4
        # Index finger tip = 8

        thumb = hand[4]
        index = hand[8]

        x1 = int(thumb.x * w)
        y1 = int(thumb.y * h)

        x2 = int(index.x * w)
        y2 = int(index.y * h)


        # ================= DISTANCE =================

        distance = math.hypot(
            x2 - x1,
            y2 - y1
        )


        # ================= VOLUME =================

        # Change these values if required

        MIN_DISTANCE = 30
        MAX_DISTANCE = 200

        target_volume = (
            (distance - MIN_DISTANCE)
            /
            (MAX_DISTANCE - MIN_DISTANCE)
        ) * 100

        target_volume = max(
            0,
            min(100, target_volume)
        )


        # Smooth volume
        current_volume = (
            current_volume * 0.85
            +
            target_volume * 0.15
        )


        # Set laptop volume
        volume.SetMasterVolumeLevelScalar(
            current_volume / 100,
            None
        )


        # ================= DRAW HAND =================

        for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:

            start = hand[connection.start]
            end = hand[connection.end]

            sx = int(start.x * w)
            sy = int(start.y * h)

            ex = int(end.x * w)
            ey = int(end.y * h)

            cv2.line(
                frame,
                (sx, sy),
                (ex, ey),
                (0, 255, 0),
                2
            )


        # Draw landmarks

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


        # Highlight thumb and index

        cv2.circle(
            frame,
            (x1, y1),
            10,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            frame,
            (x2, y2),
            10,
            (0, 0, 255),
            -1
        )


        # Line between fingers

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )


        # ================= TEXT =================

        cv2.putText(
            frame,
            f"Distance: {int(distance)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Volume: {int(current_volume)}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )


        # ================= VOLUME BAR =================

        bar_x = 20
        bar_y = 120
        bar_width = 300
        bar_height = 30

        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (255, 255, 255),
            2
        )

        filled = int(
            bar_width * current_volume / 100
        )

        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + filled, bar_y + bar_height),
            (0, 255, 0),
            -1
        )


    else:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ================= TITLE =================

    cv2.putText(
        frame,
        "HAND GESTURE VOLUME CONTROL",
        (100, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to exit",
        (20, 475),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    cv2.imshow(
        "Laptop Volume Controller",
        frame
    )


    # Q to quit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ================= CLEANUP =================

cap.release()
cv2.destroyAllWindows()
detector.close()