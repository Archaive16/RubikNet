import cv2
import numpy as np

face_names = ["front", "back", "top", "bottom", "right", "left"]
cap = cv2.VideoCapture(0)
i = 0
size = 150

while i < 6:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    height, width = frame.shape[:2]
    centerx, centery = width // 2, height // 2

    # Draw rectangle in the center
    cv2.rectangle(frame, (centerx - size, centery - size), (centerx + size, centery + size), (0, 255, 0), 2)

    # Show face name
    cv2.putText(frame, f"Place: {face_names[i].upper()} | Press 'C' to Capture",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("Rubik's Cube Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c'):
        face_filename = f"{face_names[i]}.jpeg"
        face_crop = frame[centery - size:centery + size, centerx - size:centerx + size]
        cv2.imwrite(face_filename, face_crop)
        print(f"Saved {face_filename}")
        i += 1

cap.release()
cv2.destroyAllWindows()

face_images = ["front.jpeg", "back.jpeg", "top.jpeg", "bottom.jpeg", "right.jpeg", "left.jpeg"]
face_names = ["front", "back", "top", "bottom", "right", "left"]
colors_array = np.empty(54, dtype=object)  # Fixed size for 54 tiles

def classify_color(h, s, v):
    if s < 50 and v > 160:
        return "w"
    elif 5 <= h <= 15 and s > 115:
        return "o"
    elif h < 5 or h > 170:
        return "r"
    elif 40<= h <= 80 and s > 100:
        return "g"
    elif 95 <= h <= 110 and s > 100:
        return "b"
    elif 20 <= h <= 35 and s > 100 and v > 150:
        return "y"
    else:
        return "unknown"

index = 0

for face_num, img_path in enumerate(face_images):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    tile_h = height // 3
    tile_w = width // 3

    print(f"\nProcessing face: {face_names[face_num]}")
    for row in range(3):
        for col in range(3):
            y = row * tile_h
            x = col * tile_w
            tile = hsv[y + tile_h//4 : y + 3*tile_h//4,
                       x + tile_w//4 : x + 3*tile_w//4]
            h, s, v = np.mean(tile.reshape(-1, 3), axis=0)
            color = classify_color(h, s, v)

            if color == "unknown":
                print(f"\nUnknown color detected at index a[{index}]")
                print(f"HSV = ({int(h)}, {int(s)}, {int(v)})")
                cv2.imshow(f"Tile a[{index}]", img[y:y+tile_h, x:x+tile_w])
                cv2.waitKey(1)
                manual = input("Please type the correct color (e.g. red, green, blue): ")
                color = manual
                cv2.destroyAllWindows()

            colors_array[index] = color
            index += 1

    # Show face for verification
    start = index - 9
    print(f"\nDetected colors for {face_names[face_num]} face:")
    for i in range(start, start + 9):
        print(f"a[{i}] = {colors_array[i]}", end="\t")
        if (i - start + 1) % 3 == 0:
            print()

    while True:
        verify = input("Do you want to correct any color on this face? (yes/no): ")
        if verify == "no":
            break
        elif verify == "yes":
                edit_index = int(input("Enter index to correct (e.g. 14): "))
                new_color = input("Enter correct color: ")
                colors_array[edit_index] = new_color
                print(f"Updated a[{edit_index}] to {new_color}")
        else:
            print("Please answer with yes or no.")
            
# Final print
print("\nFinal Verified Colors:")
for i in range(54):
    print(f"a[{i}] = {colors_array[i]}")

