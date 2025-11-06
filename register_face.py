import cv2
import os
def register_face(name):
    cap = cv2.VideoCapture(0)
    count = 0
    save_path = f"dataset/{name}"
    os.makedirs(save_path, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Register - Press 'c' to capture, 'q' to quit", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            count += 1
            cv2.imwrite(f"{save_path}/{name}_{count}.jpg", frame)
            print(f"[INFO] Captured {count}")
        elif key & 0xFF == ord('q') or count >= 5:
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Finished capturing for {name}")
if __name__ == "__main__":
    username = input("Enter name for registration: ")
    register_face(username)