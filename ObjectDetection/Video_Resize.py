import cv2

cap = cv2.VideoCapture('assets/3.mp4')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Resolution: {width}*{height}, and fps={fps}")

# set the new arbitrary resolution
new_w, new_h = 640, 480

# video writer to save resized frames 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('assets/33.mp4', fourcc, fps, (new_w, new_h))

while True:
    # ret = return --> bool (if cv2 could capture the video)
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (new_w, new_h))
    out.write(resized_frame)


cap.release()
out.release()
print(f"Original Resoltion: {width}*{height}, FPS: {fps}, Resized to: {new_w}*{new_h}")





