import cv2 
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

def detector():
    cap = cv2.VideoCapture("data/people.mp4")

    while cap.isOpened():
        status, frame = cap.read()
        
        if not status:
            break

        # Inferencia
        pred = model(frame)
        # xmin,ymin,xmax,ymax
        df = pred.pandas().xyxy[0]
        # Filtrar por confidence
        df = df[df["confidence"] > 0.5]
        
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax","ymax"]].values.astype(int)

            # print bboxes: frame -> (xmin, ymin) (xmax, ymax)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            # print text
            cv2.putText(frame, 
                        f"{df.iloc[i]['name']}: {round(df.iloc[i]['confidence'],4)}", 
                        (bbox[0], bbox[1] - 15), 
                        cv2.FONT_HERSHEY_PLAIN, 
                        1, (255,255,255), 
                        2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cap.release()


if __name__ == '__main__':
    detector()

        