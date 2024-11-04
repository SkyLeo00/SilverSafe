import cv2
from ultralytics import YOLO

# 웹캠 캡처기 생성
cap = cv2.VideoCapture('/Users/sangwon_back/Chrome 다운로드/IoT_Proj/falling_video.mp4')

# YOLO 모델 로드
model = YOLO('/Users/sangwon_back/Chrome 다운로드/IoT_Proj/best.pt')

while cap.isOpened():
    # 캡처기로부터 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    try:
        # 모델을 통한 detection
        results = model(frame)

        # 검출 결과를 시각화하여 출력할 프레임을 복사
        inference_plot = frame.copy()

        # 정확도가 0.6 이상인 결과만 필터링
        for box in results[0].boxes:
            if box.conf.item() >= 0.6:  # 텐서를 숫자로 변환
                # 경계 상자를 그림
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(inference_plot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 라벨과 정확도 추가
                label = f'{results[0].names[int(box.cls)]} {box.conf.item():.2f}'
                cv2.putText(inference_plot, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 검출 결과를 시각화하여 출력
        cv2.imshow('inference_plot', inference_plot)

        # 읽은 프레임을 출력 윈도우에 표시
        cv2.imshow('frame', frame)

    except Exception as e:
        print(f"예외 발생: {e}")

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 캡처기와 비디오 쓰기기 해제
cap.release()

# 출력 윈도우 닫기
cv2.destroyAllWindows()
