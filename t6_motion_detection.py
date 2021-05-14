import cv2

cap = cv2.VideoCapture('move.mp4') #讀取影片

ret, frame = cap.read() #讀取第一幀
while True:
    frame_old = frame.copy() #儲存上一幀
    ret, frame = cap.read() #讀取下一幀
    if ret == False: #如果影片沒下一幀
        cap = cv2.VideoCapture('move.mp4') #重新讀取影片
        ret, frame = cap.read() #補前面沒讀到的一幀
    
    diff = cv2.absdiff(frame, frame_old) #將上一幀和下一幀相減
    grey = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) #轉灰階
    _, binary = cv2.threshold(grey, 30, 255, cv2.THRESH_BINARY) #二值化
    filt = cv2.erode(binary, None, iterations=1) #使用侵蝕函式(濾波)
    result = cv2.dilate(filt, None, iterations=30) #使用擴張函式
    cnts, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #尋找輪廓
    draw = frame.copy() #創建一畫面
    for c in cnts: #對輪廓在此範圍的框起來
        if cv2.contourArea(c) > 20000 and cv2.contourArea(c) < 100000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(draw, (x, y), (x + w, y + h), (0,0,255), 2)
    
    cv2.imshow('0.original', frame)
    cv2.imshow('1.diff', diff)
    cv2.imshow('2.binary', binary)
    cv2.imshow('3.filt', filt)
    cv2.imshow('4.result', result)
    cv2.imshow('5.draw', draw)

    if cv2.waitKey(30) & 0xFF == ord('q'): break #按Q跳出

cap.release()
cv2.destroyAllWindows()