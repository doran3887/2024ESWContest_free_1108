import time
import serial  # 아두이노와의 시리얼 통신을 위한 라이브러리
import pygame  # 소리 재생을 위한 pygame 라이브러리

# pygame 초기화
pygame.mixer.init()

# 경고음 파일의 절대 경로 설정
# warning_sound_path = "C:/Users/Yujin/Desktop/hardware/accel_warning.wav"

'''
# 경고음 재생 함수
def play_warning_sound():
    try:
        pygame.mixer.music.load(warning_sound_path)  # 경고음 파일 로드
        pygame.mixer.music.play()  # 경고음 파일 재생
        while pygame.mixer.music.get_busy():  # 경고음이 재생 중일 때 대기
            time.sleep(0.1)
        print("Warning sound played successfully.")
    except Exception as e:
        print(f"Error playing sound: {e}")
'''

# 시리얼 포트 설정 (아두이노와의 통신을 위한 설정)
ser = serial.Serial(port = 'COM7', baudrate= 9600)  # COM 포트를 윈도우 환경에 맞게 설정

# 메인 루프
try:
    while True:
        # 임의의 위치 레벨을 전송
        for level in range(1, 4):
            ser.write(f'{level}\n'.encode())  # 상황 1과 위치 레벨을 아두이노로 전송
            print(f"Sent: Situation 1, Position Level {level}")
            
            # 상황 전송 후 경고음 재생
            #play_warning_sound()

            time.sleep(1)  # 10초 간격 대기

            # 아두이노로부터의 경고 신호 확인
            if ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                print(f"Received from Arduino: {response}")

except KeyboardInterrupt:
    print("Program stopped by user.")
finally:
    ser.close()
    print("Serial port closed.")
