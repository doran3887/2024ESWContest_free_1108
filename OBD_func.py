#pip install obd
#obd2 그냥 함수만 추출하기.

import obd

connection = obd.OBD()

if connection.is_connected():
    print("OBD-II 연결 완료")

    cmd = obd.commands.SPEED            #request랑 query 쌍으로 형성
    response = connection.query(cmd)

    if response.is_null():              #fail처리
        print("data is null")
    else:
        speed = response.value.to("mph") 
        print(f"현재 속도: {speed}")

else:
    print("OBD-II not connected")
    
def get_velocity():
    """
    OBD-II 모듈을 통해 현재 속도를 받아오는 함수.
    반환값: int형 속도 (단위: kph)
    """
    try:
        # OBD-II 블루투스 모듈에 연결 (자동으로 포트를 찾음)
        connection = obd.OBD()

        # 연결 상태 확인
        if connection.is_connected():
            # 속도 데이터를 요청하는 명령
            cmd = obd.commands.SPEED

            # 명령을 보내고 응답 받기
            response = connection.query(cmd)

            # 응답이 유효한지 확인하고 속도 출력
            if response.is_null():
                print("속도 데이터를 가져올 수 없습니다.")
                return 0
            else:
                # 속도를 kph 단위로 변환하고 정수로 반환
                speed = int(response.value.to("kph").magnitude)
                return speed
        else:
            print("OBD-II 모듈에 연결하지 못했습니다.")
            return 0
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return 0

#connection.close()

'''

# obd_speed_printer.py
import obd
import time

def print_speed():
    # OBD-II 포트에 자동으로 연결
    connection = obd.OBD()  # 기본 포트에 연결하거나 자동으로 연결 가능한 포트를 찾음

    if connection.is_connected():
        print("OBD-II 모듈에 성공적으로 연결되었습니다.")
        
        # 속도 데이터를 계속해서 받아오는 루프
        while True:
            # 속도 명령 설정
            cmd = obd.commands.SPEED
            # 명령 실행 및 응답 받기
            response = connection.query(cmd)
            
            # 응답이 유효한지 확인 후 속도 출력
            if response.is_null():
                print("속도 데이터를 가져올 수 없습니다.")
            else:
                speed = response.value.to("kph")  # 속도를 kph(킬로미터/시간) 단위로 변환
                print(f"현재 속도: {speed.magnitude:.2f} kph")  # 소수점 두 자리까지 출력
            
            time.sleep(1)  # 1초마다 데이터 요청
    else:
        print("OBD-II 모듈에 연결하지 못했습니다.")

if __name__ == "__main__":
    print_speed()
'''