import cupy as cp
import numpy as np
import threading
import time

# GPU 작업 함수
def gpu_sum(data):
    print("GPU 작업 시작")
    # CuPy를 사용하여 GPU에서 배열의 합계 계산
    gpu_array = cp.array(data)
    result = cp.sum(gpu_array)
    # GPU 결과를 CPU로 가져오기
    result = cp.asnumpy(result)
    print(f"GPU 작업 완료: 합계 = {result}")

# CPU 작업 함수1: 평균 계산
def cpu_mean(data):
    print("CPU 작업 1 (평균) 시작")
    # NumPy를 사용하여 CPU에서 배열의 평균 계산
    result = np.mean(data)
    print(f"CPU 작업 1 (평균) 완료: 평균 = {result}")

# CPU 작업 함수2: 표준편차 계산
def cpu_std(data):
    print("CPU 작업 2 (표준편차) 시작")
    # NumPy를 사용하여 CPU에서 배열의 표준편차 계산
    result = np.std(data)
    print(f"CPU 작업 2 (표준편차) 완료: 표준편차 = {result}")

def main():
    # 데이터 배열 생성
    data = np.random.rand(1000000)  # 1백만 개의 랜덤 값으로 배열 생성

    # GPU 작업을 위한 스레드
    gpu_thread = threading.Thread(target=gpu_sum, args=(data,))
    gpu_thread.start()

    # CPU 작업을 위한 스레드들
    cpu_thread1 = threading.Thread(target=cpu_mean, args=(data,))
    cpu_thread2 = threading.Thread(target=cpu_std, args=(data,))
    cpu_thread1.start()
    cpu_thread2.start()

    # 모든 스레드가 완료될 때까지 기다림
    gpu_thread.join()
    cpu_thread1.join()
    cpu_thread2.join()

if __name__ == "__main__":
    main()
