import pandas as pd
from scipy.spatial import KDTree
import time
import requests
from bs4 import BeautifulSoup

# CSV 파일을 DataFrame으로 읽어옵니다.
def load_data(csv_file):
    return pd.read_csv(csv_file, encoding='cp949')

# KDTree를 사용하여 가장 가까운 위치를 찾는 함수
def find_nearest_speed_limit(input_lat, input_lon, data):
    coords = data[['lon', 'lat']].to_numpy()
    tree = KDTree(coords)
    distance, index = tree.query([input_lat, input_lon])
    nearest_speed_limit = data.iloc[index]['limit']
    return nearest_speed_limit

# 서버에서 GPS 좌표를 HTML에서 추출하는 함수
def fetch_gps_coordinates_from_server(server_url):
    try:
        response = requests.get(server_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        location_text = soup.find('body').get_text()

        lat_start = location_text.find('위도:') + len('위도:')
        lon_start = location_text.find('경도:')
        lat_end = location_text.find(',', lat_start)
        lon_end = len(location_text)

        latitude = float(location_text[lat_start:lat_end].strip())
        longitude = float(location_text[lon_start + len('경도:'):lon_end].strip())

        return latitude, longitude
    except requests.RequestException as e:
        print(f"서버 요청 중 오류 발생: {e}")
        raise
    except ValueError as e:
        print(f"데이터 처리 중 오류 발생: {e}")
        raise

# 위치 및 제한 속도 추적을 위한 함수
def track_speed_limit(server_url, data):
    previous_latitude = None
    previous_longitude = None
    previous_time = None

    current_time = time.time()
    input_latitude, input_longitude = fetch_gps_coordinates_from_server(server_url)
    if input_latitude != previous_latitude or input_longitude != previous_longitude:
        nearest_speed_limit = find_nearest_speed_limit(input_latitude, input_longitude, data)
        if previous_time is not None:
            time_difference = current_time - previous_time
        previous_latitude = input_latitude
        previous_longitude = input_longitude
        previous_time = current_time
    return nearest_speed_limit


