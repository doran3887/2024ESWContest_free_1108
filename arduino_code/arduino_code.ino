#include <IRROBOT_EZController.h>
#include <MightyZap.h>

#define ID_NUM 1
#define POSITION1 0
#define POSITION2 2730
#define SPEED 1023
#define MOVEMENT_DELAY 2000 // 2초 동안 움직임이 완료되도록 딜레이 설정
IRROBOT_EZController Easy(&Serial1);

int positions[3] = {0, 0, 0}; // 이전 위치를 저장할 배열
int currentIndex = 0; // 현재 인덱스를 추적할 변수

void setup() {
  Serial.begin(9600);
  Easy.MightyZap.begin(9600); // MightyZap 통신 속도 설정
  Easy.MightyZap.GoalCurrent(ID_NUM, 800);
  
  // 초기 조건 설정
  Easy.MightyZap.GoalPosition(ID_NUM, POSITION1);
  Easy.MightyZap.GoalSpeed(ID_NUM, SPEED);
  Easy.MightyZap.forceEnable(ID_NUM, 1); // 모터 활성화
}

void loop() {
  if (Serial.available() > 0)
  {
    String data = Serial.readStringUntil('\n');
    int position = data.toInt();
    
    positions[currentIndex] = position;
    
    // 위치 이동
    Easy.MightyZap.GoalPosition(ID_NUM, position);
    Easy.MightyZap.GoalSpeed(ID_NUM, SPEED);
  }
}

