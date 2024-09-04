#include <IRROBOT_EZController.h>
#include <MightyZap.h>

#define ID_NUM 1
#define SPEED 1023
IRROBOT_EZController Easy(&Serial1);

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

