#모듈 임포트
import random

#-----------------------------

#인공 신경망 기본 정보

#신경망 층 개수
n = 4 
#층 별 뉴런 개수
m = [10, 5, 5, 3]
#활성화 함수
def AVfunc(x):
    if x > 1:
        return 1
    elif x > 0:
        return x
    else:
        return 0

#-----------------------------

#인공 신경망 변수
#인공 신경망 랜덤 초기화

#뉴런의 값 저장 변수
neuron = [[random.random() for i in range(m[k])] for k in range(n)]
#가중치 저장 변수
weight = [0] + [[[(1-2*random.random()) for j in range(m[k - 1])] for i in range(m[k])] for k in range(1, n)]
#편향 저장 변수
bias = [[(1-2*random.random()) for i in range(m[k])] for k in range(n)]

#-----------------------------

#입력층 값 입력
neuron[0] = list(map(float, input().split(' ')))

#순전파
for k in range(1, n):
    for i in range(m[k]):
        sum = 0
        for j in range(m[k - 1]):
            #print(str(k) + ", " + str(i) + ', ' + str(j))
            sum += neuron[k-1][j] * weight[k][i][j]
        neuron[k][i] = AVfunc(sum + bias[k][i])


#인공 신경망 값 확인
for k in range(0, n):
    for i in range(m[k]):
        print(str(k) + " , " + str(i) + " : " + str(neuron[k][i]))
        
#-----------------------------