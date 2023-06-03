#모듈 임포트
import random
import math

#-----------------------------

#인공 신경망 기본 정보

#신경망 층 개수
n = 4 
#층 별 뉴런 개수
m = [10, 5, 5, 3]
#활성화 함수
#def AVfunc(x):
#    if x > 0:
#        return x
#    else:
#        return 0
##활성화 함수의 도함수
#def dAVfunc(x):
#    if x > 0:
#        return 1
#    else:
#        return 0
#

#활성화 함수
def AVfunc(x):
    return (math.pow(math.e, x) / (math.pow(math.e, x) + 1))
#활성화 함수의 도함수
def dAVfunc(x):
    return AVfunc(x) * (1 - AVfunc(x))

#오차함수
def cost(correct, output):
    sum = 0
    if(len(correct) != len(output)):
        raise "not same length"
    for i in range(len(correct)):
        sum += (correct[i]-output[i])**2
    return sum


#-----------------------------

#인공 신경망 변수
#인공 신경망 랜덤 초기화

#뉴런의 값 저장 변수
neuron = [[random.random() for i in range(m[k])] for k in range(n)]
#가중치 저장 변수
weight = [0] + [[[(1-2*random.random()) for j in range(m[k - 1])] for i in range(m[k])] for k in range(1, n)]
#편향 저장 변수
#bias = [[(0) for i in range(m[k])] for k in range(n)]
bias = [[(1-2*random.random()) for i in range(m[k])] for k in range(n)]
#오차 저장 변수
error = [[random.random() for i in range(m[k])] for k in range(n)]
#말단이 [i][j] 인 가중치의 총합 저장 변수
weightsum= [[random.random() for i in range(m[k])] for k in range(n)]

#학습률
a = 0.01

#학습 횟수
repeat_cnt = 100000

#-----------------------------

#값 확인
def print_info():
    
    #인공 신경망 값 확인
    for k in range(0, n):
        for i in range(m[k]):
            print(str(k) + " , " + str(i) + " : " + str(neuron[k][i]))

    #오차함수 계산
    print(cost(outputCorrect, neuron[n-1]))

    #오차 값 확인
    #for k in range(0, n):
    #    for i in range(m[k]):
    #        print(str(k) + " , " + str(i) + " : " + str(error[k][i]))

#------------------------------

#입력층 값 입력
neuron[0] = list(map(float, input().split(' ')))
#출력층 예상값 입력
outputCorrect = list(map(float, input().split(' ')))

#반복 학습
for i in range(0, repeat_cnt):

    #순전파
    for k in range(1, n):
        for i in range(m[k]):
            sum = 0
            for j in range(m[k - 1]):
                #print(str(k) + ", " + str(i) + ', ' + str(j))
                sum += neuron[k-1][j] * weight[k][i][j]
            neuron[k][i] = AVfunc(sum + bias[k][i])

    #출력층 오차 구하기
    for i in range(0, m[n-1]):
        error[n-1][i] = (outputCorrect[i] - neuron[n-1][i])**2

    #가중치 합 구하기
    for k in range(1, n):
        for i in range(0, m[k]):
            weightsum[k][i] = 0
            for j in range(0, m[k-1]):
                weightsum[k][i] += weight[k][i][j]

    #오차 구하기
    for k in range(n - 2, 0, -1): 
        for i in range(0, m[k]):
            error[k][i] = 0
            for j in range(0, m[k+1]):
                error[k][i] += error[k+1][j] * ((weight[k+1][j][i])/weightsum[k+1][j])

    print_info()

    #-----------------------------

    #경사하강법
    for k in range(1, n): 
        for i in range(0, m[k]):
            for j in range(0, m[k-1]):
                weight[k][i][j] += - a * error[k][i] * dAVfunc(neuron[k][i]) * neuron[k-1][j]

