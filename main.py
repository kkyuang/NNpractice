#모듈 임포트
import random
import math

#인공신경망 기본 함수

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

#인공신경망 클래스
class neuralNetwork:


    #초기화
    def __init__(self, m, learningrate):


        #학습률
        self.lr = learningrate
        #신경망 층 개수
        self.n = n
        #층 별 뉴런 개수
        self.m = m

        #-----------------------------

        #인공 신경망 변수
        #인공 신경망 랜덤 초기화

        #뉴런의 값 저장 변수
        self.neuron = [[random.random() for i in range(self.m[k])] for k in range(n)]
        #가중치 저장 변수
        self.weight = [0] + [[[(1-2*random.random()) for j in range(self.m[k - 1])] for i in range(m[k])] for k in range(1, n)]
        #편향 저장 변수
        #bias = [[(0) for i in range(m[k])] for k in range(n)]
        self.bias = [[(1-2*random.random()) for i in range(self.m[k])] for k in range(n)]
        #오차 저장 변수
        self.error = [[random.random() for i in range(self.m[k])] for k in range(n)]
        #말단이 [i][j] 인 가중치의 총합 저장 변수
        self.weightsum= [[random.random() for i in range(self.m[k])] for k in range(n)]

        pass
        
    
    #학습
    def train(self, inputData, outputCorrect):

        self.neuron[0] = inputData
        #순전파
        for k in range(1, self.n):
            for i in range(self.m[k]):
                sum = 0
                for j in range(self.m[k - 1]):
                    sum += self.neuron[k-1][j] * self.weight[k][i][j]
                self.neuron[k][i] = AVfunc(sum + self.bias[k][i])

        #출력층 오차 구하기
        for i in range(0, self.m[n-1]):
            self.error[self.n-1][i] = (outputCorrect[i] - self.neuron[self.n-1][i])**2

        #가중치 합 구하기
        for k in range(1, self.n):
            for i in range(0, self.m[k]):
                self.weightsum[k][i] = 0
                for j in range(0, self.m[k-1]):
                    self.weightsum[k][i] += self.weight[k][i][j]

        #오차 구하기
        for k in range(self.n - 2, 0, -1): 
            for i in range(0, self.m[k]):
                self.error[k][i] = 0
                for j in range(0, self.m[k+1]):
                    self.error[k][i] += self.error[k+1][j] * ((self.weight[k+1][j][i])/self.weightsum[k+1][j])

        #경사하강법
        for k in range(1, self.n): 
            for i in range(0, self.m[k]):
                for j in range(0, self.m[k-1]):
                    self.weight[k][i][j] -= self.lr * self.error[k][i] * dAVfunc(self.neuron[k][i]) * self.neuron[k-1][j]
            pass
        
        #오차함수 출력
        print(str(cost(outputCorrect, self.neuron[self.n-1])))

    #질의
    def query(self, inputData):
        self.neuron[0] = inputData

        #순전파
        for k in range(1, self.n):
            for i in range(self.m[k]):
                sum = 0
                for j in range(self.m[k - 1]):
                    sum += self.neuron[k-1][j] * self.weight[k][i][j]
                self.neuron[k][i] = AVfunc(sum + self.bias[k][i])
        
        #출력
        for k in range(0, self.n):
            for i in range(self.m[k]):
                print(str(k) + " , " + str(i) + " : " + str(self.neuron[k][i]))
        pass

#-----------------------------

#인공 신경망 기본 정보

#신경망 층 개수
n = 4 
#층 별 뉴런 개수
m = [2, 2, 2, 1]
#학습 횟수
repeat_cnt = 10000

#입력층 값 입력
#inputData = list(map(float, input().split(' ')))
#출력층 예상값 입력
#outputCorrect = list(map(float, input().split(' ')))

#-----------------------------

nn = neuralNetwork(m, 0.001)

for i in range(0, repeat_cnt):
    a = random.random()
    b = random.random()
    nn.train([a, b], [(a + b) / 2])

a1 = float(input())
b1 = float(input())

nn.query([a1, b1])