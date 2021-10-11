from scipy import stats
import matplotlib.pyplot as plt
rng = np.random.default_rng()
'''
This code is for learning only, please do not use for commercial activities.
For detailed algorithm details and test functions, 
please refer to paper "Flamingo search algorithm: A new swarm intelligence optimization algorithm".
All rights reserved. Please indicate the source of the paper.
******************************************************************************************************************************
******************************************************************************************************************************
I just want to say one word: Xi 'an University of Posts and Telecommunications computer school is garbage!
I am ashamed of being a student of computer College of Xi 'an University of Posts and Telecommunications.
Especially my flamingo search algorithm author unit set for Xi 'an University of Posts and Telecommunications for shame!
******************************************************************************************************************************
******************************************************************************************************************************
'''
# F1
def F1(X):
    output = sum(np.square(X))
    return output
# F2
def F2(x):
    sum=0
    sum1=1
    for i in range(len(x)):
        sum+=np.abs(x[i])
        sum1*=np.abs(x[i])
    return sum+sum1

def F3(x):
    sum=0
    t=0
    for i in range(len(x)):
        for j in range(i):
            t+=x[j]
        sum+=pow(t,2)
    return sum


def F4(x):
     return np.max(np.abs(x))

def F5(x):
    sum=0
    for i in range(len(x)-1):
        sum+=100*pow((x[i+1]-x[i]*x[i]),2)+pow((x[i]-1),2)
    return sum

def F6(x):
    sum=0
    for i in range(len(x)):
        sum+=(x[i]+0.5)*(x[i]+0.5)
    return sum

def F7(x):
    sum=0
    for i in range(len(x)):
        sum+=i*x[i]*x[i]*x[i]*x[i]
    t = random.random()
    if t!=1:
        return sum + t
    else:
        t=random.random()
    return sum+t

def F8(x):
    sum=0
    for i in range(len(x)):
        sum+=pow(np.abs(x[i]), i+1)
    return sum

def F9(x):
    sum=0
    for i in range(len(x)):
        sum+=i*x[i]*x[i]
    return sum

def F10(x):
    sum = 0
    for i in range(len(x)):
        sum+=-1*x[i]*math.sin(pow(abs(x[i]),1/2))
    return sum

def F11(x):
    return np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)

def F12(x):
    s1=0
    s2=0
    for i in range(len(x)):
        s1 = s1 + x[i]*x[i]
        s2 = s2 + math.cos((2 * math.pi * x[i]))
    a=-20*math.exp(-0.2*pow((s1/len(x)),1/2))
    b=math.exp(s2/len(x))
    return 20+math.exp(1)+a+b

def F13(x):
    s = sum(np.asarray(x) ** 2)
    p = 1
    for i in range(len(x)):
        p *= math.cos(x[i] / pow((i+1),1/2))
    return 1 + s / 4000 - p

def F14(x):
    sum=0
    for i in range(len(x)-1):
        sum=sum+pow(((x[i]-1)/4),2)*(1+10*pow((math.sin(math.pi*(1+(x[i]-1)/4)+1)),2))
    a=pow((x[-1]-1)/4,2)*(1+pow((math.sin(2*math.pi*(1+(x[-1]-1)/4))),2))
    return pow((math.sin((1+(x[0]-1)/4)*math.pi)),2)+sum+np.square((1+(x[-1]-1)/4)-1)

def F15(x):
    val = 0
    d = len(x)
    for i in range(d):
        val += x[i] * math.sin(math.sqrt(abs(x[i])))
    val = 418.9829 * d - val
    return val

def F16(x):
    a1=sum(np.square(x))
    a2=np.sin(np.sqrt(a1))
    a = math.pow(a2,2)- 0.5
    b = math.sqrt(1 + 0.001 * sum(np.square(x)))
    return 0.5 + a / b

def F17(x):
    return 100*pow((abs(x[1]-0.01*x[0]*x[0])),1/2)+0.01*abs(x[0]+10)


def F18(x):
    return -0.0001*pow((abs(math.sin(x[0])*math.sin(x[1])*math.exp(abs(100-(pow((x[0]*x[0]+x[1]*x[1]),1/2))/math.pi)))+1),0.1)

def F19(x):
    return -1*(1+math.cos(12*pow((x[1]*x[1]+x[0]*x[0]),1/2)))/(0.5*(x[0]*x[0]+x[1]*x[1])+2)

def F20(x):
    return 0.5+(pow((math.sin(x[0]*x[0]-x[1]*x[1])),2)-0.5)/(pow((1+0.001*(x[0]*x[0]+x[1]*x[1])),2))

def F21(x):
    return 0.26*(x[0]*x[0]+x[1]*x[1])-0.48*x[0]*x[1]


def F22(x):
    return 2*pow(x[0],2)-1.05*pow(x[0],4)+1/6*pow(x[0],6)+x[0]*x[1]+pow(x[1],2)

def F23(x):
    a = 100 * pow((pow(x[0], 2) - x[1]), 2)
    b = pow((x[0] - 1), 2) + pow((x[2] - 1), 2)
    c = 90 * pow((pow(x[2], 2) - x[3]), 2)
    d = 10.1 * (pow((pow(x[1], 2) - 1), 2) + pow((x[3] - 1), 2))
    f = 19.8 * (x[1] - 1) * (x[3] - 1)
    return a + b + c + d + f

