import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random


#The experimental data + display
# x=[55, 71, 68, 87, 101, 87, 75, 78, 93, 73]
# y=[91, 101, 87, 109, 129, 98, 95, 101, 104, 93]


w0=0
w1=0
x=np.array([55, 71, 68, 87, 101, 87, 75, 78, 93, 73])
y=np.array([91, 101, 87, 109, 129, 98, 95, 101, 104, 93])

#Unary linear regression function
h = lambda theta_0,theta_1,x: theta_0 + theta_1*x

#least square method 
def least_square(x,y):
    ls_w1=((x*y).mean()-x.mean()*y.mean())/((x**2).mean()-(x.mean())**2)
    ls_w0=y.mean()-ls_w1*x.mean()
    print(ls_w0)
    print(ls_w1)
    # return(w0,w1)


def gradient_descent(x,y):
    w0_new=0
    w1_new=0
    alpha=0.04
    MSE=np.array([])

    for iteration in range(1,30):
        y_pred=np.array([])
        error=np.array([])
        error_x=np.array([])

        gd_w0=w0_new
        gd_w1=w1_new

        for i in x:
            y_pred=np.append(y_pred,(gd_w0+gd_w1*i))
        
        error=np.append(error,y_pred-y)
        error_x=np.append(error_x,error*x)
        MSE_val=(error**2).mean()
        MSE=np.append(MSE,MSE_val)

        w0_new=gd_w0-alpha*np.sum(error)
        w1_new=gd_w1-alpha*np.sum(error_x)
        print(w0_new)
        print(w1_new)
    return(w0_new,w1_new)

def plot_data_and_linear_reg():
    xx = np.linspace(0,21,1000)
    plt.scatter(x, y, marker='o', c='b')
    plt.plot(xx,h(theta_new[0],theta_new[1],xx))
    plt.title('our data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([40,120])
    plt.ylim([75,140])
    plt.show()


least_square(x,y)
gradient_descent(x,y)