import numpy as np
import matplotlib.pyplot as plt
def sample(x,y):
	#number of observations
	n= np.size(x)

	#mean of x and y vector
	m_x= np.mean(x)
	m_y= np.mean(y)

	#calucating cross-deviation and deviation
	s_xy = np.sum(y*x) - n * m_y * m_x
	s_xx = np.sum(x*x) - n * m_x * m_x

	#calculating regression coefficient
	b_1 = s_xy / s_xx
	b_0 = m_y - b_1 * m_x

	return(b_0,b_1)

def plot_reg(x , y , b):

	#plotting the actual points as scatter plots
	plt.scatter(x,y,color = "m", marker="o" , s=30)

	#predicted response vector
	y_pred = b[0] + b[1] * x

	#plotting the regression line 
	plt.plot(x, y_pred , color = "g")

	#putting labels
	plt.xlabel('x')
	plt.ylabel('y')

	#show plot
	plt.show()

def main():

	#observation data
	x = np.array([0,1,2,3,4,5,6,7,8,9])
	y = np.array([1,3,2,5,7,8,8,9,10,12])

	#estimating coefficients
	b = sample(x,y)
	print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1]))
	
	#plotting regression line
	plot_reg(x,y,b)

if __name__ == "__main__":
	main()
