import numpy as numpy

#print(numpy.random.randn(5))

def testFunction(array):
	value = 0;
	for number in array:
		value += number
	print(value)
	return [value, "haha"]

a = [1,2,3]

result = testFunction(a)
print("The result is ",result[1])

