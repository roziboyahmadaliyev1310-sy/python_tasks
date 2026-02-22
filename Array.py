import numpy as np

# print(np.__version__)

# main_list=[1,2,3,4,5]
# main_list=main_list*2
# print(main_list)


# array=np.array([1,2,3,4,5])

# array*=2

# print(array)

# print(type(array))

# working with ndimensional array

# array=np.array([[[1,2,3],[4,5,6],[7,8,9]],
#                 [[0,2,3],[4,5,6],[7,8,9]],
#                 [[1,2,3],[4,5,6],[7,8,9]]])

# print(array.ndim)
# print(array.shape)
# print(array[0,1,2])
# print(array.size)
# age=array[0,0,1],array[1,0,0],array[1,0,0],array[2,1,2]
# print(age)

# array=np.array([[1,2,3,4],
#                 [5,6,7,8],
#                 [9,10,11,12],
#                 [13,14,15,16]])

# # array[start:end:step]

# print(array[::-2],"\n")

# print(array[:,::-1],"\n")

# print(array[0,0:2])



# scaler arithmetic

# array=np.array([1,2,3])

# print(array+3)
# print(array-2)
# print(array*4)
# print(array/2)
# print(array**5)
# print(array%2!=0)


# Vectorized mathematical functions

# radious=np.array([1,2,3])

# print(np.sqrt(radious))

# print(np.pi*radious**2)


# x=np.array([1,4,9,16,25])
# y=np.array([3,5,7,10,11])

# print(np.add(x,y),"\n")
# print(np.sqrt(x))
# print(np.subtract(x,y))
# print(np.mean(x))

# print(np.median(y))

# print(np.std(x))
# print(np.var(y))

# grades=np.array([60,76,80,90,54,49,100])

# result=np.where(grades>=60,"pass","fail")
# average=np.mean(grades)
# print("Average grade: ", average,"\n")
# print("Passed: ",result)


# # np.dot() is used for matrix multiplicatio and vector multiplication. It multiples two arrays.
# print(np.dot(x,y))   


# logorifm=np.array([1,10,100,1000])
# print(np.log(logorifm))

# z=np.array([3,5,7,10,11])
# print(np.exp(z))

# maxx=np.array([3,101,5,7,100,10,11])
# print(np.argmax(maxx))

# end=np.array([-8,3,5,7,10])
# print(np.clip(end,0,5))



# Elementwise operations

# array1=np.array([2,3,4,5])
# array2=np.array([10,11,12,13])

# print(array2+array1)
# print(array2-array1)
# print(array2*array1)
# print(array2/array1)
# print(array2**array1)
# print(array2%array1!=0)

# Comparison operations


# scores=np.array([91,50,100,85,74])

# scores[scores<60]=0
# print(scores)

# brodcasring 

# x=np.array([[1,2,3,4],
#             [5,6,7,8],
#             [9,10,11,12],
#             [13,14,15,16]])

# y=np.array([10,20,30,40])

# print(x+y)

# z=np.array([[1,2,3,4],
#             [5,6,7,8],
#             [9,10,11,12]])

# zeta=np.array([30,20,10])

# zeta=zeta.reshape(3,1)

# print(z+zeta)


# agragade functions

# sum
# array=np.array([[1,2,3,4,5],
#                [6,7,8,9,10]])
# print(np.sum(array))

# # mean
# print(np.mean(array))

# # max
# print(np.max(array))

# # min
# print(np.min(array))

# # std
# print(np.std(array))

# # var
# print(np.var(array))

# print(np.sum(array, axis=1))

# working on random numbers

rng=np.random.default_rng()

print(rng.integers(low=1,high=30,size=(3,2)))


print(np.random.uniform(1,10, size=3))


array=np.array([1,2,3,4,5])
rng.shuffle(array)
print(array)

fruits=np.array(["apple","banana","cherry","strawberry","coconut"])

fruit=rng.choice(fruits,size=3, replace=False)
print(fruit)