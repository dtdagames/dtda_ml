class_name MLTools

# type : 0 KNN, 1 linear reg, 2 logistic reg, 3 SVM
func _get_perf(y_pred, y_test, type):
	# convert >0.5 to 1 from prediction for linear regression
	if type == 1:
		y_pred = _normalize_int(y_pred)
	# convert 0 to -1 from test for SVM
	if type == 3:
		y_pred = _normalize_negative(y_pred)
		y_test = _normalize_negative(y_test)
	
	var correctly_classified = 0
	var count = 0
	for i in y_pred.size() :
		if y_test[i] == y_pred[i]:
			correctly_classified += 1
		count += 1
	
	return snapped(float(correctly_classified) / float(count) *100, 0.01)

# convert array from float to int
func array_to_int(arr):
	var tempData = []
	for i in arr:
		tempData.push_back(int(i))
	return tempData

# convert 0 to -1 from array
func _normalize_negative(tempData):
	var newData = []
	for row in tempData:
		if row == 0:
			row = -1
		newData.push_back(row)
	return newData
# convert >0.5 to 1 from array
func _normalize_int(tempData):
	var newData = []
	for row in tempData:
		if row > 0.5:
			row = 1
		else:
			row = 0
		newData.push_back(row)
	return newData
# convert value to -1/1 from array
func _sign_array(x):
	var matrix = []
	for row in x:
		if row>0:
			row = 1
		elif row<0:
			row = -1
		matrix.push_back(row)
	return matrix

# return array with specific column
func _getVariable(tempData, tempColumnId):
	var newData = []
	for row in tempData:
		newData.push_back(row[tempColumnId])
	return newData

# return array without specific column
func _dropVariable(tempData, tempColumnId):
	var newData = []
	for i in tempData.size():
		newData.push_back([])
		for u in tempData[i].size():
			if u != tempColumnId:
				newData[i].push_back(tempData[i][u])
	return newData

# return array of zeros
func _array_zeros(n):
	var tempW = []
	for i in n:
		tempW.push_back(0)
	return tempW

# return substract of two arrays
func _substract_arrays(x1, x2):
	var matrix = []
	for i in x1.size():
		matrix.push_back(x1[i] - x2[i])
	return matrix
# return substract of array and const
func _sub_arrays_const(x1, b):
	var matrix = []
	for i in x1.size():
		matrix.push_back(x1[i] - b)
	return matrix

# add array by constant
func _add_arrays_const(x1, b):
	var matrix = []
	for i in x1.size():
		matrix.push_back(x1[i] + b)
	return matrix

# mutliply rows of array by coef
func _multiply_array_coef(x1, b):
	var matrix = []
	for i in x1.size():
		matrix.push_back(x1[i] * b)
	return matrix
# divide rows of array by coef
func _divide_array_coef(x1, b):
	var matrix = []
	for i in x1.size():
		matrix.push_back(x1[i] / b)
	return matrix

# divide coef by rows 
func _divide_inverse_array_coef(x1, b):
	var matrix = []
	for i in x1.size():
		matrix.push_back(b / x1[i])
	return matrix

# return rows of array by exp
func _exp_array_(x1):
	var matrix = []
	for i in x1.size():
		matrix.push_back(exp(x1[i]))
	return matrix

# return dot product of two arrays
func _dot_product(x1, x2):
	var matrix = []
	for i in x1.size():
		var res = 0
		for u in x1[0].size():
			res += x1[i][u] * x2[u]
		matrix.push_back(res)
	return matrix
# return dor product of array and const
func _dot_product_simple(x1, x2):
	var matrix = []
	var res = 0
	for i in x1.size():
		res += x1[i] * x2[i]
	matrix.push_back(res)
	return matrix

# transpose array
func _transpose_array(x):
	var matrix = []
	for i in x[0].size():
		matrix.push_back([])
		for u in x.size():
			matrix[i].push_back(x[u][i])
	return matrix
# transpose 1D array
func _transpose_simple_array(x):
	var matrix = []
	for i in x.size():
		matrix.push_back(x[i])
	return matrix

# return sum of all rows
func _sum_array(x):
	var total = 0
	for i in x.size():
		total += x[i]
	return total
