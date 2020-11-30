import cupy as cp

class ArraysCollection():
    """ this class is equivalent to list of array, or list of list of array, or list of list... etc. Each object can be 
        handle has if it was an array. For example, one can mutiply an array collection by a scalar, add two array collections, etc... 
    """
    def _checkDataFormat(self, list_of_arrays):
        # Check recursiveley if the list has the good format : all elements must have the same type.
        if type(list_of_arrays) == type([]): # case the list of array is a list
            if len(set([type(elem) for elem in list_of_arrays])) > 1: # chack if all elements has the same type
                return False  
            # then check recursively on each element
            return all([self._checkDataFormat(array) for array in list_of_arrays])
        return type(cp.array([])) == type(list_of_arrays) # if not a list must be an array

    def _map_xp_func_to_data(self,data, func):
        # function that apply one function to each elements. the function must have has only input a cp array
        # and return a cp array has output
        if type(data) == type([]):
            return [self._map_xp_func_to_data(array,func) for array in data]
        return func(data)
    
    def __init__(self, list_of_arrays):
        # create an array collection from list
        self._ptr = 0 
        if not self._checkDataFormat(list_of_arrays):
            raise TypeError("The data given to ArraysCollection Argument have the wrong shape")
        self._arrays = list_of_arrays
    

    def _add_float_to_data(self,data, scalar):
        # add a scalar to each elements
        return self._map_xp_func_to_data(data, lambda x: x + scalar)
    

    def _add_datas(self,data1, data2):
        # add two list elementwise, recursively
        if type(data1) == type([]) and (isinstance(data2, float) or isinstance(data2, int)):
            return self._map_xp_func_to_data(data1, lambda x: x + data2)
        if type(data1) == type([]) and type(data2) == type([]):
            if len(data1) != len(data2):
                raise ValueError("The Arrays Collections must have the same shape")
            return [self._add_datas(d1, d2) for d1, d2 in zip(data1, data2)]     
        if type(data1) == type([]) or type(data2) == type([]):
            raise ValueError("The Arrays Collections must have the same shape")
        return data1 + data2

    def _mult_float_to_data(self,data, scalar):
        # multiply a scalar to each elements
        return self._map_xp_func_to_data(data, lambda x: x * scalar)
    
    def _mult_datas(self,data1, data2):
        # multiply two list elementwise, recursively
        if type(data1) == type([]) and (isinstance(data2, float) or isinstance(data2, int)):
            return self._map_xp_func_to_data(data1, lambda x: x * data2)
        if type(data1) == type([]) and type(data2) == type([]):
            if len(data1) != len(data2):
                raise ValueError("The Arrays Collections must have the same shape")
            return [self._mult_datas(d1, d2) for d1, d2 in zip(data1, data2)]     
        if type(data1) == type([]) or type(data2) == type([]):
            raise ValueError("The Arrays Collections must have the same shape")
        return data1 * data2

    def _ldiv_datas(self,data1, data2):
        # divide two list elementwise, recursively return data1/data2
        if type(data1) == type([]) and (isinstance(data2, float) or isinstance(data2, int)):
            return self._map_xp_func_to_data(data1, lambda x: x / data2)
        if type(data1) == type([]) and type(data2) == type([]):
            if len(data1) != len(data2):
                raise ValueError("The Arrays Collections must have the same shape")
            return [self._ldiv_datas(d1, d2) for d1, d2 in zip(data1, data2)]     
        if type(data1) == type([]) or type(data2) == type([]):
            raise ValueError("The Arrays Collections must have the same shape")
        return data1 / data2

    def _rdiv_datas(self,data1, data2):
        # divide two list elementwise, recursively return data2/data1
        if type(data1) == type([]) and (isinstance(data2, float) or isinstance(data2, int)):
            return self._map_xp_func_to_data(data1, lambda x: data2/x)
        if type(data1) == type([]) and type(data2) == type([]):
            if len(data1) != len(data2):
                raise ValueError("The Arrays Collections must have the same shape")
            return [self._rdiv_datas(d1, d2) for d1, d2 in zip(data1, data2)]     
        if type(data1) == type([]) or type(data2) == type([]):
            raise ValueError("The Arrays Collections must have the same shape")
        return data2/data1
    
    
    def __add__(self, other):
        # for the addition operator
        # if isinstance(other, float) or isinstance(other, int):
        #     return ArraysCollection(self._add_float_to_data(self._arrays, other))
        
        if isinstance(other, float) or isinstance(other, int) or isinstance(other, list):
            return ArraysCollection(self._add_datas(self._arrays, other))
        
        if isinstance(other, ArraysCollection):
            return ArraysCollection(self._add_datas(self._arrays, other._arrays))

        raise TypeError("Unknown Type to add to ArrayCollection")


    def __radd__(self, other):
        # for the addition operator
        return self.__add__(other)

    def __neg__(self):
        # for the - operator
        return self.__mul__(-1)
    
    def __sub__(self, other):
        # for the substraction operator
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        # for the substraction operator
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        # for the multiplication operator

        # if isinstance(other, float) or isinstance(other, int):
        #     return ArraysCollection(self._mult_float_to_data(self._arrays, other))
        if isinstance(other, float) or isinstance(other, int) or isinstance(other, list):
            return ArraysCollection(self._mult_datas(self._arrays, other))
        
        if isinstance(other, ArraysCollection):
            return ArraysCollection(self._mult_datas(self._arrays, other._arrays))

        raise TypeError("Unknown Type to multiply with ArrayCollection")

    def __rmul__(self, other):
        # for the multiplication operator
        return self.__mul__(other)

    def __truediv__(self, other):
        # for the division operator on the left

        # if isinstance(other, float) or isinstance(other, int):
        #     return ArraysCollection(self._mult_float_to_data(self._arrays, other))
        if isinstance(other, float) or isinstance(other, int) or isinstance(other, list):
            return ArraysCollection(self._ldiv_datas(self._arrays, other))
        
        if isinstance(other, ArraysCollection):
            return ArraysCollection(self._ldiv_datas(self._arrays, other._arrays))

        raise TypeError("Unknown Type to multiply with ArrayCollection")

    def __rtruediv__(self, other):
        # for the division operator on the left

        # if isinstance(other, float) or isinstance(other, int):
        #     return ArraysCollection(self._mult_float_to_data(self._arrays, other))
        if isinstance(other, float) or isinstance(other, int) or isinstance(other, list):
            return ArraysCollection(self._rdiv_datas(self._arrays, other))
        
        if isinstance(other, ArraysCollection):
            return ArraysCollection(self._rdiv_datas(self._arrays, other._arrays))

        raise TypeError("Unknown Type to multiply with ArrayCollection")

    def __repr__(self):
        # for display
        return self._arrays.__repr__()

    def __getitem__(self, key):
        # so that array collection can be handled has if it was list
        result = self._arrays[key]
        if type(result) == type([]):
            return ArraysCollection(result)
        else:
            return result
    
    def __next__(self):
        # iterate over elements
        if self._ptr == len(self._arrays):
           raise StopIteration
        result = self.__getitem__(self._ptr)
        self._ptr = self._ptr + 1
        return result

    def __iter__(self):
        self._ptr = 0
        return self
    
    def __len__(self):
        return len(self._arrays)

    
    def _sum(self,data):
        # sum every elements of list, or list of lists, or list of lists of .... Go recursively inside list
        if type(data) == type([]):
            return sum([self._sum(d) for d in data])
        if (isinstance(data, float) or isinstance(data, int)):
            return data   
        if type(data) == type(cp.array([])):
            return cp.sum(data)
        raise TypeError("Type is not supported for array collection")
    
    def sum(self):
        # sum every elements of the array collection Go recursively inside list
        return self._sum(self._arrays)

    def map_xp_func(self, func):
        # map a function of each elements
        return ArraysCollection(self._map_xp_func_to_data(self._arrays, func))
    
    @staticmethod
    def combine(arrCol1, arrCol2):
        # create a new array collection by making a list of two array collections
        return ArraysCollection([arrCol1._arrays, arrCol2._arrays])
    


if __name__ == "__main__":
    a = cp.array([[1,2,3],[4,5,6]])
    b = cp.array([[1,2,3]])
    c = cp.array([1])
    d1 = ArraysCollection([a,b,c])
    d2 = 2 + d1 + 1
    d3 = 3 * d1 * 2
    d4 = 2 - d1 - 2


    print("#####     d1      #####")
    print(d1)
    print("##### d2 = d1 + 3 #####")
    print(d2)
    print("#####   d1 x d2   #####")
    print(d1*d2)
    print()
    print("#####      d1     #####")
    print(d1)
    print("##### d3 = d1 * 6 #####")
    print(d3)
    print("#####   d1 + d3   #####")
    print(d1+d3)
    print()
    print(d1)
    print("#####    d1     #####")
    print(d4)
    print("##### d4 = -d1  #####")
    print(d4.map_xp_func(cp.abs))
    print("##### abs(d4)   #####")

    print('\nsplitting d1')
    print('###############')
    for d in d1:
        print(d)
        print('###############')


    d1 = ArraysCollection([[a],[b],[[c]]])
    d5 = [0,1,2] + d1 + [1,1,1] 
    d6 = [3,1,2] * d1 * [1,2,3]
    #d7 = [1,1,1] / d1 * d1 
    d7 = ([1,1,1] / d1) / (1 / d1)
    d8 = (d1 / (0.5)) - (2*d1)

    print("#####    d1              #####")
    print(d1)
    print("##### d5 = d1 + [1,2,3]  #####")
    print(d5)
    print("##### d6 = d1 * [3,2,6]   #####")
    print(d6)

    print()
    print("#####    d1              #####")
    print(d1)
    print("##### d7 = ones           #####")
    print(d7)
    print("##### d8 = zeros           #####")
    print(d8)
    print("##### d7, d8               #####")
    print(ArraysCollection.combine(d7,d8))
    print(len(ArraysCollection.combine(d7,d8)))

    print()
    print("#####    d1              #####")
    print(d1)
    print("#####    sum(d1)         #####")
    print(d1.sum())
    


