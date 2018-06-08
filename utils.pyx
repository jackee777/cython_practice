from cpython cimport array
from libc.stdlib cimport srand, rand, RAND_MAX
from libc.time cimport time
import copy

cdef extern from "cutils.h":
     double _c_sum "c_sum"(double a, double b)
cdef extern from "cutils.h":
     double _c_div "c_div"(double a, double b)
cdef extern from "cutils.h":
     double _c_prod "c_prod"(double a, double b)
cdef extern from "cutils.h":
     double _c_exp "c_exp"(double a)


## task1
#sum matrix a and b
def msum(a, b):
    cdef:
        double i, j
        
    if get_shape(a)[0] == 1 and get_shape(b)[0] == 1:
        return [_c_sum(i, j) for i, j in zip(a, b)]
    elif get_shape(a)[0] == 1 and get_shape(b)[1] != 1:
        return [[_c_sum(i, j) for i, j in zip(m, n)]
                for (m, n) in zip([[i for i in a]] * len(a), b)]
    elif get_shape(a)[1] != 1 and get_shape(b)[0] == 1:
        return [[_c_sum(i, j) for i, j in zip(m, n)]
                for (m, n) in zip(a, [[i for i in b]] * len(b))]
    return [[_c_sum(i, j) for i, j in zip(m, n)] for (m, n) in zip(a, b)]


cdef double prod(double[:] a, double[:] b):
    cdef double sums = 0
    for i in range(len(a)):
        sums += _c_prod(a[i], b[i])
    return sums       


cdef double exp_sum(double[:] a, double sums = 0):
    for i in range(len(a)):
        sums += _c_exp(a[i])
    return sums 


def part_prod_matrix(mat):
    cdef:
        int i
        array.array a = array.array('d', mat[0])
    b = mat[1]
    return [prod(a, array.array('d', j)) for j in b]


# product of matrix a and b
def mprod(a, b):
    assert get_shape(a)[1] == get_shape(b)[0], "incorrect shape"
    tb = b
    b = [mpose(b)] * get_shape(a)[0]
    
    if get_shape(a)[0] == 1 and get_shape(tb)[1] == 1:
        return part_prod_matrix([a, b])
    elif get_shape(a)[0] == 1 and get_shape(tb)[1] != 1: #too slow, need revision
        return  [part_prod_matrix([a, j]) for j in b][0]
    elif get_shape(tb)[1] == 1: #I optimise only this method for using
        return mpose([prod(array.array('d', i), array.array('d', j)) for i, j in zip(a, b)])
    return [part_prod_matrix([i, j]) for i, j in zip(a, b)]


def get_shape(a):
    if type(a[0]) == float:
        return (1, len(a))
    else:
        return (len(a), len(a[0]))


#transpose a matrix
def mpose(a):
    A = copy.deepcopy(a)
    cdef:
        double i
    prod = []
    if get_shape(A)[0] == 1:
        return [[i] for i in A]
    elif get_shape(A)[1] == 1:
        return [j[0] for j in A]
    else:
        return list(map(list, zip(*A)))


#correspond to float
def relu(a): 
    cdef:
        double i
        array.array A
    if get_shape(a)[1] == 1:
        A = array.array('d', mpose(a))
        return mpose([max(0, i) for i in A])
    A = array.array('d', a)
    return [max(0, i) for i in A]


def softmax(a):
    cdef:
        double i
        double sums
        array.array A = array.array('d', a)
    sums = exp_sum(A)
    assert get_shape(a)[0] == 1, "only one dimension"
    return [_c_div(_c_exp(i), sums) for i in A]


##task2
def normalization(image, double max_value):
    cdef array.array a = array.array('d', image)
    return [_c_div(i, max_value) for i in a]


def read_file(filename):
    max_value = 1
    image = None
    
    with open(filename) as f:
        head = f.readline()
        shape = f.readline().split()
        max_value = int(f.readline().replace("\n",""))
        image = [line.split() for line in f]
        image = list(map(int, sum(image, [])))
    return (<double>max_value, image)
 
    
def get_data(filename):
    cdef:
        double max_value = 1
        
    max_value, image = read_file(filename)
    image = normalization(image, max_value)
    return image


def argmax(a):
    cdef array.array A = array.array('d', a)
    assert get_shape(a)[0] == 1, "only one dimension"
    return part_argmax(A)


cdef int part_argmax(double[:] a):
    cdef:
        int index = 0
        int i
        double max_value = -10e7
        
    for i in range(len(a)):
        if max_value < a[i]:
            max_value = a[i]
            index = i
    return index + 1


##task3
#make array that has only 1 value which is 1 and other values which are 0
#make sigma
def one_value_array(index, length, pos_nega=-1):
    output =  [0.0] * length
    output[index-1] = 1.0 * pos_nega
    return output


cdef part_backward(double[:] a, double[:] b):
    cdef double i, j
    return [0.0 if j < 0 else i for i, j in zip(a, b)]


def backward(a, b):
    cdef:
        double i
        array.array A
        array.array B
    assert get_shape(a) == get_shape(b), "incorrect shape"
    if get_shape(a)[1] == 1:
        A = array.array('d', mpose(a))
        B = array.array('d', mpose(b))
        return mpose(part_backward(A,B))
    A = array.array('d', a)
    B = array.array('d', b)
    return part_backward(A,B)


cdef part_sign(double[:] a, double ups0):
    cdef double i
    cdef array.array temp = array.array("d", 
                                        [_c_prod(-1.0, ups0) if i < 0 else i for i in a])
    return [_c_prod(1.0, ups0) if i > 0 else i for i in temp]


def sign(a, ups0 = 0.1):
    cdef array.array A = array.array('d', a)
    return part_sign(A, ups0)


cdef double cal_min_max(double i, double min_value, double denom):
    return _c_div(_c_sum(i, _c_prod(-1.0, min_value)), denom)
    

cdef part_min_max(double[:] a, double min_value, double denom, double rate):
    cdef double i
    return [<int>_c_prod(cal_min_max(i, min_value, denom), rate) for i in a]


def min_max(a, rate = 255):
    cdef:
        double max_a = max(a)
        double min_a = min(a)
        double denom = max_a - min_a
        array.array A = array.array('d', a)
    return part_min_max(A, min_a, denom, rate)


cdef part_cut_norm(double[:] a, double rate):
    cdef double i
    output = []
    for i in a:
        if i < 0:
            output.append(0)
        elif i > 1:
            output.append(<int>rate)
        else:
            output.append(<int>_c_prod(i, rate))
    return output


# cut values that are larger than 1 or smaller than 0
def cut_norm(a, rate = 255):
    cdef array.array A = array.array('d', a)
    return part_cut_norm(A, rate)


def write_data(filename, image):
    cdef int i
    with open(filename, "w") as f:
        f.writelines("P2\n")
        f.writelines("32 32\n")
        f.writelines("255\n")
        for i in range(32): #write() function moves incorrectly
            f.writelines(" ".join(map(str, image[i*32:(i+1)*32])) + "\n")
    return 0


# convert int type to double type
def convert_double(a):
    return [array.array("d", i) for i in a]
    

# make random array which values are -1 or 1
def random(length):
    cdef double mean = RAND_MAX /2
    srand(time(NULL))
    return [1 if rand() > mean else -1 for i in range(length)]


    