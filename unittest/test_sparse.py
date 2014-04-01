"""
Test written for nose.
"""
import operator
import itertools
import numpy as np
from sparse import SparseArray

from nose.tools import raises


def test_getitem_basic_slicing_integer():
    """
    Test :meth:`SparseArray.__getitem__`.
    
    Index single value in vector.
    """
    a = np.arange(12)
    s = SparseArray.fromdense(a)
    assert(s[3] == a[3])
    
    a = np.arange(12)
    s = SparseArray.fromdense(a)
    assert(s[-3] == a[-3])

@raises(IndexError)
def test_getitem_basic_slicing_integer_out_of_bounds():
    """
    Test :meth:`SparseArray.__getitem__`.
    
    Index single value in vector that is out of bounds.
    """
    a = np.arange(12)
    s = SparseArray.fromdense(a)
    s[20]

def test_getitem_basic_slicing_slice():
    """
    Test :meth:`SparseArray.__getitem__`.
    
    Index using single slice."""
    a = np.arange(10)
    s = SparseArray.fromdense(a)
    i = slice(3, 9, 2)
    np.testing.assert_array_equal(s[i], a[i])
    
    """Index using single slice."""
    a = np.arange(10)
    s = SparseArray.fromdense(a)
    i = slice(-1, 9, 2)
    np.testing.assert_array_equal(s[i], a[i])
    
def test_getitem_basic_slicing_tuple():    
    """
    Test :meth:`SparseArray.__getitem__`.
    
    Index single value in 2D array."""
    a = np.arange(12).reshape(3,4)
    s = SparseArray.fromdense(a)
    assert(s[1,1] == a[1,1])
    
    """Index single value in 3D array."""
    a = np.arange(12).reshape(2,3,2)
    s = SparseArray.fromdense(a)
    assert(s[1,1,1] == a[1,1,1])
    
    """Index a vector in a 2D array."""
    a = np.arange(12).reshape(3,4)
    s = SparseArray.fromdense(a)
    np.testing.assert_array_equal(s[1], a[1])


def test_getitem_basic_slicing_list():
    """
    Test :meth:`SparseArray.__getitem__`.
        
    Select items from a vector using a list.
    """
    a = np.arange(10)
    s = SparseArray.fromdense(a)
    i = [1,4,6]
    np.testing.assert_array_equal(s[i], a[i])
    
    """Select rows from a 2D array using a list."""
    a = np.arange(12).reshape(3,4)
    s = SparseArray.fromdense(a)
    i = [0, 2]
    np.testing.assert_array_equal(s[i], a[i])
    
    """Index a vector in a 2D array using slice operator."""
    a = np.arange(12).reshape(3,4)
    s = SparseArray.fromdense(a)
    np.testing.assert_array_equal(s[:,3], a[:,3])
    
    """Index a vector in a 3D array using slice operator."""
    a = np.arange(60).reshape(3,4,5)
    s = SparseArray.fromdense(a)
    np.testing.assert_array_equal(s[:,3,2], a[:,3,2])
    
    
    
    
#def test_getitem_advanced_slicing_lists():
    #"""Select items from a 2D array using a list."""
    #a = np.arange(12).reshape(3,4)
    #s = SparseArray.fromdense(a)
    #i = [[0,2],[1,0]]
    #np.testing.assert_array_equal(s[i], a[i]) # Error
    

def test_setitem_basic_slicing():
    """
    Test :meth:`SparseArray.__setitem__`.
    """
    
    """Assign single value in vector."""
    s = SparseArray((10))
    s[3] = 3.0
    assert(s[3]  == 3.0)
    
    """Assign single value in 2D array."""
    s = SparseArray((3,4))
    s[1,2] = 2.0
    assert(s[1,2] == 2.0)
    
    """Assign single value in 3D array."""
    s = SparseArray((2,3,4))
    s[1,2,2] = 5.0
    assert(s[1,2,2] == 5.0)

def test_setitem_basic_slicing_list():
    """
    Test :meth:`SparseArray.__setitem__`.
    """
    
    """Assign with sequence in vector."""
    s = SparseArray((10))
    a = s.todense()
    indices = [1,2,5,6]
    s[indices] = 5.0
    a[indices] = 5.0
    np.testing.assert_array_equal(s, a)

def test_setitem_basic_slicing_tuple_minus_1dim():
    """
    """
    a = np.zeros((3,3))
    s = SparseArray.fromdense(a)
    
    a[2] = 5.0
    s[2] = 5.0
    
    np.testing.assert_array_equal(s, a)
    


def test_setitem_basic_slicing_tuple_with_slice():
    """
    Test :meth:`SparseArray.__setitem__`.
    """
    a = np.zeros((3,4,5))
    s = SparseArray.fromdense(a)
    
    i = (slice(None), 1, 1)
    v = 1
    a[i] = v
    s[i] = v
    print a
    print s
    np.testing.assert_array_equal(s, a)
    
@raises(NotImplementedError)    
def test_setitem_basic_slicing_tuple_with_slice():
    """
    Test :meth:`SparseArray.__setitem__`.
    """
    a = np.zeros((3,4,5))
    s = SparseArray.fromdense(a)
    
    i = (slice(None), 1, 1)
    v = [1,1,1]
    a[i] = v
    s[i] = v
    print a
    print s
    np.testing.assert_array_equal(s, a)
    
def test_length():
    """
    Test :meth:`SparseArray.__len__`.
    """
    s = SparseArray((30,20,40))
    assert(len(s)==30)#(30*20*40))

def test_property_real():
    """
    Test :meth:`SparseArray.real`.
    """
    a = np.random.randn(2,3,4) + 1j * np.random.randn(2,3,4)
    s = SparseArray.fromdense(a)
    np.testing.assert_array_equal(s.real.todense(),a.real)

def test_property_imag():
    """
    Test :meth:`SparseArray.imag`.
    """
    a = np.random.randn(2,3,4) + 1j * np.random.randn(2,3,4)
    s = SparseArray.fromdense(a)
    np.testing.assert_array_equal(s.imag.todense(),a.imag)

def test_property_flat():
    """
    Test :meth:`SparseArray.flat`.
    """
    
    a = np.arange(10)
    s = SparseArray.fromdense(a)
    
    for i, j in zip(a.flat, s.flat):
        print i, j
        assert(i==j)
    
    a =  np.arange(100).reshape(10,10)
    s = SparseArray.fromdense(np.arange(100).reshape(10,10))
    
    for i, j in zip(a.flat, s.flat):
        print i, j
        assert(i==j)

def test_property_elements():
    """
    Test :meth:`SparseArray.elements`.
    """
    
    s = SparseArray((30,5,30))
    s[3,2,4] = 5.0
    
    assert(s.elements==1)
    
    s[10,2,5] = 3.0
    assert(s.elements==2)
    
def test_method_fill():
    """
    Test :meth:`SparseArray.fill`.
    """
    s = SparseArray((3,4))
    s[1,2] = 10.0
    s.fill(3.0)
    a = np.empty((3,4))
    a.fill(3.0)
    
    np.testing.assert_array_equal(s.todense(), a)
    

#def test_method_clean():
    #"""
    #Test :meth:`SparseArray.clean`.
    #"""
    
    
def test_method_cleaned():
    """
    Test :meth:`SparseArray.cleaned`.
    """
    
    s = SparseArray([3,3], default=0.0)
    s[1,1] = 0.0
    
    out = s.cleaned()
    assert(out.elements==0)
    
    s = SparseArray([3,3], default=5.0)
    s[1,1] = 5.0
    
    out = s.cleaned()
    assert(out.elements==0)
    

def test_method_todense():
    """
    Test :meth:`SparseArray.todense`.
    """  
    shape = (50,50)
    
    s = SparseArray(shape, default=0.0)
    s[10,10] = 1.0
    s[3, 5] = 5.0
    
    a = np.zeros(shape)
    a[10,10] = 1.0
    a[3, 5] = 5.0
    
    np.testing.assert_array_equal(s.todense(), a)

#def test_method_all:
    #"""
    #Test :meth:`SparseArray.all`.
    #"""
    #pass

def test_method_sum():
    """
    Test :meth:`SparseArray.sum`.
    """
    
    s = SparseArray((10,10), default=0.0)
    s[3,4] = 5.0
    s[2,3] = 10.0
    
    assert(s.sum()==15)
    
    s = SparseArray((5,10,10), default=5.0)
    s[2,3,4] = 5.0
    s[4,2,3] = 10.0
    
    assert(s.sum()==2505.0) 
    

def test_method_fromdense():
    """
    Test :meth:`SparseArray.fromdense`.
    """
    
    """Test in general converting dense to sparse."""
    x = np.random.randn(10,10)
    y = SparseArray.fromdense(x).todense()
    np.testing.assert_array_equal(x,y)
    
    """Test removing unnecessary entries."""
    x = np.zeros((10,10))
    x[5,5] = 1.0
    y = SparseArray.fromdense(x)
    
    assert(y.elements==1)
   

def test_operators():
    """
    Test all operators by comparing the results with numpy dense arrays.
    """
    
    operators = ['add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow']
    
    items = _get_items()
    
    for values, op in itertools.product(items, operators):
        a, b = values
        yield _operators_test, a, b, getattr(operator, op)

    
def _operators_test(a, b, op):
        
    try:
        ad = a.todense()
    except AttributeError:
        ad = a
    try:
        bd = b.todense()
    except AttributeError:
        bd = b
    
    c = op(a,b)
    try:
        c = c.todense()
    except AttributeError:
        pass
    
    np.testing.assert_array_equal(c, op(ad,bd))    

#def test_inplace_operators():
    #"""
    #Test all inplace operators by comparing the results with numpy dense arrays.
    #"""
    
    #operators = ['iadd', 'isub', 'imul', 'idiv', 'itruediv', 'ifloordiv', 'imod', 'ipow']
    
    #items = _get_items()
    
    #for values, op in itertools.product(items, operators):
        #a, b = values
        #yield _inplace_operators_test, a, b, getattr(operator, op)

def _inplace_operators_test(a, b, op):

    try:
        ad = a.todense()
    except AttributeError:
        ad = a.copy()
    try:
        bd = b.todense()
    except AttributeError:
        bd = b.copy() 
    
    op(a,b)
    op(ad, bd)
    np.testing.assert_array_equal(a, ad)

def _get_items():
    items = list()
    
    a = SparseArray((3,3)); a[2,1] = 10.0
    b = 5.0
    items.append((a,b))
    items.append((b,a))  # reverse operations fail
    
    a = SparseArray((3,3)); a[2,1] = 10.0
    b = SparseArray((3,3)); b[2,1] = 5.0
    items.append((a,b))
    items.append((b,a))
    
    a = SparseArray((4,4)); a[0,0] = 1.0
    b = SparseArray((4,4));
    items.append((a,b))
    items.append((b,a))
    
    a = SparseArray((10,10)); a[3,3] = 5.0 ; a[4,6] = 10.0
    b = SparseArray((10,10)); b[3,4] = 5.0 ; b[4,6] = 10.0
    items.append((a,b))
    items.append((b,a))
    
    a = SparseArray.fromdense(np.random.randn(10,10))
    b = np.random.randn(10,10)
    items.append((a,b))
    #items.append((b,a)) # Here we have a problem!
    
    a = SparseArray.fromdense(np.zeros((10,10)))
    b = np.zeros((10,10))
    items.append((a,b))
    #items.append((b,a)) # Here we have a problem!
    
    
    a = SparseArray.fromdense(np.random.randn(1,2,3,4,5).round())
    b = SparseArray.fromdense(np.random.randn(1,2,3,4,5).round())
    items.append((a,b))
    items.append((b,a))
    
    return items
    
    
