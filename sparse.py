
import numpy as np
import operator
import itertools

#def _boundscheck(index, length):
    
    #if index < 0:
        #index += self.shape[0]
    #try:
        #check = index > self.shape[1]
    #except IndexError: # In case we have a vector
        #check = False
    #if index < 0 or check: #index >= self.shape[1]:
        #raise IndexError('index out of bounds')

def _apply_operator(a, b, op):

    def catch_zero_division(op): # Decorator to catch zero division.
        def wrapped(a, b):
            try:
                out = op(a,b)
            except ZeroDivisionError:
                if a==0.0 or op==operator.mod:
                    out = float('nan')
                else:
                    out = float('inf')
            return out
        return wrapped
        
    op = catch_zero_division(op) # Decorate the operator to handle ZeroDivisionError in a numpy compatible way.
        
    # Single value for 'b'
    if not hasattr(b, 'shape'):
        try:
            len(b)
        except TypeError:
            out = a.__class__(a.shape, default=op(a._default, b), dtype=a.dtype)
            for k in a._data.keys():
                out._data[k] = op(a[k], b)
            return out
        else:
            return _apply_operator(a, np.array(b), op) # Iterable. Let's convert to array and see again.
    
    # Single value for 'a' (reverse operation.)
    elif not hasattr(a, 'shape'):
        try:
            len(a)
        except TypeError:
            out = b.__class__(b.shape, default=op(a, b._default), dtype=b.dtype)
            for k in b._data.keys():
                out._data[k] = op(a, b[k])
            return out
        else:
            return _apply_operator(np.array(a), b, op) # Iterable. Let's convert to array and see again.
    

    elif a.shape != b.shape:
        raise ValueError("operands could not be broadcast together with shapes {} and {}.".format(a.shape, b.shape))
    
    # So both have a shape. Assume they're arrays then.

    elif isinstance(b, np.ndarray):
        #return a.todense() + b
        out = op(a._default*np.ones_like(b), b)
        for k in a._data.keys():
            out[k] = op(a._data[k], b[k])
        return out
    
    # For reverse operations. However, ndarray manages already.
    #elif isinstance(a, np.ndarray):
        ##return a.todense() + b
        #out = op(b._default*np.ones_like(a), a)
        #for k in b._data.keys():
            #out[k] = op(a[k], b._data[k])
        #return out
    
    # If both are sparse arrays.
    elif type(b) is type(a):
    
        out = a.__class__(a.shape, default=op(a._default, b._default), dtype=a.dtype)
        
        set_a = set(a._data.keys())
        set_b = set(b._data.keys())
        
        for k in set_a.intersection(set_b): # a & b
            out._data[k] = op(a._data[k], b._data[k])
        
        for k in set_a.difference(set_b): # a - b
            out._data[k] = op(a._data[k], b._default)
        
        for k in set_b.difference(set_a): # b - a
            out._data[k] = op(a._default, b._data[k])
        
        return out.clean()
    
    else:
        #print a, b, op
        raise ValueError("Yikes")

def _apply_inplace_operator(a, b, op):
    
    out = _apply_operator(a, b, op)
    try:
        a._data = out._data
        a._default = out._default
    except AttributeError:
        for k in range(out.size): # for every element in the dense array...
            index = np.unravel_index(k, out.shape) # index of the element
            a[index] = out[index]#op(a._data.get(index, a._default), out[index])

    
##def _apply_inplace_operator(a, b, op):

    ##def catch_zero_division(op): # Decorator to catch zero division.
        ##def wrapped(a, b):
            ##try:
                ##out = op(a,b)
            ##except ZeroDivisionError:
                ##if a==0.0 or op==operator.mod:
                    ##out = float('nan')
                ##else:
                    ##out = float('inf')
            ##return out
        ##return wrapped
        
    ##op = catch_zero_division(op) # Decorate the operator to handle ZeroDivisionError in a numpy compatible way.
        
    ##"""Single value."""
    ##if not hasattr(b, 'shape'):
        ##try:
            ##len(b)
        ##except TypeError:
            ###out = a.__class__(a.shape, default=op(a._default, b), dtype=a.dtype)
            ##for k in a._data.keys():
                ##a._data[k] = op(a[k], b)
            ###return out
        ##else:
            ##_apply_inplace_operator(a, np.array(b), op) # Iterable. Let's convert to array and see again.
        
    ##elif a.shape != b.shape:
        ##raise ValueError("operands could not be broadcast together with shapes {} and {}.".format(a.shape, b.shape))
    ### So both have a shape. Assume they're arrays then.
    
    ##elif isinstance(b, np.ndarray):
        ###return a.todense() + b
        ###out = op(a._default*np.ones_like(b), b)
        ##for k in range(b.size): # for every element in the dense array...
            ##index = np.unravel_index(k, out.shape) # index of the element
            ##a[index] = op(a._data.get(index, a._default), out[index])
            ###out[k] = op(a._data[k], b[k])
        ###return out
        
    ###elif isinstance(a, np.ndarray):
        ####return a.todense() + b
        ###out = op(b._default*np.ones_like(a), a)
        ###for k in b._data.keys():
            ###out[k] = op(a[k], b._data[k])
        ###return out
    
    ###"""If both are sparse arrays."""
    ##elif type(b) is type(a):
    
        ###out = a.__class__(a.shape, default=op(a._default, b._default), dtype=a.dtype)
        
        ##new_data = dict()
        
        ##set_a = set(a._data.keys())
        ##set_b = set(b._data.keys())
        
        ##for k in set_a.intersection(set_b): # a & b
            ##new_data[k] = op(a._data[k], b._data[k])
        
        ##for k in set_a.difference(set_b): # a - b
            ##new_data[k] = op(a._data[k], b._default)
        
        ##for k in set_b.difference(set_a): # b - a
            ##new_data[k] = op(a._default, b._data[k])
        
        ##a._data = new_data
        ###return out.cleaned()
    
    ##else:
        ##raise ValueError("Yikes")
    
def _convert_slice_to_list(s, length):
    """
    Convert a slice object to a valid list of indices.
    """
    return range(*s.indices(length))

def _correct_index(i, s):
    """
    Check whether the index is within bounds. Returns a corrected index.
    """
    if i >= s:
        raise IndexError("index out of bounds.")
    elif i < 0:
        return i + s
    else:
        return i

class SparseArray(object):
    """
    Sparse array.
    
    This class uses a dictionary for storing sparse data.
    
    """
    
    def __init__(self, shape, default=0.0, dtype='float64'):
        
        self.dtype = dtype
        """
        dtype.
        
        .. note:: Not really used (yet).
        
        """
        if issubclass(type(shape), int):
            self._shape = (shape,)
        else:
            self._shape = tuple(shape)
        #self._shape = shape if isinstance(shape, tuple) else (shape,)
        """
        Shape of array.
        """
        
        self._data = {}
        self._default = default
        #self._data.setdefault(default) # We do NOT set a default value here as well.
        
        
    def __getitem__(self, index):

        # Integer -> Basic Slicing
        if issubclass(type(index), int):
            return self._slicing_basic_integer(index)

        # Slice Object -> Basic Slicing
        elif isinstance(index, slice):
            return self._slicing_basic_slice(index)
        
        # Ellipsis -> Basic Slicing
        elif type(index) is type(Ellipsis):
            return self._slicing_basic_ellipsis(index)
        
        # Sequence
        elif hasattr(index, '__iter__'):
            
            # ndarray -> Advanced Slicing
            if isinstance(index, np.ndarray):
                return self._slicing_advanced(index)
            
            elif isinstance(index, tuple):
                for item in index:
                    # Tuple with sequence or ndarray -> Advanced Slicing
                    if hasattr(item,  '__iter__'):
                        return self._slicing_advanced(index)
                else:
                    # Tuple without sequence -> Basic Slicing
                    return self._slicing_basic_tuple(index)
            else:
                for item in index:
                    # Sequence with other sequences.
                    if hasattr(item,  '__iter__'):
                        return self._slicing_advanced(index)
                else:
                    # Other sequence with newaxis, slice or ellipsis -> Basic Slicing
                    return self._slicing_basic(index)
            
        else:
            raise ValueError("Oops!")
        
    def __setitem__(self, index, value):
        
        # Integer -> Basic Slicing
        if issubclass(type(index), int):
            self._setting_basic_integer(index, value)
        
        # Slice -> Basic Slicing
        elif isinstance(index, slice):
            self._setting_basic_slice(index, value)
        
        # Ellipsis -> Basic Slicing
        elif type(index) is type(Ellipsis):
            self._setting_basic_ellipsis(index, value)
        
        # Sequence
        elif hasattr(index, '__iter__'):
            
            # ndarray -> Advanced Slicing 
            if isinstance(index, np.ndarray):
                self._setting_advanced(index, value)
            
            elif isinstance(index, tuple):
                for item in index:
                    # Tuple with sequence or ndarray -> Advanced Slicing
                    if hasattr(item,  '__iter__'):
                        self._setting_advanced(index, value)
                        break #
                else:
                    # Tuple without sequence -> Basic Slicing
                    self._setting_basic(index, value)
            else:
                for item in index:
                    # Sequence with other sequences.
                    if hasattr(item, '__iter__'):
                        self._setting_advanced(index, value)
                        break
                else:
                    # Other sequence with newaxis, slice or ellipsis -> Basic Slicing
                    self._setting_basic(index, value)
        else:
            raise ValueError("Oops!")
        
        
        ##self._assert_valid_index(index)
        #if value != self._default: # Only store non-default values.
            
            ## Slice object
            #if isinstance(index, slice):
                #raise NotImplementedError
            
            ## Tuple
            #elif isinstance(index, tuple):
                
                #ndim = len(index)
                #if ndim == self.ndim: # Equal dimensions, so single value
                    ## Bounds checking!!
                    #self._data[index] = value
                #elif ndim > self.ndim:
                    #raise IndexError("too many indices")
                #else:
                    #raise NotImplementedError
                
            ## Sequence
            #elif hasattr(index, '__iter__'):
                #for i in index:
                    #self[i] = value
            
            ## Single value
            #else:
                #if index < 0: # Negative index
                    #index += self.shape[0]
                #try:
                    #check = index > self.shape[1]
                #except IndexError: # In case we have a vector
                    #check = False
                #if index < 0 or check: #index >= self.shape[1]:
                    #raise IndexError('index out of bounds')

                #if self.ndim > 1:
                    #iterables = [xrange(dim) for dim in self.shape[1:] ]
                    
                    #for x in itertools.product(iterables):
                        #self[x] = value
                #else:
                    #self._data[(index,)] = value
    
    
    def __delitem__(self, index):
        self._assert_valid_index(index)
        del self._data[index]
    
    def __add__(self, other):
        return _apply_operator(self, other, operator.add)
    
    def __sub__(self, other):
        return _apply_operator(self, other, operator.sub)
    
    def __mul__(self, other):
        return _apply_operator(self, other, operator.mul)
    
    def __div__(self, other):
        return _apply_operator(self, other, operator.div)
    
    def __truediv__(self, other):
        return _apply_operator(self, other, operator.truediv)
    
    def __floordiv__(self, other):
        return _apply_operator(self, other, operator.floordiv)
    
    def __mod__(self, other):
        return _apply_operator(self, other, operator.mod)
    
    def __pow__(self, other):
        return _apply_operator(self, other, operator.pow)
    
    def __radd__(self, other):
        return _apply_operator(other, self, operator.add)
    
    def __rsub__(self, other):
        return _apply_operator(other, self, operator.sub)
    
    def __rmul__(self, other):
        return _apply_operator(other, self, operator.mul)
    
    def __rdiv__(self, other):
        return _apply_operator(other, self, operator.div)
    
    def __rtruediv__(self, other):
        return _apply_operator(other, self, operator.truediv)
    
    def __rfloordiv__(self, other):
        return _apply_operator(other, self, operator.floordiv)
    
    def __rmod__(self, other):
        return _apply_operator(other, self, operator.mod)
    
    def __rpow__(self, other):
        return _apply_operator(other, self, operator.pow)
    
    def __repr__(self):
        return "{} sparse array of type {} with {} stored elements.".format(self.shape, self.dtype, self.elements)
    
    def __str__(self):
        return "{} sparse array of type {} with {} stored elements.".format(self.shape, self.dtype, self.elements)
    
    def __len__(self):
        return self.shape[0]
        #return reduce(lambda x,y: x*y, self.shape, 1) # Total elements that can be stored.
    
    def __iadd__(self, other):
        _apply_inplace_operator(self, other, operator.add)
        return self
    
    def __isub__(self, other):
        _apply_inplace_operator(self, other, operator.sub)
        return self
    
    def __imul__(self, other):
        _apply_inplace_operator(self, other, operator.mul)
        return self
    
    def __idiv__(self, other):
        _apply_inplace_operator(self, other, operator.div)
        return self
    
    def __itruediv__(self, other):
        _apply_inplace_operator(self, other, operator.truediv)
        return self
    
    def __ifloordiv__(self, other):
        _apply_inplace_operator(self, other, operator.floordiv)
        return self
    
    def __imod__(self, other):
        _apply_inplace_operator(self, other, operator.mod)
        return self
    
    def __ipow__(self, other):
        _apply_inplace_operator(self, other, operator.pow)
        return self
    
    def __iter__(self):
        ndim = self.ndim
        for i in range(self.shape[0]):
            yield self[i]
    
    def _slicing_basic_integer(self, index):
        """
        Basic slicing with integer.
        """
        index = _correct_index(index, self.shape[0])
        #if index >= self.shape[0]:
            #raise IndexError("index {} is out of bounds for axis {} with size {}".format(index, 0, self.shape[0]))

        #if index < 0: # Check for negative indexing.
            #index += self.shape[0]
        
        #try:
            #check = index > self.shape[1]
        #except IndexError: # In case we have a vector
            #check = False
        #if index < 0 or check: #index >= self.shape[1]:
            #raise IndexError('index out of bounds')

        if self.ndim > 1: # Return array one dimension smaller.
            out = self.__class__(self.shape[1:], default=self._default, dtype=self.dtype)
            for key in self._data.keys():
                if key[0] == index:
                    out[key[1:]] = self._data.get(key)
            return out
        
        else:
            return self._data.get((index,), self._default) # Convert the index from integer to tuple.  

    def _slicing_basic_slice(self, index):
        """
        Basic slicing with slice object.
        """
        return _convert_slice_to_list(index, self.shape[0])



    def _slicing_basic_ellipsis(self, index):
        """
        Basic slicing with slice object. Created a copy of the object.
        """
        out = self.__class__(shape=self.shape, default=self._default, dtype=self.dtype)
        out._data = self._data.copy()
        return out


    def _slicing_basic_list(self, index):
        """
        Basic slicing with a list.
        """
        items = len(index)
        out = self.__class__(shape=items, default=self._default, dtype=self.dtype)
        
        for i, ix in enumerate(index):
            out[i] = self[ix]
        return out


    def _slicing_basic_tuple(self, index):
        """
        Basic slicing with a tuple.
        """
        shape = self.shape
        
        # Basic Slicing - Tuple
        ndim = len(index)

        if ndim == self.ndim: # Equal dimensions, so single value
            
            # Bounds checking of integers and slices. Convert slices to lists.
            index = list(index)
            for i in range(ndim):
                if issubclass(type(index[i]), int):
                    index[i] = _correct_index(index[i], shape[i])
                elif type(index[i]) is slice:
                    index[i] = _convert_slice_to_list(index[i], shape[i])
                elif type(index[i]) is type(Ellipsis):
                    raise NotImplementedError("Ellipsis is not supported.")
                else:
                    #print type(index[i])
                    raise ValueError("Oops...")
            index = tuple(index)

            try:    # If the tuple consists of just integers we can return the single value right away

                return self._data.get(index, self._default)
            except TypeError: # And if it doesn't it (likely) contains lists.
                # Use itertools product.
                
                index = list(index)
                out_shape = list()
                ranges = list()
                for i, dim in enumerate(index):
                    try:
                        l = len(dim)
                    except TypeError:
                        l = 1
                        index[i] = [dim]
                    else:
                        out_shape.append(l)
                        ranges.append(range(l))
                
                out = self.__class__(tuple(out_shape), default=self._default, dtype=self.dtype)
                
                indices = [i for i in itertools.product(*index)] # Create a flat list of indices
                new_indices = [i for i in itertools.product(*ranges)] # Create a flat list of indices for the new array
                
                for i, new_i in zip(indices, new_indices):
                    out[new_i] = self[i]  
                return out
                
        elif ndim > self.ndim:
            raise IndexError("too many indices")
        elif ndim < self.ndim:
            raise NotImplementedError
        else:
            raise IndexError("Oops")
        
    def _slicing_basic(self, index):
        """
        Basic slicing.
        """
        
        if isinstance(index, tuple):
            return self._slicing_basic_tuple(index)
        elif isinstance(index, list):
            return self._slicing_basic_list(index)
        else:
            raise NotImplementedError("unsupported type.")

    def _slicing_advanced(self, index):
        raise NotImplementedError("Advanced slicing is not supported.")

    def _slicing_advanced_ndarray(self, index):
        # If index.dtype is integer or bool continue, else 
        #IndexError: arrays used as indices must be of integer (or boolean) type
        raise NotImplementedError("Advanced slicing is not supported.")

    def _get_index_integer(self, index):
        """
        Get the value for the given index in case the index is an integer.
        """
        index = (_correct_index(index, self.shape[0]),)
        return self._data[index]
        
    def _get_index_tuple(self, index, value):
        """
        Get the value for the given index in case the index is a tuple consisting of integers.
        """
        ndim = self.ndim
        shape = self.shape
        index = list(index)
        for i in range(ndim):
            if issubclass(type(index[i]), int):
                index[i] = _correct_index(index[i], shape[i])
        index = tuple(index)
        return self._data[index]
        
    def _set_index_integer(self, index, value):
        """
        Set the value at index in case the index is an integer.
        """
        index = (_correct_index(index, self.shape[0]),)
        
        if value == self._default:
            del self._data[index]
        else:
            self._data[index] = value

    def _set_index_tuple(self, index, value):
        """
        Set the value at index in case the index a tuple.
        """
        ndim = self.ndim
        shape = self.shape
        index = list(index)
        for i in range(ndim):
            if issubclass(type(index[i]), int):
                index[i] = _correct_index(index[i], shape[i])
        index = tuple(index)
        if value == self._default:
            del self._data[index]
        else:
            self._data[index] = value
        

    def _setting_basic_integer(self, index, value):
        """
        Setting integer.
        """

        # The array is a vector. An integer can then only set a single value.
        if self.ndim == 1:
            self._set_index_integer(index, value)
            #self._data[_correct_index(index, self.shape[0])] = value
        
        else:
            iterables = [range(dim) for dim in self.shape[1:] ]
            iterables.insert(0, [index])
            #print iterables
            for x in itertools.product(*iterables):
                #print x
                self._set_index_tuple(x, value)
                #self._data[_correct_index(x, shape[0])] = value
        
        #raise NotImplementedError

    def _setting_basic_list(self, index, value):
        """
        Set array with a list.
        """
        for ix in index:
            self[_correct_index(ix, self.shape[0])] = value

    def _setting_basic_tuple(self, index, value):
        """
        Set array with a tuple.
        """
        
        ndim = self.ndim
        shape = self.shape
        
        try:
            len(value)
        except TypeError:
            pass
        else:
            raise NotImplementedError("Setting array with a sequence is not implemented.")
        
        if len(index) == ndim:
                
            # Bounds checking of integers and slices. Convert slices to lists.
            index = list(index)
            for i in range(ndim):
                if issubclass(type(index[i]), int):
                    index[i] = _correct_index(index[i], shape[i])
                elif type(index[i]) is slice:
                    index[i] = _convert_slice_to_list(index[i], shape[i])
                elif type(index[i]) is type(Ellipsis):
                    raise NotImplementedError("Ellipsis is not supported.")
                else:
                    #print type(index[i])
                    raise ValueError("Oops...")
            index = tuple(index)

            try:
                self._data[index] = value
            except TypeError: # This means it contains lists.

                index = list(index)
                out_shape = list()
                ranges = list()
                for i, dim in enumerate(index):
                    try:
                        l = len(dim)
                    except TypeError:
                        l = 1
                        index[i] = [dim]
                    else:
                        out_shape.append(l)
                        ranges.append(range(l))
                
                #out = self.__class__(tuple(out_shape), default=self._default, dtype=self.dtype)
                
                indices = [i for i in itertools.product(*index)] # Create a flat list of indices
                #new_indices = [i for i in itertools.product(*ranges)] # Create a flat list of indices for the new array
                
                for i in indices:
                    #print i
                    self._set_index_tuple(i, value)
                
                #for i, new_i in zip(indices, new_indices):
                    #out[new_i] = self[i]  
                #return out

    def _setting_basic(self, index, value):
        
        if isinstance(index, tuple):
            self._setting_basic_tuple(index, value)
        elif isinstance(index, list):
            self._setting_basic_list(index, value)
        else:
            raise NotImplementedError("unsupported type.")

    def _setting_advanced(self, index, value):
        raise NotImplementedError

    def _assert_valid_index(self, index):
        """
        Check whether the index is valid.
        """
        try:
            ndim = len(index)
        except TypeError:
            ndim = 1
        
        try:
            assert(index<=self.shape[0])
        except AssertionError:
            raise IndexError("Invalid index.")

        
        #if ndim != self.ndim:
            #raise IndexError("Wrong amount of dimensions.")
        ##if not np.all(np.array(index) < np.array(self.shape)):
            #raise IndexError("Out of range.")

    @property
    def shape(self):
        """
        Shape.
        """
        return self._shape
    
    @property
    def flat(self):
        """
        Flattened.
        """
        for i in range(len(self)):
            index = np.unravel_index(i, self.shape)
            yield self[index]
    
    @property
    def size(self):
        """
        Amount of elements in array.
        """
        return reduce(lambda x,y: x*y, self.shape, 1) # Total elements that can be stored.
    
    @property
    def ndim(self):
        """
        Amount of dimensions.
        """
        return len(self.shape)
    
    @property
    def elements(self):
        """
        Amount of stored stored elements.
        """
        return len(self._data)
    
    @property
    def real(self):
        """
        The real part of the array.
        """
        out = self.__class__(self.shape, default=np.real(self._default), dtype=self.dtype)
        for index, value in self._data.iteritems():
            out[index] = np.real(value)
        return out
        
    @property
    def imag(self):
        """
        The imaginary part of the array.
        """
        out = self.__class__(self.shape, default=np.imag(self._default), dtype=self.dtype)
        for index, value in self._data.iteritems():
            out[index] = np.imag(value)
        return out
    
    def clean(self):
        """
        Remove unnecessary elements (elements that are equal to the default value) from the array.
        
        .. note:: This method should not be necessary.
        
        """
        data = self._data.copy()
        for index, value in self._data.iteritems():
            if value == self._default:
                del data[index]
        self._data = data
        return self
    
    def cleaned(self):
        """
        Return a copy of the array with unnecessary elements (element equal to default value) removed.
        
        .. note:: This method should not be necessary.
        
        """
        out = self.__class__(self.shape, default=self._default, dtype=self.dtype)
        for index, value in self._data.iteritems():
            if value != self._default:
                out[index] = value
        return out
    
    def all(self):#, axis=None, out=None, keepdims=False):
        """
        Test whether all array elements along a given axis evaluate to True.
        """
        #if axis is not None:
            #raise NotImplementedError("axis is not yet implemented.")
        #if out is not None:
            
        #if keepdims:
            #raise NotImplementedError("keepdims is not yet implemented.")
        
        return all(self._data.values()) and self._default
            
            
    def any(self):
        """
        Test whether any array element along a given axis evaluates to True.
        """
        return any(self._data.values()) or self._default
            
    
    def sum(self, axis=None, dtype=None, out=None, keepdims=False, stored_only=False):
        """
        Sum of all elements.
        
        .. note:: Does not yet support summations along dimensions.
        
        """
        if axis is not None:
            raise NotImplementedError("axis is not yet implemented.")
        if dtype is not None:
            raise NotImplementedError("dtype is not yet implemented.")
        if keepdims:
            raise NotImplementedError("keepdims is not yet implemented.")
        
        arr = np.array(self._data.values()).sum()
        
        if not stored_only:
            arr += ( reduce(lambda x, y: x*y, self.shape, 1) - self.elements ) * self._default
        if out:
            if out.shape==arr.shape:
                out._data = arr._data.copy()
            else:
                raise ValueError("operands could not be broadcast together with shapes {} {}".format(out.shape, arr.shape))
        else:
            return arr
    
    def fill(self, value):
        """
        Fill the array with a scalar value.
        """
        self._data.clear()
        self._default = value
    
    def todense(self):
        """
        Convert to dense array.
        """
        out = self._default * np.ones(self.shape, self.dtype)
        for index, value in self._data.iteritems():
            #if index != 0.0: # This was to catch the dict default value, but we're not using a dict default value anymore.
            out[index] = value
        return out
    
    @classmethod
    def fromdense(cls, arr, default=0.0):
        """
        Create sparse array from dense array.
        """
        out = SparseArray(arr.shape, default=default, dtype=arr.dtype)
        for index, value in np.ndenumerate(arr):
            if value != default:
                out[index] = value
        return out

        
        