# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed
    # to stay the same for AutoLab.

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # Might we need to store something before returning?
        self.state = 1./(1+np.exp(-x))
        return self.state

    def derivative(self):
        # Maybe something we need later in here...
        return self.state * (1-self.state)

"""
NOTE:
在Python中,NotImplemented是一个特殊的对象,它表示当前操作或函数的实现尚未定义或未实现。当使用NotImplemented时,它通常作为一个占位符，暗示着需要后续的实现。

通常,在类中重载某些运算符或方法时,可以使用NotImplemented来表示该运算符或方法的具体实现还没有被定义。这可以帮助程序员编写代码框架,使其能够更容易地扩展和重用。例如,如果您定义了一个抽象基类,而该基类中有一个方法需要在子类中实现,则可以在该方法中使用NotImplemented,这样子类必须重载该方法并提供具体的实现。

需要注意的是,NotImplemented与NotImplementedError是不同的。NotImplementedError是一个异常,通常用于表示某个方法或函数的实现尚未完成或未被定义。而NotImplemented则是一个特殊的占位符对象,表示当前操作或函数的实现未定义。
"""

class Tanh(Activation):

    """
    Tanh non-linearity
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        e_z = np.exp(x)
        e__z = np.exp(-x)
        self.state = (e_z - e__z) / (e_z + e__z)
        return self.state

    def derivative(self):
        return 1 - np.power(self.state)


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = x
        if self.state > 0:
            return self.state
        else:
            return 0

    def derivative(self):
        if self.state > 0:
            return 1.0
        else:
            return 0
