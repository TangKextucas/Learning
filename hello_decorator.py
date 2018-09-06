"""装饰器本质上是一个python函数，它可以让其他函数在不修改代码的前提下
    增加额外功能，常用于授权、日志、URL路由、缓存"""
"""函数装饰器"""
"""运行一个函数的同时想要添加一个日志信息，表示这个函数正在运行"""
import logging
def use_log(func):
    def wrapper():
        logging.warning("%s is running" % func.__name__)
        return func()
    return wrapper
@use_log
def foo():
    print('I am foo')
@use_log
def bar():
    print('I am bar')
bar()
foo()
"""类装饰器"""
class Foo(object):
    def __init__(self, func):
        self._func = func
    def __call__(self):
        print('class decorator running')
        self._func()
        print('class decorator ending')
@Foo
def yeah():
    print('yeah!')
yeah()
"""带参数的装饰器"""
def repeat(n):
    def qwe(asd):
        def wrapper(*args, **kwargs):
            for i in range(n):
                asd(*args, **kwargs)
        return wrapper
    return qwe
@repeat(3)
def p():
    print('important things')
p()
from functools import wraps
"""wraps本身是一个装饰器，使得装饰器函数也有和原函数一样的信息"""
def cover(dust):
    @wraps(dust)
    def with_other(*args, **kwargs):
        print(dust.__name__ + ' was called')
        return dust(*args, **kwargs)
    return with_other
@cover
def f(x):
    """do some math"""
    print(x**2)
    print(f.__name__)
    print(f.__doc__)
f(2)



