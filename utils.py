import numpy as np


def deprecated(func):
    return func


def unused(func):
    return func


def utility(func):
    return func


def moved(func):
    return func


@utility
class Container:
    '''очередь из 3 элементов с проверкой равенства
    хранящимся в контейнере элементам вставляемого элемента'''

    def __init__(self):
        self.container = []

    def insert(self, elem):
        if len(self.container) < 3:  # если контейнер заполнен не полностью то вставляем элемент в начало
            self.container.append(elem)
            return False
        elif len(self.container) == 3:  # если контейнер заполнен
            # проверяем равен ли вставляемый элемент какому-то из элементов в контейнере
            if np.isclose(elem, self.container[0]) \
                    or np.isclose(elem, self.container[1]) \
                    or np.isclose(elem, self.container[2]):
                self.container.clear()  # если равен -> очищаем контейнер и возвращаем True
                return True
            else:  # если не равен -> удаляем последний элемент и вставляем новый элемент в начало, возвращаем False
                self.container = [self.container[1], self.container[2], elem]
                return False
        else:
            self.container.clear()
            return False


class Parameters:
    def __init__(self, alpha_derivative, c_derivative, alpha2_derivative, alpha_c_derivative, c2_derivative,
                 parent_directory, filename='d_native.csv'):
        self.filename = filename
        self.parent_directory = parent_directory
        self.c2_derivative = c2_derivative
        self.alpha_c_derivative = alpha_c_derivative
        self.alpha2_derivative = alpha2_derivative
        self.c_derivative = c_derivative
        self.alpha_derivative = alpha_derivative
