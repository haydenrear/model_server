import typing



class PythonDelegate(type):

    def __new__(cls, name, bases, attrs):
        for k, v in attrs['__annotations__']['delegate'].__dict__.items():
            if k not in attrs.keys():
                attrs[k] = v
        return super(PythonDelegate, cls).__new__(cls, name, bases, attrs)

