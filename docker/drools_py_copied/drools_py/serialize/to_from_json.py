import json



class ToFromJsonDict:

    @classmethod
    def fromJSON(cls, value: str):

        from drools_py.serialize.serializer_service import de_serialize, serialize
        out = json.loads(value, object_hook=lambda x: de_serialize(x))
        return out

    def to_self_dictionary(self) -> dict:
        return self.__dict__

    def to_dict(self):
        from drools_py.serialize.serializer_service import de_serialize, serialize
        return serialize(self)

    @classmethod
    def from_dict(cls, value: dict):
        v = json.dumps(value)
        return cls.fromJSON(v)

    def toJSON(self) -> str:
        out_dir = self.to_dict()
        return json.dumps(out_dir)
