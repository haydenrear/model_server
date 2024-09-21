import dataclasses


@dataclasses.dataclass(init=True)
class ConsumerConfigProps:
    max_batch: int
    timeout: int


@dataclasses.dataclass(init=True)
class ProducerConfigProps:
    pass
