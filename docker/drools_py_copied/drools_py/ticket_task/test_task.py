
class TestTask:

    @classmethod
    def test_task(cls, task_id: list[str], *args, **kwargs):

        def decorator(func):

            def wrapper(*args, **kwargs):
                # TODO: perform pre-task saving, etc.
                return func(*args, **kwargs)

            return wrapper

        return decorator

