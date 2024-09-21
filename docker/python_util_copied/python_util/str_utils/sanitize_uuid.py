import uuid


def sanitize_uuid(uuid: uuid.UUID) -> str:
    return str(uuid).replace('-', '')


def sanitize_uuid_str(uuid: str) -> str:
    return uuid.replace('-', '')
