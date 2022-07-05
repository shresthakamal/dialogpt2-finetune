from loguru import logger


def logging():
    logger.add(
        "logs/{time}.log",
        format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> <level>{message}</level>",
        rotation="1 week",
    )
    return logger
