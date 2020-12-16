from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import logging

logger = logging.getLogger(__name__)


def report_performance(y_true, y_pred):

    m = confusion_matrix(y_true, y_pred)
    log_msg = f"confusion_matrix test: \n{str(m)}"
    print(log_msg)
    logger.info(log_msg)

    acc = accuracy_score(y_true, y_pred)
    log_msg = f"acc test: {acc}"
    print(log_msg)
    logger.info(log_msg)

    f1 = f1_score(y_true, y_pred, average="weighted")
    log_msg = f"f1 test: {f1}"
    print(log_msg)
    logger.info(log_msg)

    return
