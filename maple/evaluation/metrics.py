"""Five reported metrics. Both F1 values and precision use support weighting."""
import numpy as np
from ..data.sampling import patient_groups


def weighted_metrics(y, pred, num_classes=4):
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(matrix, (y, pred), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    tp = np.diag(matrix)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(num_classes), where=precision+recall > 0)
    return float((precision*support).sum()/support.sum()), float(tp.sum()/support.sum()), float((f1*support).sum()/support.sum())


def metrics(labels, predictions, speakers, num_classes=4):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    patients_y, patients_p = [], []
    for rows in patient_groups(speakers).values():
        # Test-time majority vote; a tie chooses the smaller class index.
        patients_y.append(np.bincount(labels[rows], minlength=num_classes).argmax())
        patients_p.append(np.bincount(predictions[rows], minlength=num_classes).argmax())
    precision, accuracy, seg_f1 = weighted_metrics(labels, predictions, num_classes)
    person_f1 = weighted_metrics(patients_y, patients_p, num_classes)[2]
    return dict(seg_precision=precision,
                seg_accuracy=accuracy, seg_f1=float(seg_f1),
                person_f1=float(person_f1), final_score=float(10*person_f1+seg_f1))
