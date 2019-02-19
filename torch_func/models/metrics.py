
def accuracy(output, target):
    predictions = output.max(1)[1].type_as(target)
    correct = predictions.eq(target).sum().item()
    return 1.0 * correct / target.shape[0]
