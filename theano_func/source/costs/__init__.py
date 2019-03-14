from . import cross_entropy_loss
from . import quadratic_loss
from . import weight_decay
from . import adversarial_training
from . import virtual_adversarial_training as vat
from . import virtual_adversarial_training_finite_diff as vat_finite_diff

from . import error

cross_entropy_loss = cross_entropy_loss.cross_entropy_loss
quadratic_loss = quadratic_loss.quadratic_loss
weight_decay = weight_decay.weight_decay
adversarial_training = adversarial_training.adversarial_training

LDS = vat.LDS
average_LDS = vat.average_LDS
virtual_adversarial_training = vat.virtual_adversarial_training

LDS_finite_diff = vat_finite_diff.LDS_finite_diff
average_LDS_finite_diff = vat_finite_diff.average_LDS_finite_diff
virtual_adversarial_training_finite_diff = vat_finite_diff.virtual_adversarial_training_finite_diff
error = error.error