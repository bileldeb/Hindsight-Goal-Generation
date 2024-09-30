from .normal import NormalLearner
from .hgg_mo import HGGLearner
from .hssgg import HSSGGLearner


learner_collection = {
	'normal': NormalLearner,
	'hgg': HGGLearner,
	'hssgg' : HSSGGLearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)