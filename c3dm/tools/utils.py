from tools.attr_dict import AttrDict
import inspect
import io
import os
import tarfile
import time
import urllib.request
import zipfile

import numpy as np

def pprint_dict(d, indent=3):
	for key, value in d.items():
		print(' ' * indent + str(key),end='', flush=True)
		# print('.', end='', flush=True)
		if isinstance(value, AttrDict):
			# if len(value.keys())==1:
			# 	import pdb; pdb.set_trace()  # breakpoint 970a0708 //	
			# if not isinstance(value, AttrDict):
			print("")
			pprint_dict(value, indent+1)

		else:
			print(' = ' + str(value))

def has_method(ob,m):
	obcls=ob.__class__
	return hasattr(obcls, m) and callable(getattr(obcls,m))


def argparse_to_dict(args):
	raise NotImplementedError('finish this')
	return cfg


def get_net_input(batch):
	# move to gpu and cast to Var
	net_input = {}
	for k in batch:
		if has_method(batch[k],'cuda'):
			net_input[k] = batch[k].cuda()
		else:
			net_input[k] = batch[k]

	return net_input

def auto_init_args(obj,tgt=None,can_overwrite=False):
	# autoassign constructor arguments
	frame = inspect.currentframe().f_back # the frame above
	params  = frame.f_locals
	nparams = frame.f_code.co_argcount
	paramnames = frame.f_code.co_varnames[1:nparams]
	if tgt is not None:
		if not can_overwrite:
			assert not hasattr(obj,tgt)
		setattr(obj,tgt,AttrDict())
		tgt_attr = getattr(obj,tgt)
	else:
		tgt_attr = obj

	for name in paramnames:
		# print('autosetting %s -> %s' % (name,str(params[name])) )
		setattr(tgt_attr,name,params[name])


def untar_to_dir(url, path):
	response = urllib.request.urlopen(url)
	compressed_stream = io.BytesIO(response.read())
	if url.endswith('zip'):
		opener = lambda stream : zipfile.ZipFile(stream, "r")
	else:
		# assume tarball
		opener = lambda stream : tarfile.open(fileobj=stream, mode="r|*")

	with opener(compressed_stream) as tar:
		os.makedirs(path)
		tar.extractall(path=path)


class NumpySeedFix(object):

	def __init__(self,seed=0):
		self.rstate = None
		self.seed = seed
		
	def __enter__(self):
		self.rstate = np.random.get_state()
		np.random.seed(self.seed)

	def __exit__(self, type, value, traceback):
		if not(type is None ) and issubclass(type,Exception):
			print("error inside 'with' block")
			return
		np.random.set_state(self.rstate)

class TorchSeedFix(object):

	def __init__(self,seed=0):
		self.rstate = None
		self.seed = seed
		
	def __enter__(self):
		self.rstate = torch.random.get_rng_state()
		torch.manual_seed(self.seed)

	def __exit__(self, type, value, traceback):
		if not(type is None ) and issubclass(type,Exception):
			print("error inside 'with' block")
			return
		torch.manual_seed(self.seed)

class Timer:

	def __init__(self,name="timer",quiet=False):
		self.name = name
		self.quiet = quiet

	def __enter__(self):
		self.start = time.time()
		return self

	def __exit__(self, *args):
		self.end = time.time()
		self.interval = self.end - self.start        
		if not self.quiet:
			print( "%20s: %1.6f sec" % ( self.name , self.interval ) )