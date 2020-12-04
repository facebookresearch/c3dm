from tools.utils import auto_init_args
import torch
import torch.nn.functional as Fu
from torch.nn import Parameter
from tools.utils import Timer

class TensorAccumulator(torch.nn.Module):
	def __init__(self, db_size=30000, db_dim=3, perc_replace=0.01):
		super().__init__()
		auto_init_args(self)
		db = torch.zeros(db_dim, db_size).float()
		self.db = Parameter(db)
		self.db.requires_grad = False
		self.pointer = 0
		self.uniform_sphere_sampling = False
		
	def get_db(self, uniform_sphere=False):
		if uniform_sphere or self.uniform_sphere_sampling:
			mean_norm = (self.db.data**2).sum(0).sqrt().mean()
			db = Fu.normalize(torch.randn_like(self.db), dim=0) * mean_norm
			return db
		else:
			if hasattr(self, 'db_f'):
				return self.db_f.clone()
			else:
				return self.db.data

	def filter_db(self, nn=1e-3, perc_keep=0.9, sig=0.01, lap_size=10, lap_alpha=1.):
		print('filtering db')
		if nn < 1.: nn = int(self.db.shape[1] * nn)
		print('nn size = %d' % nn)
		db_f = self.density_filter(nn=nn, perc_keep=perc_keep, \
								  sig=sig, in_db=self.db.data.clone())
		if lap_size < 1.: lap_size = int(self.db.shape[1] * lap_size)
		db_f = self.lap_filter(lap_size=lap_size, lap_alpha=lap_alpha, in_db=db_f)
		self.db_f = db_f
		
	def get_edm(self, pts, pts2=None):
		dtype = pts.data.type()
		ba, dim, N = pts.shape
		if not(pts2 is None):
			edm     = torch.bmm(-2. * pts2.transpose(1,2), pts)
			fNorm1  = (pts*pts).sum(1,keepdim=True)
			fNorm2  = (pts2*pts2).sum(1,keepdim=True)
			edm    += fNorm2.transpose(1,2)
			edm    += fNorm1
		else:
			edm     = torch.bmm(-2 * pts.transpose(1,2), pts)
			fNorm1  = (pts*pts).sum(1,keepdim=True)
			edm    += fNorm1.transpose(1,2)
			edm	   += fNorm1
		return edm.contiguous()

	def reset(self):
		self.db.data = torch.zeros(self.db_dim, self.db_size).type_as(self.db.data)
		self.pointer = 0

	def get_nns(self, pts, pts2, nn, bsize=int(1e4)):
		# nb = int(np.ceil(pts.shape[1] / bsize))
		chunks = torch.split(pts, bsize, dim=1)
		indKNN = []
		for chunk in chunks:
			edm = self.get_edm(pts2[None], chunk[None])[0]
			_, indKNN_ = torch.topk(edm, k=nn, dim=1, largest=False)
			indKNN.append(indKNN_)
		indKNN = torch.cat(indKNN, dim=0)
		return indKNN

	def density_filter(self, nn=50, perc_keep=0.9, sig=0.01, in_db=None):
		print('density filter ...')
		if in_db is None:
			pcl = self.db.data
		else:
			pcl = in_db

		indKNN = self.get_nns(pcl, pcl, nn=nn)
		# edm = self.get_edm(pcl[None])[0]
		# _, indKNN = torch.topk(edm, k=nn, dim=0, largest=False)
		NNs = pcl[:,indKNN]
		dsity = (-((NNs - pcl[:,:,None])**2).sum(0)/sig).exp().sum(1)
		thr = torch.topk(dsity, \
			int((1.-perc_keep)*dsity.shape[0]), largest=False)[0][-1]
		pcl = pcl[:, dsity>=thr]
		if in_db is None:
			self.db.data = pcl
		else:
			return pcl
			
	def lap_filter(self, lap_size=10, lap_alpha=1., in_db=None):
		print('lap filter ...')
		if in_db is None:
			pcl = self.db.data
		else:
			pcl = in_db

		indKNN = self.get_nns(pcl, pcl, nn=lap_size)
		NNs = pcl[:,indKNN]
		pclf = NNs.mean(dim=2)
		pcl = lap_alpha * pclf + (1-lap_alpha) * pcl
		if in_db is None:
			self.db.data = pcl
		else:
			return pcl

	def forward(self, embed=None, masks=None):
		
		if not self.training: # gather only on the train set
			return None

		ba = embed.shape[0]	

		embed_flat = embed.view(ba,self.db_dim,-1).detach()
		if masks is not None:
			mask_flat = masks.view(ba, -1)
		else:
			mask_flat = embed_flat[:,0,:] * 0. + 1.

		# with Timer():
			# embed_flat = embed_flat.permute(1,2,0).contiguous().view(1,self.db_dim,-1)
			# mask_flat = mask_flat.t().contiguous().view(1,-1)
		for bi, (m, e) in enumerate(zip(mask_flat, embed_flat)):
			sel = torch.nonzero(m).squeeze()
			if sel.numel()<=2:
				continue
			nsel = max(int(self.db_size * self.perc_replace), 1)
			if self.pointer >= self.db_size: # randomly replace
				idx = sel[torch.LongTensor(nsel).random_(0, len(sel))]
				idx_replace = torch.LongTensor(nsel).random_(0, self.db_size)
				embed_sel = e[:,idx].detach().data
				self.db.data[:, idx_replace] = embed_sel
			else: # keep adding vectors
				# print('filling db ...')
				nsel = min(nsel, self.db_size - self.pointer)
				idx = sel[torch.LongTensor(nsel).random_(0, len(sel))]
				embed_sel = e[:,idx].detach().data
				self.db.data[:, self.pointer:(self.pointer+nsel)] = embed_sel
				self.pointer += nsel			

			# print(self.pointer)

		return None

