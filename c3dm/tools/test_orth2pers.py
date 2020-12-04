import torch
import torch.nn.functional as Fu

def find_camera_T(K, X, Y):

    n = X.shape[2]
    ba = X.shape[0]

    append1 = lambda x: \
        torch.cat((x,x.new_ones(x.shape[0],1,x.shape[2])), dim=1)

    # projection rays
    r = torch.bmm(torch.inverse(K), append1(Y))
    r = Fu.normalize(r, dim=1)

    # outer projection ray product (need to permute the array first)
    rr = r.permute(0,2,1).contiguous().view(n*ba, 3)
    rr = torch.bmm(rr[:,:,None], rr[:,None,:])

    # I - rr
    Irr = torch.eye(3)[None].repeat(ba*n,1,1) - rr

    # [rr - I] x
    rrIx = torch.bmm(-Irr, X.permute(0,2,1).contiguous().view(n*ba, 3, 1))

    Irr_sum  = Irr.view(ba,-1,3,3,).sum(1)
    rrIx_sum = rrIx.view(ba,-1,3).sum(1)

    rrI_sum_i = torch.inverse(Irr_sum)

    T = torch.bmm(rrI_sum_i, rrIx_sum[:,:,None])[:,:,0]

    return T

n = 500 # n points
ba = 20 # batch size

# gt 3D points
X = torch.zeros(ba, 3, n).normal_(0., 1.)

for focal in torch.linspace(10.,0.1,20):

    # cam K
    K = torch.eye(3)
    K[0,0] = focal
    K[1,1] = focal
    K = K[None].repeat(ba,1,1)

    if False:
        # persp projections - should give 0 error everywhere
        T = torch.ones(ba, 3).uniform_()*10.
        Y = torch.bmm(K, X + T[:,:,None])
        Y = Y[:,0:2,:] / Y[:,2:3,:]
    else: 
        # orth projections - should get higher error with lower focal
        Y = X[:,0:2]

    T = find_camera_T(K, X, Y)

    ## test the repro loss
    # perspective projections
    Yp = torch.bmm(K, X + T[:,:,None])
    depth_ = Yp[:,2:3, :]
    Yp = Yp[:,0:2, :] / depth_

    # the diff between orth and persp
    df = ((Y - Yp)**2).sum(1).sqrt().mean(1).mean()

    print('focal = %1.2f, repro_df = %1.2e, mean_depth = %1.2f' % \
        (focal, df, depth_.mean()) )