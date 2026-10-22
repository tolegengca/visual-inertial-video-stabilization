import unittest
import numpy as np
from vstab.motion import *

class MotionTests(unittest.TestCase):
    def test_consistency(self):
        f=np.zeros((20,30,2),np.float32); f[...,0]=2
        b=np.zeros_like(f); b[...,0]=-2
        m,e=consistency(f,b)
        self.assertGreater(m.mean(),.85)
    def test_similarity_with_outliers(self):
        rng=np.random.default_rng(4); p=rng.uniform(0,100,(300,2)).astype(np.float32)
        M=np.array([[.999,-.03,5],[.03,.999,-3]],np.float32); q=p@M[:,:2].T+M[:,2]
        q[:60]=rng.uniform(-100,200,(60,2))
        m,r,_=estimate_similarity(p,q,1.0)
        self.assertGreater(r,.7); self.assertLess(abs(m[0,2]-5),.2)
    def test_accumulate_smooth(self):
        ms=[np.array([[1,0,1],[0,1,(-1)**i]],float) for i in range(20)]
        poses=accumulate(ms); sm,raw,smooth=smooth_poses(poses,2,6)
        self.assertLess(jitter_rms(smooth),jitter_rms(raw))
    def test_grid_uniform(self):
        f=np.zeros((40,60,2),np.float32); mask=np.ones((40,60),bool); e=np.zeros((40,60))
        p,q=grid_sample(f,mask,e,4,6,2); self.assertEqual(len(p),48)

    def test_matrix_parameter_roundtrip(self):
        p=np.array([4.0,-3.0,.12,np.log(1.03)])
        self.assertTrue(np.allclose(params_from_matrix(matrix_from_params(p)),p,atol=1e-8))

    def test_identity_crop_is_full_frame(self):
        crop=safe_center_crop(np.repeat(np.eye(3)[None],5,axis=0),640,360)
        self.assertGreater(crop,.999)

    def test_zero_strength_compensation_is_identity(self):
        ms=[np.array([[1,0,2],[0,1,-1]],float) for _ in range(4)]
        poses=accumulate(ms); target,_,_=smooth_poses(poses,1,2)
        warps=compensation(poses,target,(0,0,0,0))
        self.assertTrue(np.allclose(warps,np.repeat(np.eye(3)[None],len(poses),axis=0)))

    def test_visual_row_motion_recovers_slope(self):
        rng=np.random.default_rng(9)
        src=rng.uniform([0,0],[640,360],(1200,2))
        global_m=np.array([[1,0,4],[0,1,-2]],float)
        row=src[:,1]/360-.5
        dst=src+np.array([4,-2])+np.column_stack((3*row,-5*row))
        model=visual_row_motion(src,dst,global_m,360)
        self.assertIsNotNone(model)
        self.assertAlmostEqual(model["tx_slope"],3,places=1)
        self.assertAlmostEqual(model["ty_slope"],-5,places=1)

    def test_short_path_jitter_is_zero(self):
        self.assertEqual(jitter_rms(np.zeros((2,4))),0.0)

if __name__=="__main__": unittest.main()
