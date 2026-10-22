#!/usr/bin/env python3
import argparse, json
from vstab.pipeline import stabilize

p=argparse.ArgumentParser(description="Bidirectional dense-flow video stabilization")
p.add_argument("input"); p.add_argument("output")
p.add_argument("--backend",choices=["raft","farneback"],default="raft")
p.add_argument("--device",default="auto"); p.add_argument("--working-width",type=int,default=480)
p.add_argument("--sigma",type=float,default=3.3); p.add_argument("--radius",type=int,default=10)
p.add_argument("--translation-strength",type=float,default=1.0)
p.add_argument("--rotation-strength",type=float,default=0.0)
p.add_argument("--scale-strength",type=float,default=0.0)
a=p.parse_args()
m=stabilize(a.input,a.output,backend=a.backend,device=a.device,working_width=a.working_width,
            sigma=a.sigma,radius=a.radius,strengths=(a.translation_strength,a.translation_strength,
            a.rotation_strength,a.scale_strength))
print(json.dumps(m,indent=2))

