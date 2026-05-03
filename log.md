


### 29th April
open_drawer variation success rates: {0: 6, 1: 6, 2: 12, 'mean': 0.27}
open_drawer mean success rate: 0.27



open_drawer variation success rates: {0: 8, 1: 6, 2: 13, 'mean': 0.3}
open_drawer mean success rate: 0.3

success rate for peract on online eval. 

No idea how this has gone bad again :/

possible errors: 
1. data
2. data lol.



Depth map stats. 

(Pdb) depth_l.max()
4.8897977
(Pdb) depth_r.max()
10.000005
(Pdb) depth_w.max()\
*** SyntaxError: unexpected EOF while parsing
(Pdb) depth_w.max() 
0.91584134
(Pdb) 

all data seems correct.

lets think clearly again before rerunning anything! 


currently running 

G1 trained (3dfa + miscal) + several camera trained (3dfa). 
