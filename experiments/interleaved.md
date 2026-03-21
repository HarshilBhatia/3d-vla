currently rope is implemented and applied per-head. ( D / (3*H))
but i want the features to be computed across the whole dimension. 
so basically, since the dim is D, i want first 2 features to have x_1, then next 2 features to have y_1 and next 2 to have z_1, and so on with x2,y2,z2 ...

and then this should be used inside each head. 

We need to do this efficiently, all rope features should still be pre-computed