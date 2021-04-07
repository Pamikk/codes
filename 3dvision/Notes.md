<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"> </script>

# Notes for 3D Vision
+ this notes will cover some useful knowledge and techs for 3D vision(including multi-view geometry, linear algebra, camera models, etc.)
+ Structure
  +
# Introduction to 6d object pose estimation 
+ This part is currently based on my undergraduate thesis finished in early 2019, something might be out of date, but it will update to the state-of-the-art later.
+ Concepts
  + Camera Model[1]\
    To define object pose, we need to understand how camera transfer obj from real word to images.
    + Pinhole Perspective(cental perspective)\
      ![img](imgs/pinhole-1.png)
      + inverted images
      + apparent size of objs depends on their distance
      + projections of two parallel lines lying in some plane appear to converge on a horizon line h
        + some have no images
      + Coordinates:\
        ![img](imgs/pinhole-2.png)
        + origin $O$ pinhole, basis $i$,$j$ parallel image plane
        + $c$ image center $Oc$ optical axis
        + mapping $P(X,Y,Z)$ to $p(x,y,z)$
          + $P,O,p$ colinear leads to $Op= \lambda OP$ , so that
          $$ \lambda = \frac{x}{X} = \frac{y}{Y}=\frac{d}{Z}$$
          + therefore 
          $$x=d\frac{X}{Z},y=d\frac{Y}{Z}$$  
      + weak perspective(scaled orthography)
        + fronto-parallel palnes
          + vectors are parallel to their iamges
    + Camera with Lenses
      + ideal thin len\
        ![img](imgs/cam-wt-len-1.png)
        $$\frac{1}{z}-\frac{1}{Z} = \frac{1}{f}$$
        where $f=\frac{R}{2(n-1)}$, surface redius R$R$, index of refraction $n$
        + notice field of view of a camera also depends on effective area of the retina
      + simple thick lens\ 
        ![img](imgs/cam-wt-len-2.png)
        + aberrations
          + spherical aberration
            + longitudinal spherical aberrration
            + transverse shpereical aberration
          + coma
          + astigmatism
          + field curvature
          + distortion
            + different areas of a lens have sightly different focal length
    + Instrinsic and extrinsic parameters
      + Rigid Transformation and Homogeneous Coordinates
        + transform in non-homogeneous coordinates
          $$^AP=R^BP+t$$
        + in homogeneous coordinates
          $$ ^A P =T^B P,where\ T^B= \left\{
 \begin{matrix}
   R & t  \\
   0^T & 1 
  \end{matrix}
  \right\}$$
+ Methods
  + RGB-D based
  + RGB based
+ Datasets
+ Evaluation
## References
[1] Computer Vision A Mordern Approach(second edition); David A. Forsyth,  Jean Ponce
