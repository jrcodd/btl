Memory Usage (G)    Timestep
13.5                0
13.7                136192
13.7                234496
13.8                291840
13.9                423936
14.0                1446912
14.4                2141184
----------------------------------
GPU Usage is overall very low. ~3-5% memory utilization

It takes about 5 minutes to go through 50k timesteps = about 17 hours to finish 10M timesteps
with a 20-30% increase by using more cpu threads: 
1 min for 10000 timesteps -> 1 min for 1200-1300 timesteps or 0.77-0.83 mins for 10000 timesteps
13-14 hours training time

fps is around 153+\-3 for state based and drops over time
Very high cpu utilization ~88%+ for all 8 instances but can use up to 16 threads see above for estimated performance increase (memory and the fact that 2 threads do not perform as well on the same core as seperate cores which prevents a full 2x performance boost)

learning rate is staying around 0.0002 but slowly decreasing over time

memory fluxuating a lot around 2.1M timesteps 14.5-16gb

memory usage is increasing over time indicating a possible memory leak

Image Based:

slightly Lower fps - probably will take a little longer
using 14 threads instead of 8 to speed up training
should get a higher quality model...

even after about a million timesteps image based(RGBD) seems to be doing a lot better