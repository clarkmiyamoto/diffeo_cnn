```
singularity exec --overlay diffeo_singularity.ext3:rw /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash

source /ext3/env.sh
```
First line activates singularity environment.
Second line actviates conda in the singularity environment
