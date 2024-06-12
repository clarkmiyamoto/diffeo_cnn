# diffeo_cnn

# Installation
Turn on HPC Environment
```
module purge
module load python/intel/3.8.6 
```

Create Python environment
```
cd <THIS REPO>
python -m venv myenv
```

Activate Python enviroment
```
cd <THIS REPO>
source myenv/bin/activate
```

Pip install (could take up to 5 min)
```
source myenv/bin/activate
pip install -r requirements.txt
```

# Load Virtual Environment
Assumes environment is already created
```
cd <THIS REPO>

module purge
module load python/intel/3.8.6 
source myenv/bin/activate
```

# Load Conda Environment
```
module purge
module load anaconda3/2024.02

cd <THIS REPO>
source activate ./penv
```

# SBATCH Example

```
module purge;
module load python/intel/3.8.6;
source myenv/bin/activate
```
