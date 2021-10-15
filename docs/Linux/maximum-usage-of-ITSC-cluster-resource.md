## Maximum Usage of ITSC Cluster Resource

!!! info
    Contributed by [masterzhen119](https://github.com/masterzhen119) on 2021-10-14.

As PhD members in Statistics, we have two types of ITSC source.

1. University level: Everyone owns 10 running jobs without limitation of cores simutanously.
2. Division level: Members from Statistic division could run 30 cores (or 30 jobs) at most.

Sometimes, when you have multiple tests in parallel like 100. Each one will take more than 1 hour. The wise thought is to use both resource of University  and Division level.
Here comes the way: we want to use bash file to allocate first 10 jobs to University level and the rest for STAT.
The bash file is:

```bash
#!/bin/bash
dir="/lustre/project/Stat/path/of/names/of/genes"
cd /lustre/project/Stat/code/job/path/
num=1
c=11
for i in $(ls $dir)
do
    if (($num < $c))
         then
             sbatch  --export=p=${i} try3.job
             echo $num
         else
             sbatch -p stat -q stat --export=p=${i} try3.job
    fi
    pwd
    num=$[$num+1]
    sleep 1
done
```

In this way, we can pass the gene name to job file, and the corresponding job file is:

```bash
#!/bin/bash
#SBATCH -J job_name
#SBATCH -N 1 -c 1
#SBATCH --output= job_name-%j-%a_out.log
#SBATCH --error= job_name-%j-%a_err.log

cd /lustre/project/Statcode/job/path/
python3  example.py $p
```

We have pass the name to python. The rest thing you need to do :

```python
import sys
gene_name = sys.argv[1]
```
In such way, we can run the same python script via system environment. Amazing!
