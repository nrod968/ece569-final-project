# ece569-final-project


Steps for setting up test files in virtual machine (ece569_project folder)
1. Place ECE569Project folder in labs (like a hw)
2. Place CMakeLists.txt in labs  (override whichever one you have in there already)
3. Place run_proj.slurm file in build_dir
4. Create Project_output folder in build_dir

From cmd in build_dir:
1. module load cuda11/11.0
2. CC=gcc cmake3 ../labs
3. make
4. chmod 777 -R run_proj.slurm
5. srun run_proj.slurm

If done successfully, you should see outc.pbm created in build_dir.

Add your kernels to kernel.cu
Add any utility functions to util.cu
Add your kernel call to solution.cu
