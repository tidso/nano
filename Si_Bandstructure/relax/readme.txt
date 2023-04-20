In this folder, you will perform the geometry relaxation.
In INCAR, this means ISIF=3 (the volume of the cell is allowed to change as VASP tries to minimize the energy).

INCAR, POSCAR, KPOINTS, and si_relax_jobscript do not need any changes. 
Copy-paste the contents of each into your own files on PUTTY.

For POTCAR, you will need to copy the Si potential from Dr. Peng's folder.

Once you do this, run the jobscript (sbatch) and wait for the output.
When the job has finished, open the "band" folder and proceed to the next "readme" document.