<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});
</script>
<script type="text/javascript"
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# Machine Learning Engineer Nanodegree

## Proposal
Xin Chen  
August 21st, 2017

## How to obtain the datasets

1. QM7: [http://quantum-machine.org/datasets/#qm7][1]
2. GDB-9: [https://www.nature.com/articles/sdata201422][2]
3. $\mathrm{C}_9\mathrm{H}_7\mathrm{N}$
	* `pip install ase>=3.12`

	from ase.db import connect
	db = connect('C9H7N.PBE.db')
	atoms = db.get_atoms('id=1')
	print(atoms.get_total_energy())	



[1]:	http://quantum-machine.org/datasets/#qm7
[2]:	https://www.nature.com/articles/sdata201422