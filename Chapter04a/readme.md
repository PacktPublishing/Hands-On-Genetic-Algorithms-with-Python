# Status of project:

## First try:

Implemented roundtrip puzzle as TSP.

Result: Representation as TSP does not reliably converge to optimum (within a reasonable search space and parameters).

Implemented roundtrip puzzle alternatively as:
- Restate problem as a fixed locus encoding with an alphabet of 0 to 3 (up, right, down, left). Some thought on fitness function required (e.g. longest chain, sum of full chains, ...)
- Hybrid model incorporating above encoding but limiting local options to avoid known restrictions (e.g. margins or holes).
- Calculate fitness as length of the longest coherent path, trying each tile as starting point.

Result: Improved convergence, though calculation of fitness function is maddeningly slow (many iterations required.

- Implemented multiprocessing.

Result: Speed improves though optimum is still not found. 

Conclusion: Problem might have too many local optima for simple GA to solve. Retry with island or ant model.