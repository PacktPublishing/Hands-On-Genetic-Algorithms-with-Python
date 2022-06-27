# Status of project:

## First try:

Implemented roundtrip puzzle as TSP.

Result: Representation as TSP does not reliably converge to optimum (within a reasonable search space and parameters).

Potential remedies:
- Restate problem as a fixed locus encoding with an alphabet of 0 to 3 (up, right, down, left). Some thought on fitness function required (e.g. longest chain, sum of full chains, ...)
- Hybrid model incorporating above encoding but limiting local options to avoid known restrictions (e.g. margins or holes).