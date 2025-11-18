# TOE
The Trade-Off Evaluator (TOE) calculating the trade-off extent between two traits based on Pareto-Frontier method
The TOE needs these packages installed:
1. sys
2. cv2
3. pandas
4. numpy
5. matplotlib
6. scipy

The users could run TOE by 
python TOE.py input.csv trait1 trait2 output_prefix
input.csv is the input file recording traits
trait1 and trait2 are the column names in the input.csv representing two traits
output_prefix is the prefix of result files in which output_prefixtrait1_trait2_pareto_frontier_pheno_2.csv is the file recording trade-off extent between two traits.
