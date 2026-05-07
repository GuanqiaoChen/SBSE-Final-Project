from UCLM_SIMD.genetic.abstract_genetic.genetic_executer import GeneticExecuter


class NSGAIIExecuter(GeneticExecuter):
    def __init__(self, algorithm, execs):
        from UCLM_SIMD.genetic.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
        super().__init__(algorithm,execs)
        self.algorithm: NSGAIIAlgorithm
        self.algorithm_type = "nsgaii"
