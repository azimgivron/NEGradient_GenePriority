from typing import List
import numpy as np
import pytest
import scipy.sparse as sp
from NEGradient_GenePriority import Evaluation, Results, ModelEvaluationCollection

@pytest.fixture(name="alphas", autouse=True)
def set_alphas():
    """Fixture: Set alphas for BEDROC"""
    alphas = [228.5, 160.9, 32.2, 16.1, 5.3]
    alpha_map = {228.5: "100", 160.9: "1%", 32.2: "5%", 16.1: "10%", 5.3: "30%"}
    Evaluation.alphas = alphas
    Evaluation.alpha_map = alpha_map
    return alphas
    
@pytest.fixture(name="diseases")
def get_diseases() -> int:
    """Fixture: Returns the number of diseases."""
    return 5

@pytest.fixture(name="genes")
def get_genes() -> int:
    """Fixture: Returns the number of genes."""
    return 3

@pytest.fixture(name="results")
def create_results(genes: int, diseases: int) -> List[Results]:
    """Fixture: Creates and returns Results data for testing."""
    np.random.seed(42)
    y_true = sp.coo_matrix(
        (np.ones(5), (np.hstack((np.zeros(3), np.ones(2))), np.array([0, 1, 2, 3, 4]))),
        shape=(genes, diseases),
    ).tocsr()
    y_pred1 = np.random.rand(genes, diseases)
    y_pred2 = np.random.rand(genes, diseases)
    res1 = Results(y_true, y_pred1)
    res2 = Results(y_true, y_pred2)
    return [res1, res2]

@pytest.fixture(name="evaluation")
def create_evaluation(results: List[Results]) -> Evaluation:
    """Fixture: Creates and returns an Evaluation instance."""
    return Evaluation(results)

@pytest.fixture(name="second_evaluation")
def create_second_evaluation(results: List[Results]) -> Evaluation:
    """Fixture: Creates and returns a second Evaluation instance."""
    return Evaluation(results)

def test_model_evaluation_collection_init(evaluation, second_evaluation):
    """Test: Initializes ModelEvaluationCollection and verifies its models."""
    mec = ModelEvaluationCollection({"model1": evaluation, "model2": second_evaluation})
    assert mec.model_results["model1"] == evaluation
    assert mec.model_results["model2"] == second_evaluation

@pytest.fixture(name="model_evaluation_collection")
def create_model_evaluation_collection(evaluation, second_evaluation) -> ModelEvaluationCollection:
    """Fixture: Creates and returns a ModelEvaluationCollection instance."""
    return ModelEvaluationCollection({"model1": evaluation, "model2": second_evaluation})

def test_properties(model_evaluation_collection):
    """Test: Verifies the properties of ModelEvaluationCollection."""
    mec = model_evaluation_collection
    assert all(isinstance(elem, str) for elem in mec.model_names)
    assert all(isinstance(elem, Evaluation) for elem in mec.evaluations)

def test_items(model_evaluation_collection):
    """Test: Verifies the items method of ModelEvaluationCollection."""
    mec = model_evaluation_collection
    assert all(isinstance(key, str) and isinstance(val, Evaluation) for key, val in mec.items())
    
def test_iter(model_evaluation_collection):
    """Test: Verifies the iterator method of ModelEvaluationCollection."""
    mec = model_evaluation_collection
    assert all(isinstance(elem, Evaluation) for elem in mec)

def test_auc_loss(model_evaluation_collection):
    """Test: Verifies the computation of AUC losses."""
    loss = model_evaluation_collection.compute_auc_losses()
    assert isinstance(loss, np.ndarray)
    assert loss.shape == (len(model_evaluation_collection.evaluations), 2)

def test_bedroc(model_evaluation_collection, diseases, alphas):
    """Test: Verifies the computation of BEDROC scores."""
    bedroc = model_evaluation_collection.compute_bedroc_scores()
    assert isinstance(bedroc, np.ndarray)
    assert bedroc.shape == (len(alphas), diseases, len(model_evaluation_collection.evaluations))
