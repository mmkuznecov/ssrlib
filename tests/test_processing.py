import unittest
import numpy as np

from sslib.processing.covariance import CovarianceProcessor
from sslib.processing.zca import ZCAProcessor


class TestProcessors(unittest.TestCase):
    """Test processor classes."""
    
    def test_covariance_processor(self):
        processor = CovarianceProcessor()
        
        # Create test embeddings
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        
        # Process
        result = processor.process(embeddings)
        
        # Check shape
        self.assertEqual(result.shape, (10, 10))
        
        # Check that it's symmetric
        np.testing.assert_allclose(result, result.T)
        
        # Check metadata
        metadata = processor.get_metadata()
        self.assertEqual(metadata['input_shape'], (100, 10))
        self.assertEqual(metadata['output_shape'], (10, 10))
        
    def test_zca_processor(self):
        processor = ZCAProcessor(epsilon=1e-6)
        
        # Create test embeddings
        np.random.seed(42)
        embeddings = np.random.randn(50, 8)
        
        # Process
        result = processor.process(embeddings)
        
        # Check shape
        self.assertEqual(result.shape, (50, 8))
        
        # Check that whitened data has approximately identity covariance
        cov = np.cov(result.T)
        expected_cov = np.eye(8)
        
        # Should be close to identity (within numerical precision)
        np.testing.assert_allclose(cov, expected_cov, atol=1e-10, rtol=1e-10)
        
        # Check metadata
        metadata = processor.get_metadata()
        self.assertEqual(metadata['input_shape'], (50, 8))
        self.assertEqual(metadata['output_shape'], (50, 8))
        self.assertEqual(metadata['epsilon'], 1e-6)
        
    def test_zca_processor_edge_cases(self):
        processor = ZCAProcessor()
        
        # Test with 1D input (should fail)
        with self.assertRaises(ValueError):
            processor.process(np.random.randn(10))
            
        # Test with very small dataset
        small_embeddings = np.random.randn(2, 3)
        result = processor.process(small_embeddings)
        self.assertEqual(result.shape, (2, 3))
