import unittest
import sys
import os

# Aggiungi la directory principale al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    # Scopri tutti i test nella cartella test
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.dirname(os.path.abspath(__file__)),
        pattern='*_test.py'
    )

    # Esegui i test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)