import numpy as np
import pandas as pd

grades = { "Joel": 80, "Tim": 95 }

series1 = pd.Series(grades)
series1

series2 = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
series2

npArray = np.array([1,2,3])
npArray

npOnes = np.ones((3, 4), dtype=np.int16)
npOnes

npFull = np.full((3, 4), 0.11)
npFull