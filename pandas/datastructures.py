import numpy as np
import pandas as pd

grades = { "Joel": 80, "Tim": 95 }

series1 = pd.Series(grades)
series1

series2 = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
series2

npArray = np.array([1,2,3])
npArray