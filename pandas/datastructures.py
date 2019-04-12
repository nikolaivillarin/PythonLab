import numpy as np
import pandas as pd

grades = { "Joel": 80, "Tim": 95 }

series1 = pd.Series(grades)
series1

series2 = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
series2

npArray = np.array([1, 2, 5, 7, 8])
npArray[1:3] = -1
npArray

npOnes = np.ones((3, 4), dtype=np.int16)
npOnes

npFull = np.full((3, 4), 0.11)
npFull

npArrange = np.arange(10, 30, 2)
npArrange

npLinspace = np.linspace(0, 6, 3)
npLinspace

npRandom = np.random.rand(2, 3)
npRandom

npEmpty = np.empty((2, 3))
npEmpty

npSliceOriginal = np.array([1,2,5,7,8])
npSlice = npSliceOriginal[1:5]
npSlice[1] = 1000
npSlice

npSliceCopy = npSliceOriginal[1:5].copy()
npSliceCopy[1] = 3000
npSliceCopy
npSlice

npBooleanIndexing = np.arange(12).reshape(3,4)
npBooleanIndexing

rows_on = np.array([True, False, True])
rows_on

npBooleanIndexing[rows_on, :]

people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0,3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"])
}
people_dict

people = pd.DataFrame(people_dict)
people

people[people["birthyear"] < 1990]